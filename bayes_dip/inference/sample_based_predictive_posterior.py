from typing import Optional, Dict
from math import ceil
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from bayes_dip.inference.utils import is_invalid
from .base_predictive_posterior import BasePredictivePosterior
from .utils import yield_padded_batched_images_patches, get_image_patch_mask_inds, is_invalid
from ..utils import cg

def predictive_cov_image_patch_norm(v, predictive_cov_image_patch):
    v_out = torch.linalg.solve(predictive_cov_image_patch, v)
    norm = torch.sum(v * v_out, dim=-1)
    return norm

def predictive_cov_image_patch_log_prob_unscaled_batched(
        recon_masked, ground_truth_masked, predictive_cov_image_patch):

    approx_slogdet = torch.slogdet(predictive_cov_image_patch)
    assert torch.all(approx_slogdet[0] > 0.)
    approx_log_det = approx_slogdet[1]
    diff = (ground_truth_masked - recon_masked).view(ground_truth_masked.shape[0], -1)
    norm = predictive_cov_image_patch_norm(diff, predictive_cov_image_patch)
    approx_log_prob_unscaled = (
            -0.5 * norm - 0.5 * approx_log_det +
            -0.5 * np.log(2. * np.pi) * np.prod(ground_truth_masked.shape[1:]))
    return approx_log_prob_unscaled

def approx_predictive_cov_image_patch_from_zero_mean_samples_batched(
        samples, noise_x_correction_term=None):

    batch_size, mc_samples, im_numel = samples.shape
    samples = samples.view(batch_size, mc_samples, -1)  # batch x samples x image
    samples = samples.view(batch_size * mc_samples, -1)

    prods = torch.bmm(samples[:, :, None], samples[:, None, :]).view(
            batch_size, mc_samples, im_numel, im_numel)
    cov = prods.sum(dim=1) / prods.shape[1]  # image x image

    if noise_x_correction_term is not None:
        cov[(slice(None), *np.diag_indices(im_numel))] += noise_x_correction_term

    return cov

class SampleBasedPredictivePosterior(BasePredictivePosterior):

    # sample_via_matheron
    def sample_zero_mean(self,
        num_samples: int,
        cov_obs_mat_chol: Optional[Tensor] = None,
        vec_batch_size: int = 1,
        use_conj_grad_inv: bool = False,
        cg_kwargs: Optional[Dict] = None,
        return_residual_norm_list: bool = False,
        return_on_device = None,
        ) -> Tensor:
        # pylint: disable=arguments-differ
        # pylint: disable=too-many-locals

        num_batches = ceil(num_samples / vec_batch_size)
        image_samples = []
        residual_norm_list = []
        assert use_conj_grad_inv or cov_obs_mat_chol is not None
        return_on_device = (
                self.observation_cov.device if return_on_device is None else return_on_device)
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc='sample_via_matheron',
                    miniters=num_batches//100):

                x_samples = self.observation_cov.image_cov.sample(
                    num_samples=vec_batch_size,
                    return_weight_samples=False
                    )
                samples = self.observation_cov.trafo(x_samples)

                noise_term = (self.observation_cov.log_noise_variance.exp()**.5) * torch.randn_like(
                        samples)

                samples = (noise_term - samples).view(vec_batch_size, -1)

                if not use_conj_grad_inv:
                    samples = torch.linalg.solve_triangular(
                        cov_obs_mat_chol.T, torch.linalg.solve_triangular(
                            cov_obs_mat_chol, samples.T, upper=False), upper=True).T
                else:
                    def observation_cov_closure(v):
                        return self.observation_cov(v.T.reshape(
                                vec_batch_size, 1, *self.observation_cov.trafo.obs_shape)).view(
                                        vec_batch_size, self.observation_cov.shape[0]).T
                    samples_T, residual_norm = cg(
                            observation_cov_closure, samples.T,
                            precon_closure=cg_kwargs['precon_closure'],
                            max_niter=cg_kwargs['max_niter'],
                            rtol=cg_kwargs['rtol'],
                            ignore_numerical_warning=cg_kwargs['ignore_numerical_warning']
                        )
                    residual_norm_list.append(residual_norm)
                    samples = samples_T.T

                delta_x = self.observation_cov.trafo.trafo_adjoint(samples.view(
                        vec_batch_size, 1, *self.observation_cov.trafo.obs_shape))
                delta_x = self.observation_cov.image_cov(delta_x)
                image_samples.append((x_samples + delta_x).to(device=return_on_device))
            image_samples = torch.cat(image_samples, axis=0)

        return (
                image_samples if not (use_conj_grad_inv and return_residual_norm_list)
                else (image_samples, residual_norm_list))

    # TODO document, esp. padding with 1. diagonal elements in return value
    def yield_covariances_patches(self,
            num_samples: Optional[int] = 10000,
            samples: Tensor = None,
            patch_kwargs: Optional[Dict] = None,
            cov_obs_mat_chol: Optional[Tensor] = None,
            noise_x_correction_term: float = 1e-6,
            sample_kwargs: Optional[Dict] = None,
            device = None) -> Tensor:
        # pylint: disable=too-many-arguments

        device = self.observation_cov.device if device is None else device
        if samples is None:
            sample_kwargs = sample_kwargs or {}
            samples = self.sample_zero_mean(num_samples=num_samples,
                    cov_obs_mat_chol=cov_obs_mat_chol, return_on_device='cpu', **sample_kwargs)
        for batch_patch_inds, batch_samples_patches, batch_len_mask_inds in (
                yield_padded_batched_images_patches(samples,
                        patch_kwargs=patch_kwargs, return_patch_numels=True)):

            batch_predictive_cov_image_patch = (
                    approx_predictive_cov_image_patch_from_zero_mean_samples_batched(
                            batch_samples_patches.to(device),
                            noise_x_correction_term=noise_x_correction_term))
            # use identity for padding dims in predictive_cov_image_patch
            # (the determinant then is the same as for
            #  predictive_cov_image_patch[:len_mask_inds, :len_mask_inds])
            max_len_mask_inds = max(batch_len_mask_inds)
            for k, len_mask_inds in enumerate(batch_len_mask_inds):
                batch_predictive_cov_image_patch[
                        k,
                        np.arange(len_mask_inds, max_len_mask_inds),
                        np.arange(len_mask_inds, max_len_mask_inds)] = 1.

            batch_invalid_values = is_invalid(batch_predictive_cov_image_patch)
            batch_invalid_values_patch_inds = (
                    torch.tensor(batch_patch_inds)[batch_invalid_values]).tolist()
            if len(batch_invalid_values_patch_inds) > 0:
                raise ValueError(
                        'invalid value occurred in predictive cov for patch indices '
                        f'{batch_invalid_values_patch_inds}')

            yield batch_patch_inds, batch_predictive_cov_image_patch, batch_len_mask_inds

    def log_prob(self,
            mean: Tensor,
            ground_truth: Tensor,
            noise_x_correction_term: float = 1e-6,
            patch_kwargs: Optional[Dict] = None,
            unscaled: bool = False,
            **kwargs):
        # pylint: disable=arguments-differ

        patch_kwargs = patch_kwargs or {}
        patch_kwargs.setdefault('patch_size', 1)
        patch_kwargs.setdefault('patch_idx_list', None)

        sum_log_prob_unscaled = np.sum(self.log_prob_patches(
                mean=mean, ground_truth=ground_truth,
                noise_x_correction_term=noise_x_correction_term,
                patch_kwargs=patch_kwargs, unscaled=True,
                **kwargs))
        if unscaled:
            out = sum_log_prob_unscaled
        else:
            all_patch_mask_inds = get_image_patch_mask_inds(
                    self.observation_cov.trafo.im_shape, patch_size=patch_kwargs['patch_size'])
            if patch_kwargs['patch_idx_list'] is None:
                patch_kwargs['patch_idx_list'] = list(range(len(all_patch_mask_inds)))
            total_num_pixels_in_patches = sum(len(all_patch_mask_inds[patch_idx])
                    for patch_idx in patch_kwargs['patch_idx_list'])
            out = sum_log_prob_unscaled / total_num_pixels_in_patches
        return out

    def log_prob_patches(self,
            mean: Tensor,
            ground_truth: Tensor,
            num_samples: Optional[int] = 10000,
            samples: Tensor = None,
            patch_kwargs: Optional[Dict] = None,
            cov_obs_mat_chol: Optional[Tensor] = None,
            noise_x_correction_term: float = 1e-6,
            verbose: bool = True,
            unscaled: bool = False,
            return_patch_diags: bool = False,
            sample_kwargs: Optional[Dict] = None) -> Tensor:
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        patch_kwargs = patch_kwargs or {}
        patch_kwargs.setdefault('patch_size', 1)

        if samples is None:
            sample_kwargs = sample_kwargs or {}
            samples = self.sample_zero_mean(num_samples=num_samples,
                    cov_obs_mat_chol=cov_obs_mat_chol, return_on_device='cpu', **sample_kwargs)
        log_probs = []
        patch_diags = []
        all_patch_mask_inds = get_image_patch_mask_inds(
                self.observation_cov.trafo.im_shape, patch_size=patch_kwargs['patch_size'])
        for batch_patch_inds, batch_predictive_cov_image_patch, batch_len_mask_inds in (
                self.yield_covariances_patches(
                        samples=samples,
                        patch_kwargs=patch_kwargs,
                        cov_obs_mat_chol=cov_obs_mat_chol,
                        noise_x_correction_term=noise_x_correction_term)):

            if return_patch_diags:
                patch_diags.extend([cov_image_patch.diag()[:len_mask_inds]
                        for cov_image_patch, len_mask_inds in zip(
                                batch_predictive_cov_image_patch, batch_len_mask_inds)])

            max_len_mask_inds = max(batch_len_mask_inds)
            batch_recon = torch.stack([
                    torch.nn.functional.pad(mean.flatten()[all_patch_mask_inds[patch_idx]],
                            (0, max_len_mask_inds - len_mask_inds))
                    for patch_idx, len_mask_inds in zip(batch_patch_inds, batch_len_mask_inds)])
            batch_ground_truth = torch.stack([
                    torch.nn.functional.pad(ground_truth.flatten()[all_patch_mask_inds[patch_idx]],
                            (0, max_len_mask_inds - len_mask_inds))
                    for patch_idx, len_mask_inds in zip(batch_patch_inds, batch_len_mask_inds)])

            batch_patch_log_prob_unscaled = predictive_cov_image_patch_log_prob_unscaled_batched(
                    recon_masked=batch_recon,
                    ground_truth_masked=batch_ground_truth,
                    predictive_cov_image_patch=batch_predictive_cov_image_patch)

            if verbose:
                for k, patch_idx in enumerate(batch_patch_inds):
                    mask_inds = all_patch_mask_inds[patch_idx]
                    patch_log_prob_unscaled = batch_patch_log_prob_unscaled[k]
                    print(
                            f'sample based log prob (scaled) for patch {patch_idx}: '
                            f'{patch_log_prob_unscaled / len(mask_inds)}')

            if unscaled:
                log_probs += batch_patch_log_prob_unscaled.tolist()
            else:
                batch_len_mask_inds = [
                        len(all_patch_mask_inds[patch_idx]) for patch_idx in batch_patch_inds]
                batch_patch_log_prob = batch_patch_log_prob_unscaled / torch.tensor(
                        batch_len_mask_inds, device=batch_patch_log_prob_unscaled.device)
                log_probs += batch_patch_log_prob.tolist()

        return (log_probs, patch_diags) if return_patch_diags else log_probs
