"""
Provides a sample based predictive posterior implementation,
:class:`SampleBasedPredictivePosterior`.
"""
from typing import Optional, Dict, Tuple, List, Union
from math import ceil
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from bayes_dip.inference.utils import is_invalid
from .base_predictive_posterior import BasePredictivePosterior
from .utils import yield_padded_batched_images_patches, get_image_patch_mask_inds, is_invalid
from ..utils import cg

def predictive_cov_image_patch_norm(v : Tensor, predictive_cov_image_patch : Tensor) -> Tensor:
    """
    Return the norm ``< cov^(-1) v , v >``.

    Parameters
    ----------
    v : Tensor
        Vector(s). Shape: ``(*, num_pixels)``.
    predictive_cov_image_patch : Tensor
        Covariance matrix/matrices. Shape: ``(*, num_pixels, num_pixels)``.

    Returns
    -------
    norm : Tensor
        Norm. Shape: ``(*,)``.
    """
    v_out = torch.linalg.solve(predictive_cov_image_patch, v)
    norm = torch.sum(v * v_out, dim=-1)
    return norm

def predictive_cov_image_patch_log_prob_unscaled_batched(
        recon_masked : Tensor,
        ground_truth_masked : Tensor,
        predictive_cov_image_patch : Tensor) -> Tensor:
    """
    Return the log probabilities for a batch of patches.

    Parameters
    ----------
    recon_masked : Tensor
        Reconstruction patches. Shape: ``(batch_size, num_pixels)``.
    ground_truth_masked : Tensor
        Ground truth patches. Shape: ``(batch_size, num_pixels)``.
    predictive_cov_image_patch : Tensor
        Predictive posterior covariance for the patches.
        Shape: ``(batch_size, num_pixels, num_pixels)``.

    Returns
    -------
    log_prob_unscaled : Tensor
        Log probabilities of the patches.
        Shape: ``(batch_size,)``.
    """

    slogdet = torch.slogdet(predictive_cov_image_patch)
    assert torch.all(slogdet[0] > 0.)
    log_det = slogdet[1]
    diff = (ground_truth_masked - recon_masked).view(ground_truth_masked.shape[0], -1)
    norm = predictive_cov_image_patch_norm(diff, predictive_cov_image_patch)
    log_prob_unscaled = (
            -0.5 * norm - 0.5 * log_det +
            -0.5 * np.log(2. * np.pi) * np.prod(ground_truth_masked.shape[1:]))
    return log_prob_unscaled

def approx_predictive_cov_image_patch_from_zero_mean_samples_batched(
        samples: Tensor, noise_x_correction_term: Optional[float] = None) -> Tensor:
    """
    Estimate the (co-)variances of image pixels from zero mean samples.

    Parameters
    ----------
    samples : Tensor
        Image (or patch) samples with mean zero.
        Shape: ``(batch_size, mc_samples, num_pixels)``.
    noise_x_correction_term : float, optional
        If specified, this value is added to the diagonal of each covariance matrix.

    Returns
    -------
    cov : Tensor
        Covariance estimate, shape: ``(batch_size, num_pixels, num_pixels)``.
    """

    batch_size, mc_samples, im_numel = samples.shape
    samples = samples.view(batch_size * mc_samples, -1)
    samples = samples * (mc_samples ** -0.5)

    prods = torch.bmm(samples[:, :, None], samples[:, None, :]).view(
            batch_size, mc_samples, im_numel, im_numel)
    cov = prods.sum(dim=1) # image x image

    if noise_x_correction_term is not None:
        cov[(slice(None), *np.diag_indices(im_numel))] += noise_x_correction_term

    return cov

def yield_covariances_patches(
        samples: Tensor,
        patch_kwargs: Optional[Dict] = None,
        noise_x_correction_term: Optional[float] = 1e-6,
        device = None) -> Tensor:
    """
    Yield posterior covariance matrices for image patches.

    Parameters
    ----------
    samples : Tensor
        Precomputed samples, e.g. drawn by :meth:`sample_zero_mean`.
    patch_kwargs : dict, optional
        Keyword arguments specifying the patches, see docs of :meth:`log_prob`.
    noise_x_correction_term : float or None, optional
        Noise amount that is assumed to be present in ground truth. The default is ``1e-6``.
    device : str or torch.device, optional
        Device. If ``None`` (the default), ``samples.device`` is used.

    Yields
    ------
    batch_patch_inds : list of int
        Indices of the patches (for the currently yielded batch).
    batch_predictive_cov_image_patch : Tensor
        Covariance matrices.
        Shape: ``(batch_size, max(batch_len_mask_inds), max(batch_len_mask_inds))``, where
        ``batch_size`` is ``patch_kwargs['batch_size']`` for all batches except for the
        potentially shorter last batch. If a patch has less than ``max(batch_len_mask_inds)``
        pixels, the covariance matrix is padded with the identity, i.e. ones on the diagonal and
        zeros for the other entries.
    batch_len_mask_inds : list of int
        Numbers of pixels in the patches.
    """
    # pylint: disable=too-many-arguments

    device = samples.device if device is None else device

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

def log_prob_patches(
        mean: Tensor,
        ground_truth: Tensor,
        samples: Tensor,
        patch_kwargs: Optional[Dict] = None,
        reweight_off_diagonal_entries: bool = False,
        noise_x_correction_term: Optional[float] = 1e-6,
        verbose: bool = True,
        unscaled: bool = False,
        return_patch_diags: bool = False,
        device = None
        ) -> Union[List[float], Tuple[List[float], List[Tensor]]]:
    """
    Return log probabilities for patches.

    Parameters
    ----------
    mean : Tensor
        Mean of the posterior image distribution.
    ground_truth : Tensor
        Ground truth.
    samples : Tensor
        Precomputed samples, e.g. drawn by :meth:`sample_zero_mean`.
    patch_kwargs : dict, optional
        Keyword arguments specifying the patches, see docs of :meth:`log_prob`.
    reweight_off_diagonal_entries : bool, optional
        If ``True``, replace the covariance matrix ``cov`` (for each patch) with
        ``0.5 * (cov + torch.diag(torch.diag(cov)))``.
        The default is ``False``.
    noise_x_correction_term : float or None, optional
        Noise amount that is assumed to be present in ground truth. Can help to stabilize
        computations. The default is ``1e-6``.
    verbose : bool, optional
        Whether to print information. The default is ``True``.
    unscaled : bool, optional
        If ``False``, the unscaled patch log probabilities are divided by the number of pixels in
        the respective patch. Otherwise the unscaled patch log probabilities are returned.
        The default is ``False``.
    return_patch_diags : bool, optional
        If ``True``, return the diagonals of the covariance matrices.
        The default is ``False``.
    device : str or torch.device, optional
        Device. If ``None`` (the default), ``'cuda:0'`` is chosen if available or ``'cpu'``
        otherwise.

    Returns
    -------
    log_probabilities : list of float
        Log probabilities for the patches, optionally scaled; see the ``unscaled`` argument.
    patch_diags : list of Tensor, optional
        Diagonals of the covariance matrices for the patches.
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    patch_kwargs = patch_kwargs or {}
    patch_kwargs.setdefault('patch_size', 1)

    log_probs = []
    patch_diags = []
    all_patch_mask_inds = get_image_patch_mask_inds(
            ground_truth.shape[2:], patch_size=patch_kwargs['patch_size'])
    for batch_patch_inds, batch_predictive_cov_image_patch, batch_len_mask_inds in (
            yield_covariances_patches(
                    samples=samples,
                    patch_kwargs=patch_kwargs,
                    noise_x_correction_term=noise_x_correction_term,
                    device=device)):

        if reweight_off_diagonal_entries and patch_kwargs['patch_size'] > 1:

            # The re-weighting of the off-diagonal entries is proposed by Wesley J. Maddox
            # in [1]_.

            # .. [1] Maddox, W.J., Izmailov, P., Garipov, T., Vetrov, D.P. and Wilson, A.G.,
            #         2019, "A simple baseline for bayesian uncertainty in deep learning".
            #         Advances in Neural Information Processing Systems, 32.
            #         https://arxiv.org/pdf/1902.02476.pdf

            batch_predictive_cov_image_patch = 0.5 * (
                batch_predictive_cov_image_patch + torch.diag_embed(
                    torch.diagonal(batch_predictive_cov_image_patch,
                            dim1=-2, dim2=-1), dim1=-2, dim2=-1))

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

class SampleBasedPredictivePosterior(BasePredictivePosterior):
    """
    Approximate sample-based predictive posterior.
    """

    # sample_via_matheron
    def sample_zero_mean(self,
        num_samples: int,
        cov_obs_mat_chol: Optional[Tensor] = None,
        batch_size: int = 1,
        use_conj_grad_inv: bool = False,
        cg_kwargs: Optional[Dict] = None,
        return_residual_norm_list: bool = False,
        return_on_device = None,
        ) -> Tensor:
        """
        Return samples from the Gaussian given by the predictive posterior covariance and mean zero.

        Note that, in contrast to the (abstract) :meth:`sample` method designated by
        :class:`BasePredictivePosterior`, this function does not include an image noise correction
        term (and always has mean zero as indicated by the name).

        Parameters
        ----------
        num_samples : int
            Number of samples.
        cov_obs_mat_chol : Tensor, optional
            Cholesky factor of the observation covariance matrix.
            Required if ``not use_conj_grad_inv``.
        batch_size : int, optional
            Batch size (number of images per batch). The default is ``1``.
        use_conj_grad_inv : bool, optional
            Whether to use CG instead of ``cov_obs_mat_chol`` for solving the linear system with the
            observation covariance matrix. The default is ``False``.
        cg_kwargs : dict, optional
            Keyword arguments passed to :func:`bayes_dip.utils.cg`.
        return_residual_norm_list : bool, optional
            Whether to return the list of residual norms in case of ``use_conj_grad_inv``.
            The default is ``False``.
        return_on_device : str or torch.device, optional
            Device on which samples are collected. This option only affects the storage of the
            samples to be returned, not their computation.
            If ``None`` (the default), ``self.observation_cov.device`` is used.

        Returns
        -------
        samples : Tensor
            Samples from the Gaussian given by the predictive posterior covariance and mean zero.
            Shape: ``(n, 1, *im_shape)``, where ``n`` is
            ``ceil(num_samples / batch_size) * batch_size``.
        residual_norm_list : list of scalar, optional
            Residual norms of CG solutions, only returned if
            ``use_conj_grad_inv and return_residual_norm_list``.
        """
        # pylint: disable=arguments-differ
        # pylint: disable=too-many-locals

        num_batches = ceil(num_samples / batch_size)
        image_samples = []
        residual_norm_list = []
        assert use_conj_grad_inv or cov_obs_mat_chol is not None
        cg_kwargs = cg_kwargs or {}
        return_on_device = (
                self.observation_cov.device if return_on_device is None else return_on_device)
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc='sample_via_matheron',
                    miniters=num_batches//100):

                x_samples = self.observation_cov.image_cov.sample(
                    num_samples=batch_size,
                    return_weight_samples=False
                    )
                samples = self.observation_cov.trafo(x_samples)

                noise_term = (self.observation_cov.log_noise_variance.exp()**.5) * torch.randn_like(
                        samples)

                samples = (noise_term - samples).view(batch_size, -1)

                if not use_conj_grad_inv:
                    samples = torch.linalg.solve_triangular(
                        cov_obs_mat_chol.T, torch.linalg.solve_triangular(
                            cov_obs_mat_chol, samples.T, upper=False), upper=True).T
                else:
                    def observation_cov_closure(v):
                        return self.observation_cov(v.T.reshape(
                                batch_size, 1, *self.observation_cov.trafo.obs_shape)).view(
                                        batch_size, self.observation_cov.shape[0]).T
                    samples_T, residual_norm = cg(
                            observation_cov_closure, samples.T, **cg_kwargs)
                    residual_norm_list.append(residual_norm)
                    samples = samples_T.T

                delta_x = self.observation_cov.trafo.trafo_adjoint(samples.view(
                        batch_size, 1, *self.observation_cov.trafo.obs_shape))
                delta_x = self.observation_cov.image_cov(delta_x)
                image_samples.append((x_samples + delta_x).to(device=return_on_device))
            image_samples = torch.cat(image_samples, axis=0)

        return (
                image_samples if not (use_conj_grad_inv and return_residual_norm_list)
                else (image_samples, residual_norm_list))

    def yield_covariances_patches(self,
            samples: Tensor = None,
            patch_kwargs: Optional[Dict] = None,
            noise_x_correction_term: Optional[float] = 1e-6,
            sample_kwargs: Optional[Dict] = None,
            device = None) -> Tensor:
        """
        Yield posterior covariance matrices for image patches.

        Parameters
        ----------
        samples : Tensor, optional
            Precomputed samples with mean zero, e.g. drawn by :meth:`sample_zero_mean`.
            If not specified, ``samples_kwargs['num_samples']`` samples are drawn in this function.
        patch_kwargs : dict, optional
            Keyword arguments specifying the patches, see docs of :meth:`log_prob`.
        noise_x_correction_term : float or None, optional
            Noise amount that is assumed to be present in ground truth. The default is ``1e-6``.
        sample_kwargs : dict, optional
            Keyword arguments passed to :meth:`sample_zero_mean`. Required if ``samples is None``.
        device : str or torch.device, optional
            Device. If ``None`` (the default), ``self.observation_cov.device`` is used.

        Yields
        ------
        batch_patch_inds : list of int
            Indices of the patches (for the currently yielded batch).
        batch_predictive_cov_image_patch : Tensor
            Covariance matrices.
            Shape: ``(batch_size, max(batch_len_mask_inds), max(batch_len_mask_inds))``, where
            ``batch_size`` is ``patch_kwargs['batch_size']`` for all batches except for the
            potentially shorter last batch. If a patch has less than ``max(batch_len_mask_inds)``
            pixels, the covariance matrix is padded with the identity, i.e. ones on the diagonal and
            zeros for the other entries.
        batch_len_mask_inds : list of int
            Numbers of pixels in the patches.
        """
        # pylint: disable=too-many-arguments

        device = self.observation_cov.device if device is None else device
        if samples is None:
            sample_kwargs = sample_kwargs or {}
            sample_kwargs.setdefault('return_on_device', 'cpu')
            samples = self.sample_zero_mean(**sample_kwargs)

        yield from yield_covariances_patches(samples=samples, patch_kwargs=patch_kwargs,
            noise_x_correction_term=noise_x_correction_term, device=device)

    def log_prob(self,
            mean: Tensor,
            ground_truth: Tensor,
            noise_x_correction_term: Optional[float] = 1e-6,
            patch_kwargs: Optional[Dict] = None,
            unscaled: bool = False,
            **kwargs
            ) -> np.float64:
        """
        Return the patch-based approximate log probability.

        By default, a patch size of ``1`` pixel, i.e. just the pixel-wise variance is used,
        neglecting correlations between different pixels.

        Parameters
        ----------
        mean : Tensor
            Mean of the posterior image distribution.
        ground_truth : Tensor
            Ground truth.
        noise_x_correction_term : float or None, optional
            Noise amount that is assumed to be present in ground truth. Can help to stabilize
            computations. The default is ``1e-6``.
        patch_kwargs : dict, optional
            Keyword arguments specifying how to split the image into patches.

            The arguments are:
                ``'patch_size'`` : int, optional
                    The default is ``1``.
                ``'patch_idx_list'`` : list of int, optional
                    Patch indices. If ``None``, all patches are used.
                ``'batch_size'`` : int, optional
                    The default is ``1``.
        unscaled : bool, optional
            If ``False``, the sum of the (unscaled) patch log probabilities is divided by the total
            number of pixels in the patches.
            Otherwise, the sum of the unscaled patch log probabilities is returned.
            The default is ``False``.
        kwargs : dict, optional
            Keyword arguments forwarded to :meth:`log_prob_patches`.

        Returns
        -------
        log_probability : np.float64
            Log probability, optionally scaled; see the ``unscaled`` argument.
        """
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
            samples: Tensor = None,
            patch_kwargs: Optional[Dict] = None,
            reweight_off_diagonal_entries: bool = False,
            noise_x_correction_term: Optional[float] = 1e-6,
            verbose: bool = True,
            unscaled: bool = False,
            return_patch_diags: bool = False,
            sample_kwargs: Optional[Dict] = None,
            ) -> Union[List[float], Tuple[List[float], List[Tensor]]]:
        """
        Return log probabilities for patches.

        Parameters
        ----------
        mean : Tensor
            Mean of the posterior image distribution.
        ground_truth : Tensor
            Ground truth.
        samples : Tensor, optional
            Precomputed samples with mean zero, e.g. drawn by :meth:`sample_zero_mean`.
            If not specified, ``samples_kwargs['num_samples']`` samples are drawn in this function.
        patch_kwargs : dict, optional
            Keyword arguments specifying the patches, see docs of :meth:`log_prob`.
        reweight_off_diagonal_entries : bool, optional
            If ``True``, replace the covariance matrix ``cov`` (for each patch) with
            ``0.5 * (cov + torch.diag(torch.diag(cov)))``.
            The default is ``False``.
        noise_x_correction_term : float or None, optional
            Noise amount that is assumed to be present in ground truth. Can help to stabilize
            computations. The default is ``1e-6``.
        verbose : bool, optional
            Whether to print information. The default is ``True``.
        unscaled : bool, optional
            If ``False``, the unscaled patch log probabilities are divided by the number of pixels
            in the respective patch. Otherwise the unscaled patch log probabilities are returned.
            The default is ``False``.
        return_patch_diags : bool, optional
            If ``True``, return the diagonals of the covariance matrices.
            The default is ``False``.
        sample_kwargs : dict, optional
            Keyword arguments passed to :meth:`sample_zero_mean`. Required if ``samples is None``.

        Returns
        -------
        log_probabilities : list of float
            Log probabilities for the patches, optionally scaled; see the ``unscaled`` argument.
        patch_diags : list of Tensor, optional
            Diagonals of the covariance matrices for the patches.
        """
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        patch_kwargs = patch_kwargs or {}
        patch_kwargs.setdefault('patch_size', 1)

        if samples is None:
            sample_kwargs = sample_kwargs or {}
            sample_kwargs.setdefault('return_on_device', 'cpu')
            samples = self.sample_zero_mean(**sample_kwargs)

        return log_prob_patches(
                mean=mean,
                ground_truth=ground_truth,
                samples=samples,
                patch_kwargs=patch_kwargs,
                reweight_off_diagonal_entries=reweight_off_diagonal_entries,
                noise_x_correction_term=noise_x_correction_term,
                verbose=verbose,
                unscaled=unscaled,
                return_patch_diags=return_patch_diags,
                device=self.observation_cov.device
            )
