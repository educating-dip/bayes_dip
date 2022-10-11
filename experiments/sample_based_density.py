import os
from warnings import warn
from itertools import islice
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from bayes_dip.utils.experiment_utils import (
        get_standard_ray_trafo, get_standard_dataset, assert_sample_matches,
        get_predefined_patch_idx_list, save_samples, load_samples)
from bayes_dip.utils import PSNR, SSIM, eval_mode
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import (
        get_default_unet_gaussian_prior_dicts, get_default_unet_gprior_dicts)
from bayes_dip.probabilistic_models import (
        get_neural_basis_expansion, LowRankNeuralBasisExpansion, LowRankObservationCov,
        ParameterCov, ImageCov, ObservationCov, get_image_noise_correction_term)
from bayes_dip.marginal_likelihood_optim import LowRankObservationCovPreconditioner
from bayes_dip.inference import SampleBasedPredictivePosterior, get_image_patch_mask_inds


def _save_cov_obs_mat(i: int, cov_obs_mat: Tensor) -> None:
    torch.save(cov_obs_mat.cpu(), f'cov_obs_mat_{i}.pt')


def _load_cov_obs_mat(path: str, i: int, device=None, dtype=None) -> Tensor:
    cov_obs_mat = torch.load(
            os.path.join(path, f'cov_obs_mat_{i}.pt'), map_location=device)
    if dtype == torch.float64 and cov_obs_mat.dtype == torch.float32:
        warn('Loaded cov_obs_mat with dtype float32 while expecting float64; will convert but '
                'results are probably inaccurate or unstable')
    cov_obs_mat = cov_obs_mat.to(dtype=dtype)
    return cov_obs_mat


@hydra.main(config_path='hydra_cfg', config_name='config', version_base='1.2')
def coordinator(cfg : DictConfig) -> None:
    # pylint: disable=too-many-locals,too-many-statements

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    dtype = torch.get_default_dtype()
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    ray_trafo = get_standard_ray_trafo(cfg)
    ray_trafo.to(dtype=dtype, device=device)

    # data: observation, ground_truth, filtbackproj
    dataset = get_standard_dataset(
            cfg, ray_trafo, fold=cfg.dataset.fold, use_fixed_seeds_starting_from=cfg.seed,
            device=device)

    for i, data_sample in enumerate(islice(DataLoader(dataset), cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)

        observation, ground_truth, filtbackproj = data_sample

        # assert that sample data matches with that from [exact_]dip_mll_optim.py
        assert_sample_matches(data_sample, cfg.inference.load_path, i)

        observation = observation.to(dtype=dtype, device=device)
        filtbackproj = filtbackproj.to(dtype=dtype, device=device)
        ground_truth = ground_truth.to(dtype=dtype, device=device)

        net_kwargs = {
                'scales': cfg.dip.net.scales,
                'channels': cfg.dip.net.channels,
                'skip_channels': cfg.dip.net.skip_channels,
                'use_norm': cfg.dip.net.use_norm,
                'use_sigmoid': cfg.dip.net.use_sigmoid,
                'sigmoid_saturation_thresh': cfg.dip.net.sigmoid_saturation_thresh}
        reconstructor = DeepImagePriorReconstructor(
                ray_trafo, torch_manual_seed=cfg.dip.torch_manual_seed,
                device=device, net_kwargs=net_kwargs,
                load_params_path=os.path.join(cfg.inference.load_path, f'dip_model_{i}.pt'))

        with torch.no_grad(), eval_mode(reconstructor.nn_model):
            recon = reconstructor.nn_model(filtbackproj)  # pylint: disable=not-callable

        print(f'DIP reconstruction of sample {i:d}')
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

        prior_assignment_dict, hyperparams_init_dict = (
                get_default_unet_gaussian_prior_dicts(reconstructor.nn_model)
                if not cfg.priors.use_gprior else
                get_default_unet_gprior_dicts(reconstructor.nn_model))
        parameter_cov = ParameterCov(
                reconstructor.nn_model,
                prior_assignment_dict,
                hyperparams_init_dict,
                device=device
        )
        neural_basis_expansion = get_neural_basis_expansion(
                nn_model=reconstructor.nn_model,
                nn_input=filtbackproj,
                ordered_nn_params=parameter_cov.ordered_nn_params,
                nn_out_shape=filtbackproj.shape,
                use_gprior=cfg.priors.use_gprior,
                trafo=ray_trafo,
                scale_kwargs=OmegaConf.to_object(cfg.priors.gprior.scale)
        )
        if cfg.inference.use_low_rank_neural_basis_expansion:
            if cfg.inference.load_samples_from_path is None:
                neural_basis_expansion = LowRankNeuralBasisExpansion(
                    neural_basis_expansion=neural_basis_expansion,
                    batch_size=cfg.inference.low_rank_neural_basis_expansion.batch_size,
                    low_rank_rank_dim=cfg.inference.low_rank_neural_basis_expansion.low_rank_rank_dim,
                    oversampling_param=cfg.inference.low_rank_neural_basis_expansion.oversampling_param,
                    device=device,
                    use_cpu=cfg.inference.low_rank_neural_basis_expansion.use_cpu)
            else:
                print('skipping low-rank neural basis expansion computation, since loading samples')

        image_cov = ImageCov(
                parameter_cov=parameter_cov,
                neural_basis_expansion=neural_basis_expansion
        )
        observation_cov = ObservationCov(
                trafo=ray_trafo,
                image_cov=image_cov,
                device=device
        )
        observation_cov_filename = (
                f'observation_cov_{i}.pt' if cfg.inference.load_iter is None else
                f'observation_cov_{i}_iter_{cfg.inference.load_iter}.pt')
        observation_cov.load_state_dict(
                torch.load(os.path.join(cfg.inference.load_path, observation_cov_filename)))

        noise_x_correction_term = None
        if cfg.inference.add_image_noise_correction_term:
            print('computing noise correction term')
            noise_x_correction_term = get_image_noise_correction_term(observation_cov=observation_cov)
            print('noise_x_correction_term:', noise_x_correction_term)

        cov_obs_mat_chol = None
        if not cfg.inference.sampling.use_conj_grad_inv:
            if cfg.inference.load_cov_obs_mat_from_path is None:
                cov_obs_mat = observation_cov.assemble_observation_cov(
                        batch_size=cfg.inference.cov_obs_mat.batch_size)
            else:
                cov_obs_mat = _load_cov_obs_mat(
                        path=cfg.inference.load_cov_obs_mat_from_path, i=i, device=device, dtype=dtype)
            if cfg.inference.save_cov_obs_mat:
                _save_cov_obs_mat(i=i, cov_obs_mat=cov_obs_mat)
            eps = ObservationCov.get_stabilizing_eps(
                    cov_obs_mat,
                    eps_mode=cfg.inference.cov_obs_mat.eps_mode,
                    eps=cfg.inference.cov_obs_mat.eps,
                    eps_min_for_auto=cfg.inference.cov_obs_mat.eps_min_for_auto,
                    include_zero_for_auto=cfg.inference.cov_obs_mat.include_zero_for_auto)
            print(f'Stabilizing cov_obs_mat eps: {eps}')
            cov_obs_mat[np.diag_indices(cov_obs_mat.shape[0])] += eps
            cov_obs_mat_chol = torch.linalg.cholesky(cov_obs_mat)

        predictive_posterior = SampleBasedPredictivePosterior(observation_cov)

        if cfg.inference.load_samples_from_path is None:
            sample_kwargs = {
                'batch_size': cfg.inference.sampling.batch_size,
                'use_conj_grad_inv': cfg.inference.sampling.use_conj_grad_inv,
                'cg_kwargs': OmegaConf.to_object(cfg.inference.sampling.cg_kwargs),
            }
            update_kwargs = {'batch_size': cfg.inference.sampling.cg_preconditioner.batch_size}
            if cfg.inference.sampling.use_conj_grad_inv:
                low_rank_observation_cov = LowRankObservationCov(
                        trafo=ray_trafo,
                        image_cov=image_cov,
                        low_rank_rank_dim=(
                                cfg.inference.sampling.cg_preconditioner.low_rank_rank_dim),
                        oversampling_param=(
                                cfg.inference.sampling.cg_preconditioner.oversampling_param),
                        requires_grad=False,
                        device=device,
                        **update_kwargs,
                )
                low_rank_preconditioner = LowRankObservationCovPreconditioner(
                        low_rank_observation_cov=low_rank_observation_cov,
                        default_update_kwargs=update_kwargs,
                )
                sample_kwargs['cg_kwargs']['precon_closure'] = low_rank_preconditioner.get_closure()
            samples = predictive_posterior.sample_zero_mean(
                    num_samples=cfg.inference.num_samples,
                    cov_obs_mat_chol=cov_obs_mat_chol,
                    return_on_device='cpu',
                    **sample_kwargs)
        else:
            samples = load_samples(
                    path=cfg.inference.load_samples_from_path, i=i,
                    num_samples=cfg.inference.num_samples)

        if cfg.inference.save_samples:
            save_samples(i=i, samples=samples, chunk_size=cfg.inference.save_samples_chunk_size)

        all_patch_mask_inds = get_image_patch_mask_inds(
                observation_cov.trafo.im_shape, patch_size=cfg.inference.patch_size)
        patch_idx_list = cfg.inference.patch_idx_list
        if patch_idx_list is None:
            patch_idx_list = list(range(len(all_patch_mask_inds)))
        elif isinstance(patch_idx_list, str):
            patch_idx_list = get_predefined_patch_idx_list(
                    name=patch_idx_list, patch_size=cfg.inference.patch_size)

        patch_kwargs = {
                'patch_size': cfg.inference.patch_size,
                'patch_idx_list': patch_idx_list,
                'batch_size': cfg.inference.batch_size,
        }

        log_probs_unscaled, patch_diags = predictive_posterior.log_prob_patches(
            mean=recon,
            ground_truth=ground_truth,
            samples=samples,
            patch_kwargs=patch_kwargs,
            reweight_off_diagonal_entries=cfg.inference.reweight_off_diagonal_entries,
            noise_x_correction_term=noise_x_correction_term,
            verbose=cfg.inference.verbose,
            return_patch_diags=True,
            unscaled=True)

        total_num_pixels_in_patches = sum(
                len(all_patch_mask_inds[patch_idx]) for patch_idx in patch_idx_list)
        log_prob = np.sum(log_probs_unscaled) / total_num_pixels_in_patches
        print('log_prob:', log_prob)

        torch.save({
            'patch_mask_inds': [all_patch_mask_inds[idx] for idx in patch_idx_list],
            'patch_log_probs_unscaled': log_probs_unscaled,
            'log_prob': log_prob,
            'patch_cov_diags': [diag.cpu() for diag in patch_diags],
            'image_noise_correction_term': noise_x_correction_term,
        }, f'sample_based_predictive_posterior_{i}.pt')

if __name__ == '__main__':
    coordinator()  # pylint: disable=no-value-for-parameter
