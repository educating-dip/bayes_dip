import os
from warnings import warn
from itertools import islice
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from bayes_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_dataset, get_predefined_patch_idx_list
from bayes_dip.utils import PSNR, SSIM, eval_mode
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts, get_default_unet_gprior_dicts
from bayes_dip.probabilistic_models import NeuralBasisExpansion, LowRankNeuralBasisExpansion, LowRankObservationCov, ParameterCov, ImageCov, ObservationCov, GpriorNeuralBasisExpansion, get_image_noise_correction_term
from bayes_dip.marginal_likelihood_optim import LowRankPreC
from bayes_dip.inference import SampleBasedPredictivePosterior, get_image_patch_mask_inds
from bayes_dip.data.datasets import get_walnut_2d_inner_patch_indices


def _save_samples(i: int, samples: Tensor, chunk_size: int) -> None:
    for j, start_i in enumerate(range(0, len(samples), chunk_size)):
        sample_chunk = samples[start_i:start_i+chunk_size].clone()
        torch.save(sample_chunk, f'samples_{i}_chunk_{j}.pt')


def _load_samples(path: str, i: int, num_samples: int, restrict_to_num_samples=True) -> Tensor:
    sample_chunks = []
    num_loaded_samples = 0
    j = 0
    while num_loaded_samples < num_samples:
        try:
            chunk = torch.load(os.path.join(path, f'samples_{i}_chunk_{j}.pt'))
        except FileNotFoundError:
            raise RuntimeError(
                    f'Failed to load {num_samples} samples from {path}, '
                    f'only found {num_loaded_samples}.')
        sample_chunks.append(chunk)
        num_loaded_samples += chunk.shape[0]
        j += 1
    samples = torch.cat(sample_chunks)
    if restrict_to_num_samples:
        samples = samples[:num_samples]
    return samples


@hydra.main(config_path='hydra_cfg', config_name='config', version_base='1.2')
def coordinator(cfg : DictConfig) -> None:

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
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        observation, ground_truth, filtbackproj = data_sample

        # assert that sample data matches with that from [exact_]dip_mll_optim.py
        sample_dict = torch.load(os.path.join(cfg.inference.load_path, 'sample_{}.pt'.format(i)), map_location=device)
        assert torch.allclose(sample_dict['filtbackproj'].float(), filtbackproj.float(), atol=1e-6)

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
                load_params_path=os.path.join(cfg.inference.load_path, 'dip_model_{}.pt'.format(i)))

        with torch.no_grad(), eval_mode(reconstructor.nn_model):
            recon = reconstructor.nn_model(filtbackproj)

        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

        prior_assignment_dict, hyperparams_init_dict = get_default_unet_gaussian_prior_dicts(
                reconstructor.nn_model) if not cfg.priors.use_gprior else get_default_unet_gprior_dicts(
                        nn_model=reconstructor.nn_model,
        )
        parameter_cov = ParameterCov(
                reconstructor.nn_model,
                prior_assignment_dict,
                hyperparams_init_dict,
                device=device
        )
        neural_basis_expansion = NeuralBasisExpansion(
                nn_model=reconstructor.nn_model,
                nn_input=filtbackproj,
                ordered_nn_params=parameter_cov.ordered_nn_params,
                nn_out_shape=filtbackproj.shape,
        ) if cfg.priors.use_gprior else GpriorNeuralBasisExpansion(
                    trafo=ray_trafo,
                    nn_model=reconstructor.nn_model,
                    nn_input=filtbackproj,
                    ordered_nn_params=parameter_cov.ordered_nn_params,
                    nn_out_shape=filtbackproj.shape,
                    scale_kwargs=OmegaConf.to_object(cfg.priors.gprior.scale)
            )
        if cfg.inference.use_low_rank_neural_basis_expansion:
            neural_basis_expansion = LowRankNeuralBasisExpansion(
                neural_basis_expansion=neural_basis_expansion,
                vec_batch_size=cfg.inference.low_rank_neural_basis_expansion.batch_size,
                oversampling_param=cfg.inference.low_rank_neural_basis_expansion.oversampling_param,
                low_rank_rank_dim=cfg.inference.low_rank_neural_basis_expansion.low_rank_rank_dim,
                device=device,
                use_cpu=cfg.inference.low_rank_neural_basis_expansion.use_cpu)

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

        print('computing noise correction term')
        noise_x_correction_term = get_image_noise_correction_term(observation_cov=observation_cov)
        print('noise_x_correction_term:', noise_x_correction_term)

        if cfg.inference.load_cov_obs_mat_from_path is None:
            cov_obs_mat = observation_cov.assemble_observation_cov(
                    vec_batch_size=cfg.inference.cov_obs_mat.batch_size)
        else:
            cov_obs_mat = torch.load(
                    os.path.join(cfg.inference.load_cov_obs_mat_from_path, f'cov_obs_mat_{i}.pt'),
                    map_location=observation_cov.device)
            if cfg.use_double and cov_obs_mat.dtype == torch.float32:
                warn('Loaded cov_obs_mat with dtype float32 but running with use_double=True')
            cov_obs_mat = cov_obs_mat.to(dtype=dtype)
        if cfg.inference.save_cov_obs_mat:
            torch.save(cov_obs_mat.cpu(), f'cov_obs_mat_{i}.pt')
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
            sample_kwargs = OmegaConf.to_object(cfg.inference.sampling)
            if cfg.inference.sampling.use_conj_grad_inv:
                low_rank_observation_cov = LowRankObservationCov(
                        trafo=ray_trafo,
                        image_cov=image_cov,
                        low_rank_rank_dim=cfg.inference.cov_obs_mat.low_rank_rank_dim,
                        oversampling_param=cfg.inference.cov_obs_mat.oversampling_param,
                        vec_batch_size=cfg.inference.cov_obs_mat.batch_size,
                        device=device
                )
                low_rank_preconditioner = LowRankPreC(
                        pre_con_obj=low_rank_observation_cov
                )
                sample_kwargs['cg_kwargs']['precon_closure'] = (
                        lambda v: low_rank_preconditioner.matmul(v.T, use_inverse=True).T)
            samples = predictive_posterior.sample_zero_mean(
                    num_samples=cfg.inference.num_samples,
                    cov_obs_mat_chol=cov_obs_mat_chol,
                    return_on_device='cpu',
                    **sample_kwargs)
        else:
            samples = _load_samples(
                    path=cfg.inference.load_samples_from_path, i=i,
                    num_samples=cfg.inference.num_samples)

        if cfg.inference.save_samples:
            _save_samples(i=i, samples=samples, chunk_size=cfg.inference.save_samples_chunk_size)

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
            noise_x_correction_term=noise_x_correction_term,
            verbose=cfg.inference.verbose,
            return_patch_diags=True,
            unscaled=True)

        total_num_pixels_in_patches = sum(len(all_patch_mask_inds[patch_idx]) for patch_idx in patch_idx_list)
        log_prob = np.sum(log_probs_unscaled) / total_num_pixels_in_patches
        print('log_prob:', log_prob)

        torch.save({
            'patch_mask_inds': [all_patch_mask_inds[idx] for idx in patch_idx_list],
            'patch_log_probs_unscaled': log_probs_unscaled,
            'log_prob': log_prob,
            'patch_cov_diags': [diag.cpu() for diag in patch_diags],
        }, f'sample_based_predictive_posterior_{i}.pt')

if __name__ == '__main__':
    coordinator()
