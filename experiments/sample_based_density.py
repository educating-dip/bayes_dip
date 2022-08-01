import os
from itertools import islice
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import scipy.sparse
import tensorly as tl
from torch.utils.data import DataLoader
from bayes_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_dataset
from bayes_dip.utils import PSNR, SSIM, eval_mode
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts
from bayes_dip.probabilistic_models import NeuralBasisExpansion, LowRankObservationCov, ParameterCov, ImageCov, ObservationCov
from bayes_dip.marginal_likelihood_optim import LowRankPreC
from bayes_dip.inference import SampleBasedPredictivePosterior, get_image_patch_mask_inds
from bayes_dip.data.datasets import get_walnut_2d_inner_patch_indices

@hydra.main(config_path='hydra_cfg', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    dtype = torch.get_default_dtype()
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    ray_trafo = get_standard_ray_trafo(cfg)
    ray_trafo.to(dtype=dtype, device=device)

    # data: observation, ground_truth, filtbackproj
    dataset = get_standard_dataset(
            cfg, ray_trafo, use_fixed_seeds_starting_from=cfg.seed,
            device=device)

    for i, data_sample in enumerate(islice(DataLoader(dataset), cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        observation, ground_truth, filtbackproj = data_sample

        observation = observation.to(dtype=dtype, device=device)
        filtbackproj = filtbackproj.to(dtype=dtype, device=device)
        ground_truth = ground_truth.to(dtype=dtype, device=device)
        sample_dict = torch.load(os.path.join(cfg.inference.load_path, 'sample_{}.pt'.format(i)), map_location=device)
        assert torch.allclose(sample_dict['filtbackproj'], filtbackproj, atol=1e-6)

        net_kwargs = {
                'scales': cfg.dip.net.scales,
                'channels': cfg.dip.net.channels,
                'skip_channels': cfg.dip.net.skip_channels,
                'use_norm': cfg.dip.net.use_norm,
                'use_sigmoid': cfg.dip.net.use_sigmoid,
                'sigmoid_saturation_thresh': cfg.dip.net.sigmoid_saturation_thresh}
        reconstructor = DeepImagePriorReconstructor(
                ray_trafo, torch_manual_seed=cfg.dip.torch_manual_seed,
                device=device, net_kwargs=net_kwargs)
        reconstructor.nn_model.load_state_dict(
                os.path.join(cfg.inference.load_path, 'dip_model_{}.pt'.format(i))))

        with torch.no_grad(), eval_mode(reconstructor.model):
            recon = reconstructor.model(filtbackproj)

        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

        prior_assignment_dict, hyperparams_init_dict = get_default_unet_gaussian_prior_dicts(
                reconstructor.nn_model)
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
        )
        image_cov = ImageCov(
                parameter_cov=parameter_cov,
                neural_basis_expansion=neural_basis_expansion
        )
        observation_cov = ObservationCov(
                trafo=ray_trafo,
                image_cov=image_cov,
                device=device
        )
        observation_cov.load_state_dict(
                os.path.join(cfg.inference.load_path, f'observation_cov_{i}.pt'))
        if cfg.inference.use_conj_grad_inv:
            low_rank_observation_cov = LowRankObservationCov(
                    trafo=ray_trafo,
                    image_cov=image_cov,
                    low_rank_rank_dim=200,
                    oversampling_param=5,
                    vec_batch_size=1,
                    device=device
            )
            low_rank_preconditioner = LowRankPreC(
                    pre_con_obj=low_rank_observation_cov
            )


        noise_x_correction_term = None
        trafo_mat = ray_trafo.matrix
        if trafo_mat.is_sparse:
            # pseudo-inverse computation
            U_trafo, S_trafo, Vh_trafo = scipy.sparse.linalg.svds(trafo, k=100)
            # (Vh.T S U.T U S Vh)^-1 == (Vh.T S^2 Vh)^-1 == Vh.T S^-2 Vh
            S_inv_Vh_trafo = scipy.sparse.diags(1/S_trafo) @ Vh_trafo
            # trafo_T_trafo_diag = np.diag(S_inv_Vh_trafo.T @ S_inv_Vh_trafo)
            trafo_T_trafo_diag = np.sum(S_inv_Vh_trafo**2, axis=0)
            noise_x_correction_term = np.mean(trafo_T_trafo_diag) * observation_cov.log_noise_variance.exp().item()
        else:
            # pseudo-inverse computation
            trafo = ray_trafo.matrix
            trafo_T_trafo = trafo.T @ trafo
            U, S, Vh = tl.truncated_svd(trafo_T_trafo, n_eigenvecs=100) # costructing tsvd-pseudoinverse
            noise_x_correction_term = (Vh.T @ torch.diag(1/S) @ U.T * observation_cov.log_noise_variance.exp()).diag().mean().item()
        print('noise_x_correction_term:', noise_x_correction_term)

        predictive_posterior = SampleBasedPredictivePosterior(observation_cov)
        predictive_posterior.log_prob_patches()

        patch_idx_list = cfg.inference.patch_idx_list
        if isinstance(patch_idx_list, str):
            if patch_idx_list == 'walnut_inner':
                patch_idx_list = get_walnut_2d_inner_patch_indices(patch_size=cfg.inference.patch_size)
            else:
                raise ValueError(f'Unknown patch_idx_list configuration: {patch_idx_list}')

        sum_log_prob = np.sum(predictive_posterior.log_prob_patches(
                patch_size=cfg.inference.patch_size, patch_idx_list=patch_idx_list, unscaled=True))
        all_patch_mask_inds = get_image_patch_mask_inds(
                observation_cov.trafo.im_shape, patch_size=cfg.inference.patch_size, flatten=True)
        total_num_pixels_in_patches = sum(len(all_patch_mask_inds[patch_idx]) for patch_idx in patch_idx_list)
        print('log_prob:', sum_log_prob / total_num_pixels_in_patches)

if __name__ == '__main__':
    coordinator()
