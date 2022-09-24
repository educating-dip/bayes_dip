import os
from itertools import islice
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from bayes_dip.utils.experiment_utils import (
        get_standard_ray_trafo, get_standard_dataset, get_predefined_patch_idx_list, load_samples)
from bayes_dip.utils import PSNR, SSIM, eval_mode
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_trafo_t_trafo_pseudo_inv_diag_mean
from bayes_dip.inference import log_prob_patches, get_image_patch_mask_inds
from baselines import bayesianize_unet_architecture, approx_kernel_density

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

        torch.save(
                {'observation': observation,
                 'filtbackproj': filtbackproj,
                 'ground_truth': ground_truth},
                f'sample_{i}.pt')

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
                load_params_path=cfg.load_pretrained_dip_params)

        log_noise_variance = None
        if cfg.baseline.name == 'deterministic':
            with torch.no_grad(), eval_mode(reconstructor.nn_model):

                dip_params_filepath = os.path.join(cfg.load_dip_params_from_path, f'dip_model_{i}.pt')
                print(f'loading DIP network parameters from {dip_params_filepath}')
                reconstructor.load_params(dip_params_filepath)
                recon = reconstructor.nn_model(filtbackproj)  # pylint: disable=not-callable

                print(f'DIP reconstruction of sample {i:d}')
                print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
                print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

                if cfg.baseline.load_log_noise_variance:
                    log_noise_variance = torch.load(
                        os.path.join(cfg.inference.load_path, f'observation_cov_{i}.pt'))['log_noise_variance']

        if cfg.baseline.name == 'mcdo':
            bayesianize_unet_architecture(
                reconstructor.nn_model, p=cfg.baseline.p)
            assert cfg.baseline.load_mcdo_dip_params_from_path is not None
            with torch.no_grad(), eval_mode(reconstructor.nn_model):

                mcdo_dip_params_filepath = os.path.join(
                    cfg.baseline.load_mcdo_dip_params_from_path, f'mcdo_dip_model_{i}.pt')
                print(f'loading mcdo DIP network parameters from {mcdo_dip_params_filepath}')
                reconstructor.load_params(mcdo_dip_params_filepath)
                recon = reconstructor.nn_model(filtbackproj)  # pylint: disable=not-callable

            print(f'DIP reconstruction of sample {i:d}')
            print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

        if log_noise_variance is None:
            log_noise_variance = torch.tensor(1).log()

        noise_x_correction_term = None
        if cfg.inference.add_image_noise_correction_term:
            print('computing noise correction term')
            diag_mean = get_trafo_t_trafo_pseudo_inv_diag_mean(ray_trafo)
            noise_x_correction_term = diag_mean * log_noise_variance.exp().item()
            print('noise_x_correction_term:', noise_x_correction_term)

        if cfg.baseline.name == 'mcdo':
            assert cfg.baseline.load_samples_from_path is not None
            samples = load_samples(
                    path=cfg.baseline.load_samples_from_path, i=i,
                    num_samples=cfg.baseline.num_samples
                ).to(dtype=dtype, device=device)

            mean_recon = samples.mean(dim=0, keepdim=True)

            print(f'DIP mean reconstruction of sample {i:d}')
            print('PSNR:', PSNR(mean_recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(mean_recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

            log_prob_kernel_density = None
            if cfg.dataset.name in ['kmnist']:
                log_prob_kernel_density = approx_kernel_density(
                    ground_truth=ground_truth,
                    samples=samples,
                    noise_x_correction_term=noise_x_correction_term,
                    bw=cfg.baseline.kernel_density_kwargs.bw
                ) / ground_truth.numel()
                log_prob_kernel_density = log_prob_kernel_density.item()

            all_patch_mask_inds = get_image_patch_mask_inds(
                    ray_trafo.im_shape, patch_size=cfg.inference.patch_size)
            patch_idx_list = cfg.inference.patch_idx_list
            if patch_idx_list is None:
                patch_idx_list = list(range(len(all_patch_mask_inds)))
            elif isinstance(patch_idx_list, str):
                patch_idx_list = get_predefined_patch_idx_list(
                        name=patch_idx_list, patch_size=cfg.inference.patch_size)

            log_probs_unscaled, patch_diags = log_prob_patches(
                mean=mean_recon,
                ground_truth=ground_truth,
                samples=samples - samples.mean(dim=0),
                patch_kwargs = {
                    'patch_size': cfg.inference.patch_size,
                    'patch_idx_list': patch_idx_list,
                    'batch_size': cfg.inference.batch_size,
                },
                reweight_off_diagonal_entries=cfg.inference.reweight_off_diagonal_entries,
                noise_x_correction_term=noise_x_correction_term,
                verbose=False,
                return_patch_diags=True,
                unscaled=True
            )

            total_num_pixels_in_patches = sum(
                    len(all_patch_mask_inds[patch_idx]) for patch_idx in patch_idx_list)
            log_prob = np.sum(log_probs_unscaled) / total_num_pixels_in_patches

            print('log_prob_kernel_density:', log_prob_kernel_density)
            print('log_prob:', log_prob)

            torch.save({
                'patch_mask_inds': [all_patch_mask_inds[idx] for idx in patch_idx_list],
                'patch_log_probs_unscaled': log_probs_unscaled,
                'log_prob': log_prob,
                'log_prob_kernel_density': log_prob_kernel_density,
                'patch_cov_diags': [diag.cpu() for diag in patch_diags],
                'image_noise_correction_term': noise_x_correction_term,
            }, f'sample_based_mcdo_predictive_posterior_{i}.pt')

        elif cfg.baseline.name == 'deterministic':

            stddev = (noise_x_correction_term**.5)
            dist = torch.distributions.normal.Normal(loc=recon.flatten(), scale=stddev)
            log_prob=dist.log_prob(ground_truth.flatten()).sum() / ground_truth.numel()
            log_prob = log_prob.item()
            print('log_prob:', log_prob)
            torch.save({
                'log_prob': log_prob,
                'image_noise_correction_term': noise_x_correction_term,
            }, f'deterministic_baseline_load_log_noise_variance_{cfg.baseline.load_log_noise_variance}_{i}.pt')

        else:
            raise NotImplementedError

if __name__ == '__main__':
    coordinator()  # pylint: disable=no-value-for-parameter
