import os
from itertools import islice
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from bayes_dip.utils.experiment_utils import (
        get_standard_ray_trafo, get_standard_dataset, save_samples)
from bayes_dip.utils import PSNR, SSIM, eval_mode
from bayes_dip.dip import DeepImagePriorReconstructor
from baselines import bayesianize_unet_architecture, sample_from_bayesianized_model

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

        optim_kwargs = {
                'lr': cfg.dip.optim.lr,
                'iterations': cfg.dip.optim.iterations,
                'loss_function': cfg.dip.optim.loss_function
            }

        bayesianize_unet_architecture(
            reconstructor.nn_model, p=cfg.baseline.p)
        if cfg.baseline.load_mcdo_dip_params_from_path is not None:
            with torch.no_grad(), eval_mode(reconstructor.nn_model):
                mcdo_dip_params_filepath = os.path.join(
                    cfg.baseline.load_mcdo_dip_params_from_path, f'mcdo_dip_model_{i}.pt')
                print(f'loading mcdo DIP network parameters from {mcdo_dip_params_filepath}')
                reconstructor.load_params(mcdo_dip_params_filepath)
                recon = reconstructor.nn_model(filtbackproj)  # pylint: disable=not-callable
                torch.save(
                    reconstructor.nn_model.state_dict(),
                    f'mcdo_dip_model_{i}.pt')
        else:
            recon = reconstructor.reconstruct(
                    observation,
                    filtbackproj=filtbackproj,
                    ground_truth=ground_truth,
                    recon_from_randn=cfg.dip.recon_from_randn,
                    log_path=os.path.join(cfg.dip.log_path, f'dip_optim_{i}'),
                    use_tv_loss=False,
                    optim_kwargs=optim_kwargs
                    )
            torch.save(
                reconstructor.nn_model.state_dict(),
                f'mcdo_dip_model_{i}.pt')

        print(f'DIP reconstruction of sample {i:d}')
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

        samples = sample_from_bayesianized_model(
                    reconstructor.nn_model,
                    filtbackproj,
                    mc_samples=cfg.baseline.num_samples
            )

        if cfg.baseline.save_samples:
            save_samples(i=i, samples=samples, chunk_size=cfg.baseline.save_samples_chunk_size)

if __name__ == '__main__':
    coordinator()  # pylint: disable=no-value-for-parameter
