import os
from itertools import islice
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import scipy.sparse
from torch.utils.data import DataLoader
from bayes_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_dataset
from bayes_dip.utils import PSNR, SSIM, eval_mode
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts
from bayes_dip.probabilistic_models import ParameterCov, ImageCov, MatmulObservationCov, MatmulNeuralBasisExpansion, get_image_noise_correction_term
from bayes_dip.inference import ExactPredictivePosterior

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
            cfg, ray_trafo, use_fixed_seeds_starting_from=cfg.seed,
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
                reconstructor.nn_model)
        parameter_cov = ParameterCov(
                reconstructor.nn_model,
                prior_assignment_dict,
                hyperparams_init_dict,
                device=device
        )
        neural_basis_expansion = MatmulNeuralBasisExpansion(
                nn_model=reconstructor.nn_model,
                nn_input=filtbackproj,
                ordered_nn_params=parameter_cov.ordered_nn_params,
                nn_out_shape=filtbackproj.shape,
        )
        image_cov = ImageCov(
                parameter_cov=parameter_cov,
                neural_basis_expansion=neural_basis_expansion
        )
        observation_cov = MatmulObservationCov(
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

        predictive_posterior = ExactPredictivePosterior(observation_cov)

        cov = predictive_posterior.covariance(noise_x_correction_term=noise_x_correction_term)

        log_prob = predictive_posterior.log_prob(
            mean=recon,
            ground_truth=ground_truth,
            noise_x_correction_term=noise_x_correction_term)

        print('log_prob:', log_prob)

        torch.save({
            'log_prob': log_prob,
            'cov': cov.detach().cpu(),
            'noise_x_correction_term': noise_x_correction_term,
        }, f'exact_predictive_posterior_{i}.pt')

if __name__ == '__main__':
    coordinator()
