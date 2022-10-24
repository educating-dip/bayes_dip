import os
from itertools import islice
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from bayes_dip.utils.experiment_utils import (
        get_standard_ray_trafo, get_standard_dataset, assert_sample_matches)
from bayes_dip.utils import PSNR, SSIM, eval_mode
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts
from bayes_dip.probabilistic_models import (
        ParameterCov, ImageCov, MatmulObservationCov, get_matmul_neural_basis_expansion,
        get_image_noise_correction_term)
from bayes_dip.inference import ExactPredictivePosterior

@hydra.main(config_path='hydra_cfg', config_name='config', version_base='1.2')
def coordinator(cfg : DictConfig) -> None:
#     pylint: disable=too-many-locals

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

        prior_assignment_dict, hyperparams_init_dict = get_default_unet_gaussian_prior_dicts(
                reconstructor.nn_model)
        parameter_cov = ParameterCov(
                reconstructor.nn_model,
                prior_assignment_dict,
                hyperparams_init_dict,
                device=device
        )
        neural_basis_expansion = get_matmul_neural_basis_expansion(
                nn_model=reconstructor.nn_model,
                nn_input=filtbackproj,
                ordered_nn_params=parameter_cov.ordered_nn_params,
                nn_out_shape=filtbackproj.shape,
                use_gprior=cfg.priors.use_gprior,
                trafo=ray_trafo,
                scale_kwargs=OmegaConf.to_object(cfg.priors.gprior.scale),
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

        assert not cfg.inference.reweight_off_diagonal_entries

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
    coordinator()  # pylint: disable=no-value-for-parameter
