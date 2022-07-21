from itertools import islice
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from bayes_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_dataset
from bayes_dip.utils import PSNR, SSIM
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts
from bayes_dip.probabilistic_models import ExactNeuralBasisExpansion, ParameterCov, ImageCov, ExactObservationCov
from bayes_dip.marginal_likelihood_optim import marginal_likelihood_hyperparams_optim

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
        optim_kwargs = {
                'lr': cfg.dip.optim.lr,
                'iterations': cfg.dip.optim.iterations,
                'loss_function': cfg.dip.optim.loss_function,
                'gamma': cfg.dip.optim.gamma}
        recon = reconstructor.reconstruct(
                observation,
                filtbackproj=filtbackproj,
                ground_truth=ground_truth,
                recon_from_randn=cfg.dip.recon_from_randn,
                log_path=cfg.dip.log_path,
                optim_kwargs=optim_kwargs)
        torch.save(reconstructor.nn_model.state_dict(),
                './dip_model_{}.pt'.format(i))

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
        exact_neural_basis_expansion = ExactNeuralBasisExpansion(
                nn_model=reconstructor.nn_model,
                nn_input=filtbackproj,
                ordered_nn_params=parameter_cov.ordered_nn_params,
                nn_out_shape=filtbackproj.shape,
        )
        image_cov = ImageCov(
                parameter_cov=parameter_cov,
                neural_basis_expansion=exact_neural_basis_expansion
        )
        observation_cov = ExactObservationCov(
                trafo=ray_trafo,
                image_cov=image_cov,
                device=device
        )
  
        marglik_optim_kwargs = {
                'iterations': 2000, 
                'lr': 0.01,
                'min_log_variance': -4.5,
                'include_predcp': False,
                'compute_exact_logdet': True, 
                'linearize_weights': {
                        'iterations': 1500, 
                        'lr': 1e-4,
                        'wd': 1e-6, 
                        'gamma': 1e-4, 
                        'simplified_eqn': False, 
                        'use_sigmoid': True, 
                        }
                }

        marginal_likelihood_hyperparams_optim(
                observation_cov=observation_cov,
                observation=observation, 
                recon=recon,
                ground_truth=ground_truth, 
                use_linearized_weights=False,
                optim_kwargs=marglik_optim_kwargs, 
                log_path='./', 
        )


if __name__ == '__main__':
    coordinator()
