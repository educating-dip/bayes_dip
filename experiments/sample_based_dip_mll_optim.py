import os
import torch
import hydra
from warnings import warn
from itertools import islice
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import (
        ParameterCov, ImageCov, ObservationCov, LowRankObservationCov, 
        get_neural_basis_expansion, get_default_unet_gprior_dicts)
from bayes_dip.marginal_likelihood_optim import (
        get_preconditioner, get_ordered_nn_params_vec, 
        sample_based_marginal_likelihood_optim)
from bayes_dip.inference import SampleBasedPredictivePosterior
from bayes_dip.utils.experiment_utils import ( 
        get_standard_ray_trafo, get_standard_dataset, assert_sample_matches)
from bayes_dip.utils import PSNR, SSIM

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
            cfg, ray_trafo, use_fixed_seeds_starting_from=cfg.seed,
            device=device)

    for i, data_sample in enumerate(islice(DataLoader(dataset), cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)

        observation, ground_truth, filtbackproj = data_sample

        load_dip_params_from_path = cfg.load_dip_params_from_path
        if cfg.mll_optim.init_load_path is not None and load_dip_params_from_path is None:
            load_dip_params_from_path = cfg.mll_optim.init_load_path

        if load_dip_params_from_path is not None:
            # assert that sample data matches with that from the dip to be loaded
            assert_sample_matches(
                data_sample, load_dip_params_from_path, i, raise_if_file_not_found=False)

        torch.save(
            {'observation': observation,
                'filtbackproj': filtbackproj,
                'ground_truth': ground_truth},
            f'sample_{i}.pt')

        observation = observation.to(dtype=dtype, device=device)
        filtbackproj = filtbackproj.to(dtype=dtype, device=device)
        ground_truth = ground_truth.to(dtype=dtype, device=device)

        try:
            assert cfg.dip.net.use_sigmoid is False
        except AssertionError:
            raise(AssertionError('active sigmoid activation function'))

        net_kwargs = OmegaConf.to_object(cfg.dip.net)
        reconstructor = DeepImagePriorReconstructor(
            ray_trafo, torch_manual_seed=cfg.dip.torch_manual_seed,
            device=device, net_kwargs=net_kwargs,
            load_params_path=cfg.load_pretrained_dip_params)
        if cfg.load_dip_params_from_path is None:
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
                log_path=os.path.join(cfg.dip.log_path, f'dip_optim_{i}'),
                optim_kwargs=optim_kwargs)
        else:
            dip_params_filepath = os.path.join(load_dip_params_from_path, f'dip_model_{i}.pt')
            print(f'loading DIP network parameters from {dip_params_filepath}')
            reconstructor.load_params(dip_params_filepath)
            assert not cfg.dip.recon_from_randn  # would need to re-create random input
            recon = reconstructor.nn_model(filtbackproj).detach()  # pylint: disable=not-callable
        torch.save(reconstructor.nn_model.state_dict(),
                f'dip_model_{i}.pt')
        torch.save(recon.cpu(),
                f'recon_{i}.pt'
        )

        print(f'DIP reconstruction of sample {i}')
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

        assert cfg.priors.use_gprior # sample_based_marginal_likelihood_optim needs a gprior
        prior_assignment_dict, hyperparams_init_dict = get_default_unet_gprior_dicts(
            nn_model=reconstructor.nn_model)
        parameter_cov = ParameterCov(
            reconstructor.nn_model,
            prior_assignment_dict,
            hyperparams_init_dict,
            device=device
        )
        
        if cfg.load_gprior_scale_from_path is not None:
            load_scale_from_path = os.path.join(
                cfg.load_gprior_scale_from_path, f'gprior_scale_vector_{i}.pt')

        neural_basis_expansion = get_neural_basis_expansion(
            nn_model=reconstructor.nn_model,
            nn_input=filtbackproj,
            ordered_nn_params=parameter_cov.ordered_nn_params,
            nn_out_shape=filtbackproj.shape,
            use_gprior=True,
            trafo=ray_trafo,
            load_scale_from_path=load_scale_from_path,
            scale_kwargs=OmegaConf.to_object(cfg.priors.gprior.scale)
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
        observation_cov.image_cov.inner_cov.priors.gprior.log_variance = torch.tensor(
                cfg.priors.gprior.init_prior_variance_value
            ).log() # init prior variance
        observation_cov.log_noise_variance.data = torch.tensor(
                cfg.mll_optim.noise_variance_init_value
            ).log() # init noise variance

        optim_kwargs = {
            'iterations': cfg.mll_optim.iterations,
            'activate_debugging_mode': cfg.mll_optim.activate_debugging_mode,
            'num_samples': cfg.mll_optim.num_samples, 
            'activate_debugging_mode': cfg.mll_optim.activate_debugging_mode
        }
        optim_kwargs['sample_kwargs'] = OmegaConf.to_object(cfg.mll_optim.sampling)

        precon_kwargs = OmegaConf.to_object(cfg.mll_optim.preconditioner)
        
        if cfg.load_sample_based_precon_state_from_path is not None:
            precon_kwargs['load_approx_basis'] = os.path.join(
                cfg.load_sample_based_precon_state_from_path, f'preconditioner_{i}.pt')
            precon_kwargs['load_state_dict'] = os.path.join(
                cfg.load_sample_based_precon_state_from_path, f'observation_cov_{i}.pt')

        cg_preconditioner = None
        if cfg.mll_optim.use_preconditioner:
            cg_preconditioner = get_preconditioner(
                observation_cov=observation_cov,
                kwargs=precon_kwargs)
            optim_kwargs['sample_kwargs']['cg_kwargs']['precon_closure'] = cg_preconditioner.get_closure()
        optim_kwargs['cg_preconditioner'] = cg_preconditioner

        if cfg.mll_optim.activate_debugging_mode: optim_kwargs['debugging_mode_kwargs'] = OmegaConf.to_object(
                cfg.mll_optim.debugging_mode_kwargs
            )

        predictive_posterior = SampleBasedPredictivePosterior(observation_cov)

        linearized_weights, linearized_recon = sample_based_marginal_likelihood_optim(
            predictive_posterior=predictive_posterior,
            map_weights=torch.clone(
                get_ordered_nn_params_vec(parameter_cov)
            ),
            observation=observation,
            nn_recon=recon,
            ground_truth=ground_truth,
            optim_kwargs=optim_kwargs,
            log_path=os.path.join(
                    cfg.mll_optim.log_path, f'mrglik_optim_{i}'
                ),
        )

        torch.save(observation_cov.state_dict(), f'observation_cov_{i}.pt')
        torch.save(linearized_weights, f'linearized_weights_{i}.pt')
        torch.save(linearized_recon, f'linearized_recon_{i}.pt')

if __name__ == '__main__':
    coordinator()
