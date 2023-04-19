import os
import torch
import hydra
from itertools import islice
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import (
        ParameterCov, ImageCov, ObservationCov, 
        get_neural_basis_expansion, get_default_unet_gprior_dicts)
from bayes_dip.marginal_likelihood_optim import sample_then_optimise
from bayes_dip.utils.experiment_utils import ( 
        get_standard_ray_trafo, get_standard_dataset, assert_sample_matches)
from bayes_dip.utils import PSNR, SSIM

@hydra.main(config_path='hydra_cfg', config_name='config', version_base='1.2')
def coordinator(cfg : DictConfig) -> None:
    # pylint: disable=too-many-locals,too-many-statements

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)
    dtype = torch.get_default_dtype()
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
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
        
        em_step = cfg.get('em_step', 0)
        load_previous_em_step_from_path = cfg.get(
                'load_previous_em_step_from_path', None)
        load_previous_weight_sample_from_path = cfg.get(
                'load_previous_weight_sample_from_path', None)

        observation, ground_truth, filtbackproj = data_sample
        load_dip_params_from_path = cfg.load_dip_params_from_path
        if cfg.mll_optim.init_load_path is not None and load_dip_params_from_path is None:
            load_dip_params_from_path = cfg.mll_optim.init_load_path

        if load_dip_params_from_path is not None:
            # assert that sample data matches with that from the dip to be loaded
            assert_sample_matches(
                data_sample, load_dip_params_from_path, i, raise_if_file_not_found=False)
        torch.save(
            {'observation': observation, 'filtbackproj': filtbackproj, 'ground_truth': ground_truth}, f'sample_{i}.pt')

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
        torch.save(reconstructor.nn_model.state_dict(), f'dip_model_{i}.pt')
        torch.save(recon.cpu(), f'recon_{i}.pt')

        print(f'DIP reconstruction of sample {i}')
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

        assert cfg.priors.use_gprior # sample_based_marginal_likelihood_optim needs a g-prior
        prior_assignment_dict, hyperparams_init_dict = get_default_unet_gprior_dicts(
                nn_model=reconstructor.nn_model, 
                gprior_hyperparams_init={'variance': cfg.priors.gprior.init_prior_variance_value})
        parameter_cov = ParameterCov(
            reconstructor.nn_model,
            prior_assignment_dict,
            hyperparams_init_dict,
            device=device
            )
    
        # overwrite g_prior variance with the optimised
        if cfg.load_gprior_scale_from_path is not None:
            load_scale_from_path = os.path.join(
                cfg.load_gprior_scale_from_path, f'gprior_scale_vector_{i}.pt')
        else:
            load_scale_from_path = None

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
        neural_basis_expansion.save_scale(filepath=f'gprior_scale_vector_{i}')
        image_cov = ImageCov(parameter_cov=parameter_cov, neural_basis_expansion=neural_basis_expansion)
        observation_cov = ObservationCov(trafo=ray_trafo, image_cov=image_cov, device=device)
    
        prev_weight_sample = None
        if load_previous_em_step_from_path is not None:
            loaded_observation_cov_data = torch.load(
                os.path.join(load_previous_em_step_from_path, f'observation_cov_iter_{em_step - 1}.pt'))
            loaded_gprior_log_variance = loaded_observation_cov_data['image_cov.inner_cov.priors.gprior._log_variance']
            observation_cov.image_cov.inner_cov.priors.gprior.log_variance = loaded_gprior_log_variance
        if load_previous_weight_sample_from_path is not None:
            if em_step > 0:
                # TODO: check seeds are matching in this output folder
                prev_weight_sample = torch.load(
                    os.path.join(load_previous_weight_sample_from_path, f'weight_sample_{i}_em={em_step-1}_seed={cfg.seed + i}.pt'))

        optim_kwargs = {'num_samples': cfg.mll_optim.num_samples}
        optim_kwargs['sample_kwargs'] = OmegaConf.to_object(cfg.mll_optim.sampling)

        weight_sample = sample_then_optimise(
            observation_cov=observation_cov,
            neural_basis_expansion=observation_cov.image_cov.neural_basis_expansion, 
            noise_variance=observation_cov.log_noise_variance.exp().detach(), 
            variance_coeff=observation_cov.image_cov.inner_cov.priors.gprior.log_variance.exp().detach(), 
            num_samples=cfg.mll_optim.get('num_samples_per_device', 1),
            optim_kwargs=optim_kwargs['sample_kwargs']['hyperparams_update']['optim_kwargs'],
            init_at_previous_samples=prev_weight_sample,
            name_prefix=f'weight_sample_{i}_em={em_step}_seed={cfg.seed + i}'
            )
        # Zero mean samples.
        image_samples = observation_cov.image_cov.neural_basis_expansion.jvp(weight_sample).squeeze(dim=1)
        obs_samples = observation_cov.trafo(image_samples)
        posterior_obs_samples_sq_sum = {'value': obs_samples.pow(2).sum(dim=0), 'num_samples': cfg.mll_optim.get('num_samples_per_device', 1)}
        
        torch.save(posterior_obs_samples_sq_sum, f'posterior_obs_samples_sq_sum_{i}_em={em_step}_seed={cfg.seed + i}.pt')
        torch.save(weight_sample, f'weight_sample_{i}_em={em_step}_seed={cfg.seed + i}.pt')
        torch.save(image_samples, f'image_sample_{i}_em={em_step}_seed={cfg.seed + i}.pt')


if __name__ == '__main__':
    coordinator()
