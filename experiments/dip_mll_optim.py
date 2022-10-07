import os
from itertools import islice
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from bayes_dip.utils.experiment_utils import (
        get_standard_ray_trafo, get_standard_dataset, assert_sample_matches)
from bayes_dip.utils import PSNR, SSIM
from bayes_dip.dip import DeepImagePriorReconstructor, UNetReturnPreSigmoid
from bayes_dip.probabilistic_models import (
        get_default_unet_gaussian_prior_dicts, get_default_unet_gprior_dicts)
from bayes_dip.probabilistic_models import (
        get_neural_basis_expansion, ParameterCov, ImageCov, ObservationCov)
from bayes_dip.marginal_likelihood_optim import (
        marginal_likelihood_hyperparams_optim, get_preconditioner, weights_linearization,
        get_ordered_nn_params_vec)

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
        if load_dip_params_from_path is None:
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
        image_cov = ImageCov(
                parameter_cov=parameter_cov,
                neural_basis_expansion=neural_basis_expansion
        )
        observation_cov = ObservationCov(
                trafo=ray_trafo,
                image_cov=image_cov,
                device=device
        )
        if cfg.mll_optim.init_load_path is not None:
            # assert that sample data matches with that from the initial checkpoint to be loaded
            assert_sample_matches(data_sample, cfg.mll_optim.init_load_path, i)
            init_load_filepath = os.path.join(cfg.mll_optim.init_load_path,
                    (f'observation_cov_{i}.pt' if cfg.mll_optim.init_load_iter is None else
                     f'observation_cov_{i}_iter_{cfg.mll_optim.init_load_iter}.pt'))
            print(f'loading initial MLL hyperparameters from {init_load_filepath}')
            observation_cov.load_state_dict(torch.load(init_load_filepath))
        linearized_weights = None
        if cfg.mll_optim.use_linearized_weights:
            if load_dip_params_from_path is not None:
                try:
                    linearized_weights = torch.load(
                            os.path.join(load_dip_params_from_path, f'lin_weights_{i}.pt'))
                    lin_recon = torch.load(
                            os.path.join(load_dip_params_from_path, f'lin_recon_{i}.pt'))
                except FileNotFoundError:
                    pass
            if linearized_weights is None:
                weights_linearization_optim_kwargs = OmegaConf.to_object(
                        cfg.mll_optim.weights_linearization)
                weights_linearization_optim_kwargs['gamma'] = cfg.dip.optim.gamma
                map_weights = torch.clone(get_ordered_nn_params_vec(parameter_cov))
                neural_basis_expansion_no_sigmoid = (
                        neural_basis_expansion if not reconstructor.nn_model.use_sigmoid else
                        get_neural_basis_expansion(
                                nn_model=UNetReturnPreSigmoid(reconstructor.nn_model),
                                nn_input=filtbackproj,
                                ordered_nn_params=parameter_cov.ordered_nn_params,
                                nn_out_shape=filtbackproj.shape,
                                use_gprior=cfg.priors.use_gprior,
                                trafo=ray_trafo,
                                scale_kwargs=OmegaConf.to_object(cfg.priors.gprior.scale))
                )
                linearized_weights, lin_recon = weights_linearization(
                        trafo=ray_trafo,
                        neural_basis_expansion=neural_basis_expansion_no_sigmoid,
                        use_sigmoid=reconstructor.nn_model.use_sigmoid,
                        map_weights=map_weights,
                        observation=observation,
                        ground_truth=ground_truth,
                        optim_kwargs=weights_linearization_optim_kwargs,
                )
            print(f'linearized weights reconstruction of sample {i:d}')
            print('PSNR:', PSNR(lin_recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(lin_recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
            torch.save(linearized_weights,
                    f'lin_weights_{i}.pt'
            )
            torch.save(lin_recon.cpu(),
                    f'lin_recon_{i}.pt'
            )
        cg_preconditioner = None
        if cfg.mll_optim.linear_cg.use_preconditioner:
            cg_preconditioner = get_preconditioner(
                    observation_cov=observation_cov,
                    kwargs=OmegaConf.to_object(cfg.mll_optim.linear_cg.preconditioner))
        predcp_kwargs = OmegaConf.to_object(cfg.mll_optim.predcp)
        predcp_kwargs['gamma'] = cfg.dip.optim.gamma
        marglik_optim_kwargs = {
                'iterations': cfg.mll_optim.iterations,
                'lr': cfg.mll_optim.lr,
                'scheduler':{
                    'use_scheduler': cfg.mll_optim.scheduler.use_scheduler,
                    'step_size': cfg.mll_optim.scheduler.step_size,
                    'gamma': cfg.mll_optim.scheduler.gamma,
                },
                'num_probes': cfg.mll_optim.num_probes,
                'linear_cg': {
                    'preconditioner': cg_preconditioner,
                    'max_iter': cfg.mll_optim.linear_cg.max_iter,
                    'rtol': cfg.mll_optim.linear_cg.rtol,
                    'use_log_re_variant': cfg.mll_optim.linear_cg.use_log_re_variant,
                    'update_freq': cfg.mll_optim.linear_cg.update_freq,
                    'use_preconditioned_probes': cfg.mll_optim.linear_cg.use_preconditioned_probes
                },
                'min_log_variance': cfg.mll_optim.min_log_variance,
                'include_predcp': cfg.mll_optim.include_predcp,
                'predcp': predcp_kwargs,
                }

        assert not (cfg.priors.use_gprior and cfg.mll_optim.include_predcp)
        marginal_likelihood_hyperparams_optim(
                observation_cov=observation_cov,
                observation=observation,
                recon=recon,
                linearized_weights=linearized_weights,
                optim_kwargs=marglik_optim_kwargs,
                log_path=os.path.join(cfg.mll_optim.log_path, f'mrglik_optim_{i}'),
                comment=f'{i}',
        )
        torch.save(
                observation_cov.state_dict(),
                f'observation_cov_{i}.pt')


if __name__ == '__main__':
    coordinator()  # pylint: disable=no-value-for-parameter
