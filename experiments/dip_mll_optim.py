from itertools import islice
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from bayes_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_dataset
from bayes_dip.utils import PSNR, SSIM
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts
from bayes_dip.probabilistic_models import NeuralBasisExpansion, ApproxNeuralBasisExpansion, ParameterCov, ImageCov, ObservationCov
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

        # TODO

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
        # print(prior_assignment_dict)
        # print(hyperparams_init_dict)
        parameter_cov = ParameterCov(reconstructor.nn_model, prior_assignment_dict, hyperparams_init_dict, device=device)
        # print('parameter_cov shape:', parameter_cov.shape)

        # v = torch.randn(3, sum(n for n in parameter_cov.params_numel_per_prior_type.values())).to(device)
        # out = parameter_cov(v)
        # print(out.shape)
        # nn_input = torch.randn((1, 1, 28, 28), device=device)

        neural_basis_expansion = NeuralBasisExpansion(
                nn_model=reconstructor.nn_model,
                nn_input=filtbackproj,
                ordered_nn_params=parameter_cov.ordered_nn_params,
                nn_out_shape=filtbackproj.shape,
        )

        # approx_neural_basis_expansion = ApproxNeuralBasisExpansion(
        #         neural_basis_expansion=neural_basis_expansion,
        #         vec_batch_size=1,
        #         oversampling_param=5,
        #         low_rank_rank_dim=10,
        #         device=device,
        #         use_cpu=True
        # )

        image_cov = ImageCov(
                parameter_cov=parameter_cov,
                neural_basis_expansion=neural_basis_expansion
        )
        # print('image_cov shape:', image_cov.shape)

        # image_cov_approx = ImageCov(
        #         parameter_cov=parameter_cov,
        #         neural_basis_expansion=approx_neural_basis_expansion
        # )

        # tests separate clousers

        # v = torch.randn((3, 1, 1, 28, 28), device=device)
        # out = neural_basis_expansion.vjp(v)
        # print(out.shape)

        # v = torch.randn((3, neural_basis_expansion.num_params), device=device)
        # out = neural_basis_expansion.jvp(v)
        # print(out.shape)

        # v = torch.randn((3, 1, 28, 28), device=device)
        # out = image_cov(v)
        # print(out.shape)

        # v = torch.randn((3, 1, 28, 28), device=device)
        # out = image_cov_approx(v)
        # print(out.shape)

        observation_cov = ObservationCov(
                trafo=ray_trafo,
                image_cov=image_cov,
                device=device
        )
        # print('observation_cov shape:', observation_cov.shape)

        # v = torch.randn( (3, 1, ) + ray_trafo.obs_shape, device=device)
        # v = observation_cov(v)
        # print(v.shape)

        # v = observation_cov(v)
        # observation_cov_mat = observation_cov.assemble_observation_cov()
        # print(observation_cov_mat.shape)

        marglik_optim_kwargs = {
                'iterations': 1000, 
                'lr': 0.01,
                'num_probes': 1,
                'min_log_variance': -4.5, 
                'include_predcp': False,
                }

        marginal_likelihood_hyperparams_optim(
                observation_cov=observation_cov,
                observation=observation, 
                recon=recon,
                use_linearized_weights=False,
                optim_kwargs=marglik_optim_kwargs, 
                log_path='./', 
        )


if __name__ == '__main__':
    coordinator()
