from itertools import islice
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from bayes_dip.utils import get_standard_ray_trafo, get_standard_dataset
from bayes_dip.utils import PSNR, SSIM
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts
from bayes_dip.probabilistic_models.linearized_dip import ParameterCov

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

        # recon = reconstructor.reconstruct(
        #         observation,
        #         filtbackproj=filtbackproj,
        #         ground_truth=ground_truth,
        #         recon_from_randn=cfg.dip.recon_from_randn,
        #         log_path=cfg.dip.log_path,
        #         optim_kwargs=optim_kwargs)

        # torch.save(reconstructor.model.state_dict(),
        #         './dip_model_{}.pt'.format(i))

        # print('DIP reconstruction of sample {:d}'.format(i))
        # print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        # print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

        prior_assignment_dict, hyperparams_init_dict = get_default_unet_gaussian_prior_dicts(
                reconstructor.model)
        print(prior_assignment_dict)
        print(hyperparams_init_dict)

        parameter_cov = ParameterCov(reconstructor.model, prior_assignment_dict, hyperparams_init_dict, device=device)
        v = torch.randn(3, sum(n for n in parameter_cov.params_numel_per_prior_type.values())).to(device)
        print(v.shape)
        v2 = parameter_cov(v)
        print(v2.shape)

if __name__ == '__main__':
    coordinator()
