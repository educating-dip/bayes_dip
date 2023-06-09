import os
import torch
import hydra
from glob import glob
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
from bayes_dip.inference.sample_based_predictive_posterior import log_prob_patches
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
        
        em_step = cfg.get('em_step', 0)
        load_previous_em_step_from_path = cfg.get('load_previous_em_step_from_path', None)

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
            raise ValueError('load_dip_params_from_path must be set')
        else:
            dip_params_filepath = os.path.join(load_dip_params_from_path, f'dip_model_{i}.pt')
            print(f'loading DIP network parameters from {dip_params_filepath}')
            reconstructor.load_params(dip_params_filepath)
            assert not cfg.dip.recon_from_randn  # would need to re-create random input
            recon = reconstructor.nn_model(filtbackproj).detach()  # pylint: disable=not-callable
        
        print(f'DIP reconstruction of sample {i}')
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        
        samples = []
        print(f'debug : load_previous_em_step_from_path = {load_previous_em_step_from_path}')
        if load_previous_em_step_from_path is not None:
            samples_paths = glob(
                    os.path.join(load_previous_em_step_from_path, f'image_sample_{i}_em={em_step}_seed=*.pt'))
            print(samples_paths)
            for k, path in enumerate(samples_paths):
                print(f'Loading sample from : ', path)
                sample_i = torch.load(path, map_location='cpu')
                
                samples.append(sample_i)
        
        samples = torch.cat(samples, dim=0)
        
        image_samples = samples
        image_samples = image_samples[:, :, 73:93, :, :]
        patch_kwargs = {'patch_size': 1, 'batch_size': 1024,}
        log_prob = log_prob_patches(
            mean=recon.cpu(),
            ground_truth=ground_truth.cpu(),
            samples=image_samples,
            patch_kwargs=patch_kwargs,
            reweight_off_diagonal_entries=False,
            noise_x_correction_term=1e-6,
            verbose = False,
            unscaled = False,
            return_patch_diags = False,
            device = 'cpu'
        )
        print(f'mean log prob : ', torch.mean(torch.FloatTensor(log_prob)))
        torch.save(log_prob, f'log_prob_{i}.pt')

if __name__ == '__main__':
    coordinator()
