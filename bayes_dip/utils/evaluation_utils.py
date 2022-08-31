"""
Utilities for evaluation.
"""
import os
from typing import Tuple, Dict
import torch
from torch import Tensor
from omegaconf import OmegaConf
from bayes_dip.utils.experiment_utils import get_standard_ray_trafo
from bayes_dip.probabilistic_models import get_trafo_t_trafo_pseudo_inv_diag_mean

DEFAULT_OUTPUTS_PATH = '../experiments/outputs'

def translate_output_path(path : str, outputs_path=DEFAULT_OUTPUTS_PATH):
    path = path.rstrip('/')
    path = os.path.join(outputs_path, os.path.basename(path))
    return path

def get_abs_diff(
        run_path: str, sample_idx: int,
        outputs_path: str = DEFAULT_OUTPUTS_PATH) -> Tensor:
    run_path = translate_output_path(run_path, outputs_path=outputs_path)
    ground_truth = torch.load(os.path.join(run_path, f'sample_{sample_idx}.pt'),
            map_location='cpu')['ground_truth'].detach()
    recon = torch.load(os.path.join(run_path, f'recon_{sample_idx}.pt'),
            map_location='cpu').detach()
    abs_diff = torch.abs(ground_truth - recon)
    return abs_diff

def get_density_data(
        run_path: str, sample_idx: int,
        outputs_path: str = DEFAULT_OUTPUTS_PATH) -> Tuple[Dict, bool]:
    run_path = translate_output_path(run_path, outputs_path=outputs_path)
    exact_filepath = os.path.join(
            run_path, f'exact_predictive_posterior_{sample_idx}.pt')
    sample_based_filepath = os.path.join(
            run_path, f'sample_based_predictive_posterior_{sample_idx}.pt')
    if os.path.isfile(exact_filepath):
        filepath, is_exact = exact_filepath, True
    elif os.path.isfile(sample_based_filepath):
        filepath, is_exact = sample_based_filepath, False
    else:
        raise RuntimeError('Cannot find density data.')
    data = torch.load(filepath, map_location='cpu')
    return data, is_exact

def _recompute_image_noise_correction_term(run_path: str, sample_idx: int) -> float:
    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
    ray_trafo = get_standard_ray_trafo(cfg)
    observation_cov_filename = (
            f'observation_cov_{sample_idx}.pt' if cfg.inference.load_iter is None else
            f'observation_cov_{sample_idx}_iter_{cfg.inference.load_iter}.pt')
    log_noise_variance = torch.load(os.path.join(
            translate_output_path(cfg.inference.load_path),
            observation_cov_filename)
            )['log_noise_variance']
    diag_mean = get_trafo_t_trafo_pseudo_inv_diag_mean(ray_trafo)
    image_noise_correction_term = diag_mean * log_noise_variance.exp().item()
    return image_noise_correction_term

def get_stddev(run_path: str, sample_idx: int,
        subtract_image_noise_correction: bool = True,
        outputs_path: str = DEFAULT_OUTPUTS_PATH) -> Tensor:
    run_path = translate_output_path(run_path, outputs_path=outputs_path)
    data, is_exact = get_density_data(run_path=run_path, sample_idx=sample_idx)
    if is_exact:
        cov_diag = data['cov'].detach().diag()
    else:
        assert all(len(d) == 1 for d in data['patch_cov_diags'])
        cov_diag = torch.cat([d.detach() for d in data['patch_cov_diags']])
    if subtract_image_noise_correction:
        image_noise_correction_term = _recompute_image_noise_correction_term(
                run_path=run_path, sample_idx=sample_idx)
        print(f'subtracting {image_noise_correction_term} (image noise correction) from cov_diag')
        cov_diag -= image_noise_correction_term
    stddev = torch.sqrt(cov_diag)
    return stddev
