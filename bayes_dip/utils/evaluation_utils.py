"""
Utilities for evaluation.
"""
import os
from typing import Tuple, Dict, List, Union, Optional
import numpy as np
import torch
from torch import Tensor
from omegaconf import OmegaConf
from bayes_dip.utils.experiment_utils import get_standard_ray_trafo, get_predefined_patch_idx_list
from bayes_dip.probabilistic_models import get_trafo_t_trafo_pseudo_inv_diag_mean
from bayes_dip.inference import get_image_patch_mask_inds

DEFAULT_OUTPUTS_PATH = '../experiments/outputs'

def translate_output_path(path: str, outputs_path: Optional[str] = DEFAULT_OUTPUTS_PATH):
    path = path.rstrip('/')
    if outputs_path is not None:
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
    abs_diff = torch.abs(ground_truth - recon).squeeze(1).squeeze(0)
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

def get_patch_idx_to_mask_inds_dict(
        patch_idx_list: Union[List[int], str, None], im_shape: Tuple[int, int], patch_size: int):
    all_patch_mask_inds = get_image_patch_mask_inds(im_shape, patch_size=patch_size)
    if patch_idx_list is None:
        patch_idx_list = list(range(len(all_patch_mask_inds)))
    elif isinstance(patch_idx_list, str):
        patch_idx_list = get_predefined_patch_idx_list(
                name=patch_idx_list, patch_size=patch_size)
    return {idx: all_patch_mask_inds[idx] for idx in patch_idx_list}

def get_sample_based_cov_diag(
        run_path: str, data: Dict, patch_idx_list: Union[List[int], str, None]) -> Tensor:

    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
    im_shape = (cfg.dataset.im_size,) * 2

    # fill in all patches from data
    cov_diag = torch.full(im_shape, torch.nan, dtype=data['patch_cov_diags'][0].dtype)
    for mask_inds, diag in zip(data['patch_mask_inds'], data['patch_cov_diags']):
        cov_diag.view(-1)[mask_inds] = diag.detach()

    cov_diag_requested_mask = torch.zeros(im_shape, dtype=torch.bool)
    for mask_inds in get_patch_idx_to_mask_inds_dict(
            patch_idx_list=patch_idx_list, im_shape=im_shape,
            patch_size=cfg.inference.patch_size).values():
        cov_diag_requested_mask.view(-1)[mask_inds] = True
    # assert that the parts inside the requested patch_idx_list are filled in
    assert not torch.any(torch.isnan(cov_diag[cov_diag_requested_mask]))
    # set parts outside the requested patch_idx_list to nan
    cov_diag[torch.bitwise_not(cov_diag_requested_mask)] = torch.nan

    return cov_diag

def restrict_sample_based_density_data_to_new_patch_idx_list(
        run_path: str, data: Dict, patch_idx_list: Union[List[int], str, None],
        outputs_path: str = DEFAULT_OUTPUTS_PATH) -> Dict:
    run_path = translate_output_path(run_path, outputs_path=outputs_path)
    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
    im_shape = (cfg.dataset.im_size,) * 2
    orig_patch_idx_list = list(get_patch_idx_to_mask_inds_dict(  # original indices
            patch_idx_list=cfg.inference.patch_idx_list, im_shape=im_shape,
            patch_size=cfg.inference.patch_size))
    patch_idx_to_mask_inds_dict = get_patch_idx_to_mask_inds_dict(  # news indices and mask inds
            patch_idx_list=patch_idx_list, im_shape=im_shape,
            patch_size=cfg.inference.patch_size)
    assert all(a < b for a, b in zip(orig_patch_idx_list[:-1], orig_patch_idx_list[1:]))
    indices = np.searchsorted(orig_patch_idx_list, list(patch_idx_to_mask_inds_dict))
    data_restricted = {
        'patch_mask_inds': [data['patch_mask_inds'][i] for i in indices],
        'patch_log_probs_unscaled': [data['patch_log_probs_unscaled'][i] for i in indices],
        'log_prob': None,  # fill later
        'patch_cov_diags': [data['patch_cov_diags'][i] for i in indices],
    }
    assert all(np.array_equal(mask_inds, orig_mask_inds) for mask_inds, orig_mask_inds in zip(
            patch_idx_to_mask_inds_dict.values(), data_restricted['patch_mask_inds']))
    total_num_pixels_in_patches_restricted = sum(
            len(mask_inds) for mask_inds in data_restricted['patch_mask_inds'])
    data_restricted['log_prob'] = np.sum(
            data_restricted['patch_log_probs_unscaled']) / total_num_pixels_in_patches_restricted
    return data_restricted

def get_stddev(run_path: str, sample_idx: int,
        patch_idx_list: Optional[Union[List[int], str]] = None,
        subtract_image_noise_correction: bool = True,
        outputs_path: str = DEFAULT_OUTPUTS_PATH) -> Tensor:
    run_path = translate_output_path(run_path, outputs_path=outputs_path)
    data, is_exact = get_density_data(run_path=run_path, sample_idx=sample_idx)
    if is_exact:
        cov_diag = data['cov'].detach().diag()
    else:
        cov_diag = get_sample_based_cov_diag(
                run_path=run_path, data=data, patch_idx_list=patch_idx_list)
    if subtract_image_noise_correction:
        image_noise_correction_term = _recompute_image_noise_correction_term(
                run_path=run_path, sample_idx=sample_idx)
        print(f'subtracting {image_noise_correction_term} (image noise correction) from cov_diag')
        cov_diag -= image_noise_correction_term
    stddev = torch.sqrt(cov_diag)
    return stddev
