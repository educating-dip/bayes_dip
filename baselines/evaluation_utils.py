from typing import Optional, Dict, Tuple, List, Union
import os
from bayes_dip.utils.evaluation_utils import translate_path, get_sample_based_cov_diag
import torch
from torch import Tensor
from omegaconf import OmegaConf
from bayes_dip.utils.experiment_utils import load_samples

def compute_mcdo_reconstruction(
        run_path: str, sample_idx: int,
        device=None,
        experiment_paths: Optional[Dict] = None,
        ) -> float:
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
    device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    dtype = torch.double if cfg.use_double else torch.float
    samples = load_samples(
            path=cfg.baseline.load_samples_from_path, i=sample_idx,
            num_samples=cfg.baseline.num_samples
        ).to(dtype=dtype, device=device)
    mean_recon = samples.mean(dim=0, keepdim=True)
    return mean_recon

def get_mcdo_density_data(
        run_path: str, sample_idx: int,
        experiment_paths: Optional[Dict] = None) -> Tuple[Dict, bool]:
    """
    Return the MCDO density data for sample ``sample_idx`` from ``run_path``.

    Returns
    -------
    data : dict
        Density data from "sample_based_mcdo_predictive_posterior_{sample_idx}.pt".
    """
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    filepath = os.path.join(
            run_path, f'sample_based_mcdo_predictive_posterior_{sample_idx}.pt')
    data = torch.load(filepath, map_location='cpu')
    return data

def get_mcdo_stddev(run_path: str, sample_idx: int,
        patch_idx_list: Optional[Union[List[int], str]] = None,
        subtract_image_noise_correction_if_any: bool = True,
        experiment_paths: Optional[Dict] = None) -> Tensor:
    """
    Return the MCDO standard deviation (i.e. the square root of the diagonal covariance) for sample
    ``sample_idx`` from ``run_path``.

    Parameters
    ----------
    run_path : str
        Path of the hydra run.
    sample_idx : int
        Sample index.
    patch_idx_list : list of int or str, optional
        Patch indices to restrict to.
        Must be a subset of ``cfg.inference.patch_idx_list`` of the original run.
        If a string,
        ``bayes_dip.utils.experiment_utils.get_predefined_patch_idx_list(patch_idx_list)`` is used.
        If ``None`` (the default), all patch indices are used.
    subtract_image_noise_correction_if_any : bool, optional
        Whether to subtract the image noise correction term (if any) from the covariance diagonal
        before taking the square root.
        The default is ``True``.
    experiment_paths : dict, optional
        See :func:`translate_path`.

    Returns
    -------
    stddev : Tensor
        Standard deviation. Shape: ``(im_size, im_size)``.
    """
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
    data = get_mcdo_density_data(
            run_path=run_path, sample_idx=sample_idx, experiment_paths=experiment_paths)
    cov_diag = get_sample_based_cov_diag(
            data=data,
            patch_idx_list=patch_idx_list,
            patch_size=cfg.inference.patch_size,
            im_shape=(cfg.dataset.im_size,) * 2)
    if subtract_image_noise_correction_if_any and cfg.inference.add_image_noise_correction_term:
        image_noise_correction_term = data['image_noise_correction_term']
        print(f'subtracting {image_noise_correction_term} (image noise correction) from cov_diag')
        cov_diag -= image_noise_correction_term
    stddev = torch.sqrt(cov_diag)
    return stddev
