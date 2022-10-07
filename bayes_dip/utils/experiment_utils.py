"""
Utilities for experiments.
"""

from typing import List, Optional
import os
from warnings import warn
import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from omegaconf import DictConfig
from bayes_dip.data import get_ray_trafo, SimulatedDataset, BaseRayTrafo
from bayes_dip.data import (
        get_mnist_testset, get_kmnist_testset, get_mnist_trainset, get_kmnist_trainset,
        RectanglesDataset,
        get_walnut_2d_observation, get_walnut_2d_ground_truth)
from bayes_dip.data.datasets.walnut import get_walnut_2d_inner_patch_indices
from .utils import get_original_cwd

def get_standard_ray_trafo(cfg: DictConfig) -> BaseRayTrafo:
    """Return the ray transform by hydra config."""
    kwargs = {}
    kwargs['angular_sub_sampling'] = cfg.trafo.angular_sub_sampling
    if cfg.dataset.name in ('mnist', 'kmnist', 'rectangles'):
        kwargs['im_shape'] = (cfg.dataset.im_size, cfg.dataset.im_size)
        kwargs['num_angles'] = cfg.trafo.num_angles
    elif cfg.dataset.name in ('walnut', 'walnut_120_angles'):
        kwargs['data_path'] = os.path.join(get_original_cwd(), cfg.dataset.data_path)
        kwargs['matrix_path'] = os.path.join(get_original_cwd(), cfg.dataset.data_path)
        kwargs['walnut_id'] = cfg.dataset.walnut_id
        kwargs['orbit_id'] = cfg.trafo.orbit_id
        kwargs['proj_col_sub_sampling'] = cfg.trafo.proj_col_sub_sampling
    else:
        raise ValueError
    return get_ray_trafo(cfg.dataset.name, kwargs=kwargs)

def get_standard_dataset(
        cfg: DictConfig, ray_trafo: BaseRayTrafo, fold: str = 'test',
        use_fixed_seeds_starting_from: Optional[int] = 1, device=None) -> Dataset:
    """
    Return a dataset of tuples ``noisy_observation, x, filtbackproj``, where
        * ``noisy_observation`` has shape ``(1,) + obs_shape``
        * ``x`` is the ground truth (label) and has shape ``(1,) + im_shape``
        * ``filtbackproj = FBP(noisy_observation)`` has shape ``(1,) + im_shape``

    Parameters
    ----------
    fold : str, optional
        Dataset fold, either ``'test'`` or ``'validation'``.
        Only the (K)MNIST datasets support ``'validation'``, using the respective training set.
        The default is ``'test'``.
    use_fixed_seeds_starting_from : int, optional
        Fixed seed for noise generation, only used in simulated datasets.
        If ``fold == 'validation'``, ``1000000`` is added to the seed (if not ``None``).
    device : str or torch.device, optional
        If specified, data will be moved to the device. ``ray_trafo``
        (including ``ray_trafo.fbp``) must support tensors on the device.

    Returns
    -------
    dataset : torch.utils.data.Dataset
        Dataset.
    """
    assert fold in ('test', 'validation')

    if fold == 'validation' and use_fixed_seeds_starting_from is not None:
        use_fixed_seeds_starting_from = use_fixed_seeds_starting_from + 1000000

    if cfg.dataset.name == 'mnist':

        image_dataset = get_mnist_testset() if fold == 'test' else get_mnist_trainset()
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=cfg.dataset.noise_stddev,
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)

    elif cfg.dataset.name == 'kmnist':

        image_dataset = get_kmnist_testset() if fold == 'test' else get_kmnist_trainset()
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=cfg.dataset.noise_stddev,
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)

    elif cfg.dataset.name == 'rectangles':

        image_dataset = RectanglesDataset(
                (cfg.dataset.im_size, cfg.dataset.im_size),
                num_rects=cfg.dataset.num_rects,
                num_angle_modes=cfg.dataset.num_angle_modes,
                angle_modes_sigma=cfg.dataset.angle_modes_sigma,
                fixed_seed=1 if fold == 'test' else 1000001)
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=cfg.dataset.noise_stddev,
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)

    elif cfg.dataset.name == 'walnut':

        if fold == 'validation':
            raise ValueError('Walnut dataset has no validation fold implemented.')

        noisy_observation = get_walnut_2d_observation(
                data_path=os.path.join(get_original_cwd(), cfg.dataset.data_path),
                walnut_id=cfg.dataset.walnut_id, orbit_id=cfg.trafo.orbit_id,
                angular_sub_sampling=cfg.trafo.angular_sub_sampling,
                proj_col_sub_sampling=cfg.trafo.proj_col_sub_sampling,
                scaling_factor=cfg.dataset.scaling_factor).to(device=device)
        ground_truth = get_walnut_2d_ground_truth(
                data_path=os.path.join(get_original_cwd(), cfg.dataset.data_path),
                walnut_id=cfg.dataset.walnut_id, orbit_id=cfg.trafo.orbit_id,
                scaling_factor=cfg.dataset.scaling_factor).to(device=device)
        filtbackproj = ray_trafo.fbp(
                noisy_observation[None].to(device=device))[0].to(device=device)
        dataset = TensorDataset(  # include batch dims
                noisy_observation[None], ground_truth[None], filtbackproj[None])

    else:
        raise ValueError

    return dataset

def get_predefined_patch_idx_list(name: str, patch_size: int) -> List[int]:
    """
    Return a predefined list of patch indices.

    Parameters
    ----------
    name : str
        Name of the patch index list.
    patch_size : int
        Side length of the patches (patches are usually square).

    Returns
    -------
    patch_idx_list : list of int
        Indices of the patches.
    """
    if name == 'walnut_inner':
        patch_idx_list = get_walnut_2d_inner_patch_indices(patch_size=patch_size)
    else:
        raise ValueError(f'Unknown patch_idx_list configuration: {name}')
    return patch_idx_list

def assert_sample_matches(data_sample, path, i, raise_if_file_not_found=True) -> None:
    """
    Assert that the saved data for sample ``i`` in ``path`` matches the passed ``data_sample``.

    Parameters
    ----------
    data_sample : 3-tuple of Tensor
        Sample data ``(observation, ground_truth, filtbackproj)``. Only ``filtbackproj`` is used.
    path : str
        Hydra output directory of a previous run.
    i : int
        Sample index.
    raise_if_file_not_found : bool, optional
        If ``False``, warn instead of raising a ``FileNotFoundError``. The default is ``True``.
    """
    _, _, filtbackproj = data_sample
    try:
        sample_dict = torch.load(os.path.join(path, f'sample_{i}.pt'), map_location='cpu')
        assert torch.allclose(
                sample_dict['filtbackproj'].float(), filtbackproj.cpu().float(), atol=1e-6)
    except FileNotFoundError as e:
        if raise_if_file_not_found:
            raise e
        warn(f'Did not find sample {i} in {path}, so could not verify it matches.')

def save_samples(i: int, samples: Tensor, chunk_size: int, prefix: str = '') -> None:
    """
    Save samples to file(s) in the current working directory.

    The files are named ``f'{prefix}samples_{i}_chunk_{j}.pt'`` (where ``j`` is the chunk index).

    Parameters
    ----------
    i : int
        Data sample index.
    samples : Tensor
        Samples. Shape: ``(num_samples, ...)``.
    chunk_size : int
        Number of samples per file.
    prefix : str, optional
        If specified, prefix each filename with this string.
    """
    for j, start_i in enumerate(range(0, len(samples), chunk_size)):
        sample_chunk = samples[start_i:start_i+chunk_size].clone()
        torch.save(sample_chunk, f'{prefix}samples_{i}_chunk_{j}.pt')

def load_samples(
        path: str, i: int, num_samples: int, restrict_to_num_samples=True, prefix: str = '',
        map_location='cpu') -> Tensor:
    """
    Load samples from file(s) in ``path`` that were saved by :func:`save_samples`.

    Parameters
    ----------
    path : str
        Path containing the samples file(s).
    i : int
        Data sample index.
    num_samples : int
        Minimum number of samples to load.
        If ``restrict_to_num_samples`` or the number of saved samples is divisible by the chunk
        size, this is the number of returned samples.
    restrict_to_num_samples : bool, optional
        Whether to restrict the loaded samples to the first ``num_samples`` samples; otherwise more
        samples may be returned (due to the chunk size of the saved files).
        The default is ``True``.
    prefix : str, optional
        Prefix of the filename(s).

    Returns
    -------
    samples : Tensor
        Samples. Shape: ``(eff_num_samples, ...)``, where
        ``eff_num_samples`` is ``num_samples`` if ``restrict_to_num_samples`` or
        ``ceil(num_samples / chunk_size) * chunk_size`` otherwise.
    """
    sample_chunks = []
    num_loaded_samples = 0
    j = 0
    while num_loaded_samples < num_samples:
        try:
            chunk = torch.load(
                    os.path.join(path, f'{prefix}samples_{i}_chunk_{j}.pt'),
                    map_location=map_location)
        except FileNotFoundError as e:
            raise RuntimeError(
                    f'Failed to load {num_samples} samples from {path}, '
                    f'only found {num_loaded_samples}.') from e
        sample_chunks.append(chunk)
        num_loaded_samples += chunk.shape[0]
        j += 1
    samples = torch.cat(sample_chunks)
    if restrict_to_num_samples:
        samples = samples[:num_samples]
    return samples
