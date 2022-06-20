"""
Provides walnut projection data and ground truth.
"""
import torch
from torch import Tensor
from bayes_dip.data.walnut_utils import (
        get_projection_data, get_single_slice_ray_trafo,
        get_single_slice_ind, get_ground_truth)


DEFAULT_WALNUT_SCALING_FACTOR = 14.


def get_walnut_2d_observation(
        data_path: str,
        walnut_id: int = 1, orbit_id: int = 2,
        angular_sub_sampling: int = 1,
        proj_col_sub_sampling: int = 1,
        scaling_factor: float = DEFAULT_WALNUT_SCALING_FACTOR) -> Tensor:
    """
    Return walnut projection data.

    Parameters
    ----------
    data_path : str
        Walnut dataset path (containing `'Walnut1'` as a subfolder).
    walnut_id : int, optional
        Walnut ID, an integer from 1 to 42.
        The default is `1`.
    orbit_id : int, optional
        Orbit (source position) ID, options are `1`, `2` or `3`.
        The default is `2`.
    angular_sub_sampling : int, optional
        Sub-sampling factor for the angles.
        The default is `1` (no sub-sampling).
    proj_col_sub_sampling : int, optional
        Sub-sampling factor for the projection columns.
        The default is `1` (no sub-sampling).
    scaling_factor : float, optional
        Scaling factor to multiply with.
        The default is `DEFAULT_WALNUT_SCALING_FACTOR`, scaling image values to
        approximately `[0., 1.]`.
    """

    walnut_kwargs = dict(
            walnut_id=walnut_id, orbit_id=orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling)

    observation_full = get_projection_data(
            data_path=data_path, **walnut_kwargs)

    # WalnutRayTrafo instance needed for selecting and masking the projections
    walnut_ray_trafo = get_single_slice_ray_trafo(
            data_path=data_path, **walnut_kwargs)

    observation = walnut_ray_trafo.flat_projs_in_mask(
            walnut_ray_trafo.projs_from_full(observation_full))[None]

    if scaling_factor != 1.:
        observation *= scaling_factor

    return torch.from_numpy(observation)[None]  # add channel dim


def get_walnut_2d_ground_truth(
        data_path: str,
        walnut_id: int = 1, orbit_id: int = 2,
        scaling_factor: float = DEFAULT_WALNUT_SCALING_FACTOR) -> Tensor:
    """
    Return walnut ground truth slice.

    Parameters
    ----------
    data_path : str
        Walnut dataset path (containing `'Walnut1'` as a subfolder).
    walnut_id : int, optional
        Walnut ID, an integer from 1 to 42.
        The default is `1`.
    orbit_id : int, optional
        Orbit (source position) ID, options are `1`, `2` or `3`.
        The default is `2`.
    scaling_factor : float, optional
        Scaling factor to multiply with.
        The default is `DEFAULT_WALNUT_SCALING_FACTOR`, scaling image values to
        approximately `[0., 1.]`.
    """

    slice_ind = get_single_slice_ind(
            data_path=data_path,
            walnut_id=walnut_id, orbit_id=orbit_id)
    ground_truth = get_ground_truth(
            data_path=data_path,
            walnut_id=walnut_id,
            slice_ind=slice_ind)

    if scaling_factor != 1.:
        ground_truth *= scaling_factor

    return torch.from_numpy(ground_truth)[None]  # add channel dim
