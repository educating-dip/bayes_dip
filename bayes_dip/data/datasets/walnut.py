"""
Provides walnut projection data and ground truth.
"""
from typing import List, Tuple
from math import ceil
import torch
from torch import Tensor
from bayes_dip.data.walnut_utils import (
        get_projection_data, get_single_slice_ray_trafo,
        get_single_slice_ind, get_ground_truth, VOL_SZ)


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
        Walnut dataset path (containing ``'Walnut1'`` as a subfolder).
    walnut_id : int, optional
        Walnut ID, an integer from 1 to 42.
        The default is ``1``.
    orbit_id : int, optional
        Orbit (source position) ID, options are ``1``, ``2`` or ``3``.
        The default is ``2``.
    angular_sub_sampling : int, optional
        Sub-sampling factor for the angles.
        The default is ``1`` (no sub-sampling).
    proj_col_sub_sampling : int, optional
        Sub-sampling factor for the projection columns.
        The default is ``1`` (no sub-sampling).
    scaling_factor : float, optional
        Scaling factor to multiply with.
        The default is ``DEFAULT_WALNUT_SCALING_FACTOR``, scaling image values to
        approximately ``[0., 1.]``.

    Returns
    -------
    observation : Tensor
        Projection data. Shape: ``(1, 1, obs_numel)``, where
        ``obs_numel = ceil(1200 / angular_sub_sampling) * ceil(768 / proj_col_sub_sampling)``.
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
        Walnut dataset path (containing ``'Walnut1'`` as a subfolder).
    walnut_id : int, optional
        Walnut ID, an integer from 1 to 42.
        The default is ``1``.
    orbit_id : int, optional
        Orbit (source position) ID, options are ``1``, ``2`` or ``3``.
        The default is ``2``.
    scaling_factor : float, optional
        Scaling factor to multiply with.
        The default is ``DEFAULT_WALNUT_SCALING_FACTOR``, scaling image values to
        approximately ``[0., 1.]``.

    Returns
    -------
    ground_truth : Tensor
        Ground truth. Shape: ``(1, 501, 501)``.
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


INNER_PART_START_0 = 72
INNER_PART_START_1 = 72
INNER_PART_END_0 = 424
INNER_PART_END_1 = 424

def _get_walnut_2d_inner_patch_slices(patch_size: int) -> Tuple[slice, slice]:
    start_patch_0 = INNER_PART_START_0 // patch_size
    start_patch_1 = INNER_PART_START_1 // patch_size
    end_patch_0 = ceil(INNER_PART_END_0 / patch_size)
    end_patch_1 = ceil(INNER_PART_END_1 / patch_size)
    patch_slice_0 = slice(start_patch_0, end_patch_0)
    patch_slice_1 = slice(start_patch_1, end_patch_1)
    return patch_slice_0, patch_slice_1

def get_walnut_2d_inner_patch_indices(patch_size: int) -> List[int]:
    """
    Return patch indices for the inner part of the walnut image (that contains the walnut)
    into the list returned by :func:`bayes_dip.inference.utils.get_image_patch_slices`.

    Parameters
    ----------
    patch_size : int
        Side length of the patches (patches are usually square).

    Returns
    -------
    patch_idx_list : list of int
        Indices of the patches.
    """
    num_patches_0 = VOL_SZ[1] // patch_size
    num_patches_1 = VOL_SZ[2] // patch_size
    patch_slice_0, patch_slice_1 = _get_walnut_2d_inner_patch_slices(patch_size)

    patch_idx_list = [
        patch_idx for patch_idx in range(num_patches_0 * num_patches_1)
        if patch_idx % num_patches_0 in range(num_patches_0)[patch_slice_0] and
        patch_idx // num_patches_0 in range(num_patches_1)[patch_slice_1]]

    return patch_idx_list

def get_walnut_2d_inner_part_defined_by_patch_size(patch_size: int) -> Tuple[slice, slice]:
    """
    Return a pair of slices specifying the inner part of the walnut image, which depends (to a minor
    extent) on the ``patch_size``, since the inner part is defined by patch indices into the list
    returned by :func:`bayes_dip.inference.utils.get_image_patch_slices`.

    Parameters
    ----------
    patch_size : int
        Side length of the patches (patches are usually square).
    """
    num_patches_0 = VOL_SZ[1] // patch_size
    num_patches_1 = VOL_SZ[2] // patch_size
    patch_slice_0, patch_slice_1 = _get_walnut_2d_inner_patch_slices(patch_size)
    slice_0 = slice(
            patch_slice_0.start * patch_size,
            (patch_slice_0.stop * patch_size if patch_slice_0.stop < num_patches_0 else VOL_SZ[1]))
    slice_1 = slice(
            patch_slice_1.start * patch_size,
            (patch_slice_1.stop * patch_size if patch_slice_1.stop < num_patches_1 else VOL_SZ[2]))
    return slice_0, slice_1
