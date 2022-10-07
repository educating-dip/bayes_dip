"""
Provides utilities for inference.

In particular, functionality for patch-wise evaluation is included.
"""
from typing import Optional, Dict, Tuple, List, Iterator, Union
import numpy as np
import torch
from torch import Tensor

def get_image_patch_slices(
        image_shape: Tuple[int, int], patch_size: int) -> Tuple[List[slice], List[slice]]:
    """
    Return slice objects defining patches of an image.

    The ``i``-th patch of a 2D ``image`` tensor is defined as
    ``image[patch_slices_0[i], patch_slices_1[i]]``.

    If an ``image_shape`` dimension is not divisible by ``patch_size``, the last patches along this
    dimension that would fit in the image are enlarged to also contain the remaining pixels in this
    dimension; patches can be non-square for this reason.

    Parameters
    ----------
    image_shape : 2-tuple of int
        Image shape.
    patch_size : int
        Side length of the patches (patches are usually square).
        It is clipped to the maximum value ``min(*image_shape)``.

    Returns
    -------
    patch_slices_0 : list of slice
        Slices in image dimension 0. The length is the number of patches.
    patch_slices_1 : list of slice
        Slices in image dimension 1. The length is the number of patches.
    """
    image_size_0, image_size_1 = image_shape
    patch_size = min(patch_size, min(*image_shape))

    patch_slices_0 = []
    for start_0 in range(0, image_size_0 - (patch_size-1), patch_size):
        if start_0 + patch_size < image_size_0 - (patch_size-1):
            end_0 = start_0 + patch_size
        else:
            # last full patch, also include the remaining pixels
            end_0 = image_size_0
        patch_slices_0.append(slice(start_0, end_0))
    patch_slices_1 = []
    for start_1 in range(0, image_size_1 - (patch_size-1), patch_size):
        if start_1 + patch_size < image_size_1 - (patch_size-1):
            end_1 = start_1 + patch_size
        else:
            # last full patch, also include the remaining pixels
            end_1 = image_size_1
        patch_slices_1.append(slice(start_1, end_1))
    return patch_slices_0, patch_slices_1

def get_image_patch_mask_inds(
        image_shape: Tuple[int, int], patch_size: int, flatten: bool = True) -> List[np.ndarray]:
    """
    Return mask indices defining patches of an image.

    The flattened ``i``-th patch of an ``image`` tensor is defined as ``image[patch_mask_inds[i]]``.

    If an ``image_shape`` dimension is not divisible by ``patch_size``, the last patches along this
    dimension that would fit in the image are enlarged to also contain the remaining pixels in this
    dimension; patches can be non-square for this reason.

    Parameters
    ----------
    image_shape : 2-tuple of int
        Image shape.
    patch_size : int
        Side length of the patches (patches are usually square).
        It is clipped to the maximum value ``min(*image_shape)``.
    flatten : bool, optional
        Whether to flatten each array in the returned ``patch_mask_inds``. The default is ``True``.

    Returns
    -------
    patch_mask_inds : list of array
        Mask indices for each patch. The length is the number of patches.
    """
    patch_slices_0, patch_slices_1 = get_image_patch_slices(image_shape, patch_size)

    patch_mask_inds = []
    for slice_0 in patch_slices_0:
        for slice_1 in patch_slices_1:
            mask_inds = np.ravel_multi_index(np.mgrid[slice_0,slice_1], image_shape)
            if flatten:
                mask_inds = mask_inds.flatten()
            patch_mask_inds.append(mask_inds)
    return patch_mask_inds

def yield_padded_batched_images_patches(
        images: Tensor, patch_kwargs: Optional[Dict] = None, return_patch_numels: bool = False
        ) -> Union[
                Iterator[Tuple[List[int], Tensor]],
                Iterator[Tuple[List[int], Tensor, List[int]]]]:
    """
    Yield batches of patches from images.

    The effective batch size (denote it by ``eff_batch_size``) is
    ``patch_kwargs.get('batch_size', 1)`` for all batches except for the potentially smaller last
    batch.

    Parameters
    ----------
    images : Tensor
        Images. Shape: ``(n, 1, *im_shape)``.
    patch_kwargs : dict, optional
        Keyword arguments specifying how to split the image into patches.

        The arguments are:
            ``'patch_size'`` : int, optional
                The default is ``1``.
            ``'patch_idx_list'`` : list of int, optional
                Patch indices. If ``None``, all patches are used.
            ``'batch_size'`` : int, optional
                The default is ``1``.
    return_patch_numels : bool, optional
        If ``True``, also return the number of pixels for each patch in the batch.
        The default is ``False``.

    Yields
    ------
    batch_patch_inds : list of int
        Patch indices. The length is ``eff_batch_size``.
    batch_samples_patches : Tensor
        Batch of patches from images.
        Shape: ``(eff_batch_size, num_samples, max(batch_len_mask_inds))``.
    batch_len_mask_inds : list of int, optional
        Number of pixels for each patch. Only returned if ``return_patch_numels``.
        These numbers can be used to remove the padding from the
        individual elements in the batch: ``batch_samples_patches[i, :, :batch_len_mask_inds[i]]``.
        The length is ``eff_batch_size``.
    """

    assert images.shape[1] == 1
    assert images.ndim == 4
    patch_kwargs = patch_kwargs or {}
    patch_kwargs.setdefault('patch_size', 1)
    patch_kwargs.setdefault('patch_idx_list', None)
    patch_kwargs.setdefault('batch_size', 1)

    all_patch_mask_inds = get_image_patch_mask_inds(
            tuple(images.shape[2:]), patch_size=patch_kwargs['patch_size'])
    if patch_kwargs['patch_idx_list'] is None:
        patch_kwargs['patch_idx_list'] = list(range(len(all_patch_mask_inds)))

    for j in range(0, len(patch_kwargs['patch_idx_list']), patch_kwargs['batch_size']):
        batch_patch_inds = patch_kwargs['patch_idx_list'][j:j+patch_kwargs['batch_size']]

        batch_len_mask_inds = [
                len(all_patch_mask_inds[patch_idx]) for patch_idx in batch_patch_inds]
        max_len_mask_inds = max(batch_len_mask_inds)

        batch_samples_patches = torch.stack([
                torch.nn.functional.pad(
                        images.view(images.shape[0], -1)[:, all_patch_mask_inds[patch_idx]],
                        (0, max_len_mask_inds - len_mask_inds))
                for patch_idx, len_mask_inds in zip(batch_patch_inds, batch_len_mask_inds)])

        if return_patch_numels:
            yield batch_patch_inds, batch_samples_patches, batch_len_mask_inds
        else:
            yield batch_patch_inds, batch_samples_patches

def is_invalid(x: Tensor) -> Tensor:
    """
    Return whether all numbers are finite per batch.

    Parameters
    ----------
    x : Tensor
        Tensor. Shape: ``(batch_size, ...)``.

    Returns
    -------
    batch_invalid_values : Tensor
        Boolean tensor specifying if all numbers are finite, batch-wise. Shape: ``(batch_size,)``.
    """
    batch_invalid_values = torch.sum(
            torch.logical_not(torch.isfinite(x.view(x.shape[0], -1))), dim=1) != 0
    return batch_invalid_values
