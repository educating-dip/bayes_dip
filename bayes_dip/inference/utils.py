import numpy as np
import torch

def get_image_patch_slices(image_shape, patch_size):
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

def get_image_patch_mask_inds(image_shape, patch_size, flatten=True):
    patch_slices_0, patch_slices_1 = get_image_patch_slices(image_shape, patch_size)

    patch_mask_inds = []
    for slice_0 in patch_slices_0:
        for slice_1 in patch_slices_1:
            mask_inds = np.ravel_multi_index(np.mgrid[slice_0,slice_1], image_shape)
            if flatten:
                mask_inds = mask_inds.flatten()
            patch_mask_inds.append(mask_inds)
    return patch_mask_inds

def yield_padded_batched_images_patches(images, patch_size, patch_idx_list=None, batch_size=1, return_patch_numels=False):
    assert images.shape[1] == 1
    assert images.ndim == 4
    all_patch_mask_inds = get_image_patch_mask_inds(tuple(images.shape[2:]), patch_size=patch_size, flatten=True)
    if patch_idx_list is None:
        patch_idx_list = list(range(len(all_patch_mask_inds)))
    for j in range(0, len(patch_idx_list), batch_size):
        batch_patch_inds = patch_idx_list[j:j+batch_size]

        batch_len_mask_inds = [len(all_patch_mask_inds[patch_idx]) for patch_idx in batch_patch_inds]
        max_len_mask_inds = max(batch_len_mask_inds)

        batch_samples_patches = torch.stack([
                torch.nn.functional.pad(images.view(images.shape[0], -1)[:, all_patch_mask_inds[patch_idx]], (0, max_len_mask_inds - len_mask_inds))
                for patch_idx, len_mask_inds in zip(batch_patch_inds, batch_len_mask_inds)])

        if return_patch_numels:
            yield batch_patch_inds, batch_samples_patches, batch_len_mask_inds
        else:
            yield batch_patch_inds, batch_samples_patches

def is_invalid(x):
    batch_invalid_values = torch.sum(torch.logical_not(torch.isfinite(x.view(x.shape[0], -1))), dim=1) != 0
    return batch_invalid_values
