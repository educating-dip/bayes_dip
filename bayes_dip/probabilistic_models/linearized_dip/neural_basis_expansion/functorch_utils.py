"""
Utilities for using the flattened parameter representation from :class:`BaseNeuralBasisExpansion`
with functorch.
"""
from typing import Sequence, Tuple
import torch
from torch import nn, Tensor

def flatten_grad_functorch(
        inds_from_ordered_params: Sequence[int],
        grads: Tuple[Tensor]
        ) -> Tensor:
    """
    Flatten and concatenate selected tensors from a tuple as returned from functorch.

    Parameters
    ----------
    inds_from_ordered_params : sequence of int
        Indices of the selected parameters in ``grads`` (which is usually ordered like the full list
        of parameters of a model, ``list(nn_model.parameters())``).
    grads : sequence of Tensor
        Gradient tensors.

    Returns
    -------
    flat_grads : Tensor
        Flat and concatenated gradients. Shape: ``(total_selected_num_params,)``, i.e.
        ``sum(grads[ind].numel() for ind in inds_from_ordered_params)``.
    """
    flat_grads = torch.cat([grads[ind].detach().reshape(-1) for ind in inds_from_ordered_params])
    return flat_grads

def unflatten_nn_functorch(
        nn_model: nn.Module,
        inds_from_ordered_params: Sequence[int],
        slices_from_ordered_params: Sequence[slice],
        weights: Tensor,
        ) -> Tuple[Tensor]:
    """
    Unpack flat and concatenated selected parameters into a tuple that can be passed to functorch.

    For the parameters that are not selected by ``inds_from_ordered_params``, zero tensors are
    placed in the returned tuple.

    Parameters
    ----------
    nn_model : :class:`nn.Module`
        Network.
    inds_from_ordered_params : sequence of int
        Indices of the selected parameters in ``list(nn_model.parameters())``.
    slices_from_ordered_params : sequence of slice
        Slice objects to sub-slice ``weights``, in the same order as ``inds_from_ordered_params``.
    weights : Tensor
        Flat and concatenated selected parameters. Shape: ``(total_selected_num_params,)``, i.e.
        ``sum(list(nn_model.parameters())[ind].numel() for ind in inds_from_ordered_params)``.

    Returns
    -------
    weights_tuple : tuple of Tensor
        Weights tuple, in the order of ``nn_model.parameters()``.
        Values for the parameters selected by ``inds_from_ordered_params`` are unpacked from
        ``weights``; for the other parameters, zeros are inserted.
    """
    params = list(nn_model.parameters())
    weight_list = [None] * len(params)

    for ind, slice_param in zip(inds_from_ordered_params, slices_from_ordered_params):
        weight_list[ind] = weights[slice_param].view(*params[ind].shape)

    for ind, weight in enumerate(weight_list):
        if weight is None:
            weight_list[ind] = torch.zeros(
                    params[ind].shape, dtype=torch.float, device=weights.device)

    return tuple(weight_list)
