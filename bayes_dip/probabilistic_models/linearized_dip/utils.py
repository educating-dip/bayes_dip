"""
Provides utilities specific to the Linearized DIP.
"""
from typing import List, Sequence
from torch import nn

def get_inds_from_ordered_params(
        nn_model: nn.Module, ordered_nn_params: Sequence[nn.Parameter]) -> List[int]:
    """
    Return the indices into ``list(nn_model.parameters())`` that result in ``ordered_nn_params``.

    Parameters
    ----------
    nn_model : :class:`nn.Module`
        Network.
    ordered_nn_params : sequence of nn.Parameter
        Sequence of parameters.

    Returns
    -------
    inds_in_full_model : list of int
        Indices into ``list(nn_model.parameters())`` that result in ``ordered_nn_params``.
    """
    params = list(nn_model.parameters())

    inds_in_full_model = []
    for param in ordered_nn_params:
        inds_in_full_model.append(
                next(i for i, p in enumerate(params) if p is param))

    return inds_in_full_model

def get_slices_from_ordered_params(ordered_nn_params: Sequence[nn.Parameter]) -> List[slice]:
    """
    Return slice objects corresponding to the individual parameters in a flattened and concatenated
    vector of all parameters.

    Parameters
    ----------
    ordered_nn_params : sequence of nn.Parameter
        Sequence of parameters.

    Returns
    -------
    slices : list of slice
        Slice objects corresponding to the individual parameters in a flattened and concatenated
        vector of all parameters. The slice step is always ``None``.
    """
    slices = []
    w_pointer = 0
    for param in ordered_nn_params:
        slices.append(
            slice(w_pointer, w_pointer + param.data.numel())
            )
        w_pointer += param.data.numel()

    return slices
