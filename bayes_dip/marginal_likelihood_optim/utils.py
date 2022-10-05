"""
Provides general utilities for the marginal likelihood optimization.
"""
import torch
from torch import Tensor
from ..probabilistic_models import ParameterCov

def get_ordered_nn_params_vec(parameter_cov: ParameterCov) -> Tensor:
    """
    Return the flattened and concatenated network weights of the parameters under prior with the
    given :class:`ParameterCov` instance.

    Parameters
    ----------
    parameter_cov : :class:`ParameterCov`
        Parameter covariance.

    Returns
    -------
    ordered_nn_params_vec : Tensor
        Flattened and concatenated network weights. Shape: ``(parameter_cov.shape[0],)``.
    """

    ordered_nn_params_vec = []
    for param in parameter_cov.ordered_nn_params:
        ordered_nn_params_vec.append(param.data.flatten())

    return torch.cat(ordered_nn_params_vec)
