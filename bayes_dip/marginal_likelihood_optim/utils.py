"""
Provides general utilities for the marginal likelihood optimization.
"""
import torch

def get_ordered_nn_params_vec(parameter_cov):

    ordered_nn_params_vec = []
    for param in parameter_cov.ordered_nn_params:
        ordered_nn_params_vec.append(param.data.flatten())

    return torch.cat(ordered_nn_params_vec, )
