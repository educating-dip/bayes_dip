
import torch
from ..probabilistic_models import GPprior

def get_ordered_nn_params_vec(parameter_cov):

    ordered_nn_params_vec = []
    for param in parameter_cov.ordered_nn_params:
        ordered_nn_params_vec.append(param.data.flatten())

    return torch.cat(ordered_nn_params_vec, )

def get_params_list_under_GPpriors(parameter_cov):

    params = []
    for prior_type, priors in parameter_cov.priors_per_prior_type.items():
        if prior_type is GPprior:
            for prior in priors:
                params.extend(list(prior.parameters()))
    return params
