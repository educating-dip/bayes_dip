from typing import Sequence
import torch
from torch import autograd
from torch import Tensor

from ..probabilistic_models import BaseImageCov
from ..utils import batch_tv_grad

def compute_log_hyperparams_grads(params_list_under_GPpriors: Sequence,
        first_derivative_grads: Sequence,
        second_derivative_grads: Sequence,
        scaling: float
        ):

    assert len(first_derivative_grads) == len(params_list_under_GPpriors)

    grads = {}
    for i, param in enumerate(params_list_under_GPpriors):
        grads[param] = -(-first_derivative_grads[i] + second_derivative_grads[i]) * scaling
    return grads

def sample_based_predcp_grads(
        image_cov: BaseImageCov,
        params_list_under_predcp: Sequence,
        image_mean: Tensor,
        num_samples: int = 100,
        scale: float = 1.,
        return_shifted_loss: bool = True):

    x_samples, weight_samples = image_cov.sample(
        num_samples=num_samples,
        return_weight_samples=True,
        mean=image_mean,
        )

    # the mean of weight_samples does not change the gradients (the mean is zero here)

    with torch.no_grad():
        tv_x_samples = batch_tv_grad(x_samples)
        jac_tv_x_samples = image_cov.lin_op_transposed(tv_x_samples)

    shifted_loss = (weight_samples * jac_tv_x_samples).sum(dim=1).mean(dim=0)
    first_derivative_grads = autograd.grad(
        shifted_loss,
        params_list_under_predcp,
        allow_unused=True,
        create_graph=True,
        retain_graph=True
    )

    log_dets = [grad.abs().log() for grad in first_derivative_grads]
    second_derivative_grads = [
        autograd.grad(log_det, log_params, allow_unused=True, retain_graph=True)[0]
        for log_det, log_params in zip(log_dets, params_list_under_predcp)]

    with torch.no_grad():
        grads = compute_log_hyperparams_grads(
                params_list_under_predcp, first_derivative_grads, second_derivative_grads, scale)
        shifted_loss = scale * (shifted_loss - torch.stack(log_dets).sum().detach())

    return (grads, shifted_loss) if return_shifted_loss else grads

# TODO do not pass mean
