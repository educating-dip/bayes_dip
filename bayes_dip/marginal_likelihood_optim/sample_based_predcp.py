from typing import Sequence, Union
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
        weight_mean: Union[Tensor, float, None] = None,
        return_loss: bool = True):

    x_samples, weight_samples = image_cov.sample(
        num_samples=num_samples,
        return_weight_samples=True,
        mean=image_mean,
        )

    # weight_samples return from image_cov has zero mean, so add weight_mean;
    # it affects only the loss, not the gradients
    assert not (return_loss and weight_mean is None), (
            '`weight_mean` required for loss computation. To use zero weight mean, pass '
            '``weight_mean=0.`` and make sure `image_mean` is consistent with it; alternatively, '
            'pass ``return_loss=False`` to disable loss computation '
            '(gradients will not be affected by `weight_mean`).')
    if weight_mean is not None:
        weight_samples = weight_samples + weight_mean

    with torch.no_grad():
        tv_x_samples = batch_tv_grad(x_samples)
        jac_tv_x_samples = image_cov.lin_op_transposed(tv_x_samples)

    loss = (weight_samples * jac_tv_x_samples).sum(dim=1).mean(dim=0)
    first_derivative_grads = autograd.grad(
        loss,
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
        loss = scale * (loss - torch.stack(log_dets).sum().detach())

    return (grads, loss) if return_loss else grads
