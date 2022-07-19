from typing import Sequence
import torch
from torch import Tensor
import torch.autograd as autograd
from ..probabilistic_models import ObservationCov

def batch_tv_grad(x: Tensor):

    assert x.shape[-1] == x.shape[-2]
    batch_size = x.shape[0]
    sign_diff_x = torch.sign(torch.diff(-x, n=1, dim=-1))
    pad = torch.zeros((batch_size, 1, x.shape[-2], 1), device = x.device)
    diff_x_pad = torch.cat([pad, sign_diff_x, pad], dim=-1)
    grad_tv_x = torch.diff(diff_x_pad, n=1, dim=-1)
    sign_diff_y = torch.sign(torch.diff(-x, n=1, dim=-2))
    pad = torch.zeros((batch_size, 1, 1, x.shape[-1]), device = x.device)
    diff_y_pad = torch.cat([pad, sign_diff_y, pad], dim=-2)
    grad_tv_y = torch.diff(diff_y_pad, n=1, dim=-2)
    
    return grad_tv_x + grad_tv_y

def compute_log_hyperparams_grads(observation_cov: ObservationCov,
        first_derivative_grads: Sequence,
        second_derivative_grads, scaling: Sequence
        ):
    
    image_cov_parameters = list(observation_cov.image_cov.parameters())
    assert len(first_derivative_grads) == len(image_cov_parameters)

    grads = {}
    for i, param in enumerate(image_cov_parameters): 
        grads[param] =  - (-first_derivative_grads[i] + second_derivative_grads[i]) * scaling
    return grads


def set_sample_based_predcp_grads(observation_cov: ObservationCov, 
        num_samples: int = 100,
        scale: float = 1.):

    x_samples, weight_samples = observation_cov.image_cov.sample(
        num_samples=num_samples,
        return_weight_samples=True
        )
    with torch.no_grad():
        tv_x_samples = batch_tv_grad(x_samples)
        jac_tv_x_samples = observation_cov.image_cov.lin_op_transposed(tv_x_samples)
    
    loss = (weight_samples * jac_tv_x_samples).sum(dim=1).mean(dim=0)
    first_derivative_grads = autograd.grad(
        loss,
        observation_cov.image_cov.parameters(),
        allow_unused=True,
        create_graph=True,
        retain_graph=True
    )

    log_dets = [grad.abs().log() for grad in first_derivative_grads]
    second_derivative_grads = [autograd.grad(log_det,
        log_params, allow_unused=True, retain_graph=True)[0] for log_det, log_params in zip(log_dets, observation_cov.image_cov.parameters())]
    
    with torch.no_grad():
        grads = compute_log_hyperparams_grads(observation_cov, first_derivative_grads, second_derivative_grads, scale)
        loss = scale * (loss - torch.stack(log_dets).sum().detach())
    return grads, loss 
