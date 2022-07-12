from typing import Optional, Callable, Dict
import torch
from torch import Tensor
from ..probabilistic_models import ObservationCov, LinearSandwichCov
from .random_probes import generate_probes_bernoulli

def linear_cg(
        observation_cov: ObservationCov,
        v: Tensor,
        preconditioner: Optional[Callable] = None,
        ) -> Tensor:
    raise NotImplementedError

def approx_observation_cov_log_det_grads(
        observation_cov: ObservationCov,
        preconditioner=None,
        num_probes=1,
        device=None,
        ) -> Dict[Tensor]:
    """
    Estimates the gradient for the log-determinant ``log|observation_cov|`` w.r.t. its parameters
    via Hutchinson's trace estimator
    ``E(v.T @ observation_cov**-1 @ d observation_cov / d params @ v)``,
    with ``v.T @ observation_cov**-1`` being approximated by the conjugate gradient (CG) method.
    """
    trafo = observation_cov.trafo
    image_cov = observation_cov.image_cov

    parameters = list(observation_cov.parameters())
    image_cov_parameters = list(image_cov.parameters())
    assert len(parameters) == len(image_cov.parameters()) + 1  # log_noise_variance

    assert isinstance(image_cov, LinearSandwichCov)
    # image_cov == image_cov.lin_op @ image_cov.inner_cov @ image_cov.lin_op_transposed
    # => d image_cov / d params ==
    #    image_cov.lin_op @ d image_cov.inner_cov / d params @ image_cov.lin_op_transposed

    v_flat = generate_probes_bernoulli(
            side_length=observation_cov.shape[0],
            num_probes=num_probes,
            device=observation_cov.device,
            jacobi_vector=None)  # (obs_numel, num_probes)

    grads = {}

    ## gradients for parameters in image_cov

    v_obs_left_flat = linear_cg(observation_cov, v_flat, preconditioner=preconditioner)
    v_im_left_flat = trafo.trafo_adjoint_flat(v_obs_left_flat)  # (im_numel, num_probes)
    v_left = v_im_left_flat.T.reshape(1, num_probes, *trafo.im_shape)
    v_left = image_cov.lin_op_transposed(v_left)  # (num_probes, nn_params_numel)
    # v_left = v.T @ observation_cov**-1 @ trafo @ lin_op

    v_right = trafo.trafo_adjoint_flat(v_flat)
    v_right = v_right.T.reshape(1, num_probes, *trafo.im_shape)
    v_right = image_cov.lin_op_transposed(v_right)  # (num_probes, nn_params_numel)
    # v_right = lin_op_transposed @ trafo_adjoint @ v

    # estimate expected value E(v_left @ d image_cov.inner_cov / d params @ v_right.T)
    v_scalar = torch.sum(image_cov.inner_cov(v_left) * v_right, dim=1).mean()
    # (scalar product over network params)
    image_cov_grads = torch.autograd.grad((v_scalar,), image_cov_parameters)
    for param, grad in zip(image_cov_parameters, image_cov_grads):
        grads[param] = grad

    ## gradient for log_noise_variance

    # estimate expected value E(exp(v_obs_left_flat.T @ v_flat))
    log_noise_variance_grad = torch.sum(v_obs_left_flat * v_flat, dim=0).mean().exp()
    # (scalar product over observation)
    grads[observation_cov.log_noise_variance] = log_noise_variance_grad

    return grads
