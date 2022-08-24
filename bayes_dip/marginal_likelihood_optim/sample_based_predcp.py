from typing import Sequence
import torch
from torch import autograd
from torch import Tensor

from ..probabilistic_models import (
        BaseImageCov, ImageCov, ParameterCov, GPprior, BaseMatmulNeuralBasisExpansion)
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
        prior_list_under_predcp: Sequence[GPprior],
        image_mean: Tensor,
        num_samples: int = 100,
        scale: float = 1.,
        return_shifted_loss: bool = True):
    """
    Compute PredCP gradients.

    Assumes that each prior in prior_list_under_predcp has distinct parameters (i.e. no shared
    parameter between priors).

    `image_cov.inner_cov` should be a ParameterCov instance.
    """

    assert isinstance(image_cov.inner_cov, ParameterCov)

    grads = {}
    total_shifted_loss = torch.zeros((1,), device=image_cov.inner_cov.device)

    lin_op_supports_sub_slicing = (
            isinstance(image_cov, ImageCov) and
            isinstance(image_cov.neural_basis_expansion, BaseMatmulNeuralBasisExpansion))

    for prior in prior_list_under_predcp:
        x_samples, weight_samples = image_cov.sample(
            num_samples=num_samples,
            return_weight_samples=True,
            mean=image_mean,
            sample_only_from_prior=prior,  # params under other priors assumed to be deterministic
            )

        with torch.no_grad():
            tv_x_samples = batch_tv_grad(x_samples)
            if lin_op_supports_sub_slicing:
                jac_tv_x_samples = image_cov.lin_op_transposed(
                        tv_x_samples, sub_slice=image_cov.inner_cov.params_slices_per_prior[prior])
            else:
                jac_tv_x_samples = image_cov.lin_op_transposed(
                        tv_x_samples)
                jac_tv_x_samples = jac_tv_x_samples[
                        :, image_cov.inner_cov.params_slices_per_prior[prior]]

        # could restrict weight_samples and jac_tv_x_samples to just the prior, since
        # dot product will be zero anyways

        shifted_loss = (weight_samples * jac_tv_x_samples).sum(dim=1).mean(dim=0)
        first_derivative_grads = autograd.grad(
            shifted_loss,
            (prior.log_lengthscale, prior.log_variance),
            allow_unused=True,
            create_graph=True,
            retain_graph=True
        )

        log_det = first_derivative_grads[0].abs().log()  # log_lengthscale
        second_derivative_grads = autograd.grad(
                log_det, (prior.log_lengthscale, prior.log_variance), retain_graph=True)

        with torch.no_grad():
            grads_for_prior = {
                prior.log_lengthscale:
                    -(-first_derivative_grads[0] + second_derivative_grads[0]) * scale,
                prior.log_variance:
                    -(-first_derivative_grads[1] + second_derivative_grads[1]) * scale,
                }
            shifted_loss = scale * (shifted_loss - log_det)

        assert all(param not in grads for param in grads_for_prior)
        grads.update(grads_for_prior)
        total_shifted_loss += shifted_loss

    return (grads, total_shifted_loss) if return_shifted_loss else grads
