"""
Provides sample based gradient estimation for the PredCP term, :func:`sample_based_predcp_grads`.
"""
from typing import Sequence, Dict, Tuple, Union
import torch
from torch import autograd, nn
from torch import Tensor

from ..probabilistic_models import (
        BaseImageCov, ImageCov, ParameterCov, GPprior, BaseMatmulNeuralBasisExpansion)
from ..utils import batch_tv_grad

def sample_based_predcp_grads(
        image_cov: BaseImageCov,
        prior_list_under_predcp: Sequence[GPprior],
        image_mean: Tensor,
        num_samples: int = 100,
        scale: float = 1.,
        return_shifted_loss: bool = True,
        ) -> Union[Tuple[Dict[nn.Parameter, Tensor], Tensor], Dict[nn.Parameter, Tensor]]:
    """
    Estimate PredCP gradients.

    Assumes that each prior in ``prior_list_under_predcp`` has distinct parameters (i.e. no shared
    parameter between priors).

    Parameters
    ----------
    image_cov : :class:`.BaseImageCov`
        Image space covariance module. ``image_cov.inner_cov`` must be a :class:`.ParameterCov`
        instance.
    prior_list_under_predcp : sequence of :class:`bayes_dip.probabilistic_models.GPprior`
        GP priors for whose hyperparameters (``log_lengthscale`` and ``log_variance``) gradients
        are computed by this function.
    image_mean : Tensor
        Mean of the Gaussian image distribution (with covariance ``image_cov``).
    num_samples : int, optional
        Number of image samples to use for the gradient estimation. The default is ``100``.
    scale : float, optional
        Scaling factor; should usually be chosen proportional to the value ``optim_kwargs['gamma']``
        passed to :meth:`bayes_dip.dip.DeepImagePriorReconstructor.reconstruct`.
        The default is ``1.``.
    return_shifted_loss : bool, optional
        Whether to return a loss value; note that the value is not the PredCP loss, but an
        equivalent shifted version of it. The default is ``True``.

    Returns
    -------
    grads : dict
        Gradient dictionary, with :class:`torch.nn.Parameter` instances as keys and gradient tensors
        as values.
    total_shifted_loss : Tensor, optional
        Sum of shifted loss values for the priors in ``prior_list_under_predcp``.
        Only returned if ``return_shifted_loss``.
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
