"""
Provides :func:`approx_observation_cov_log_det_grads`.
"""
from typing import Dict, Tuple
import torch
from torch import nn
from torch import Tensor
from ..probabilistic_models import ObservationCov, LinearSandwichCov
from .random_probes import generate_probes_bernoulli
from .preconditioner import BasePreconditioner
from ..utils import cg

def approx_observation_cov_log_det_grads(
        observation_cov: ObservationCov,
        precon: BasePreconditioner = None,
        max_cg_iter: int = 50,
        cg_rtol: float = 1e-3,
        num_probes: int = 1,
        use_log_re_variant: bool = False,
        use_preconditioned_probes: bool = False,
        ignore_numerical_warning: bool = False
        ) -> Tuple[Dict[nn.Parameter, Tensor], Tensor]:
    """
    Estimates the gradient for the log-determinant ``0.5*log|observation_cov|`` w.r.t. its
    parameters via Hutchinson's trace estimator
    ``E(0.5 * v.T @ observation_cov**-1 @ d observation_cov / d params @ v)``,
    with ``v.T @ observation_cov**-1`` being approximated by the conjugate gradient (CG) method.

    Parameters
    ----------
    observation_cov : :class:`ObservationCov`
        Observation covariance.
    precon : :class:`BasePreconditioner`, optional
        Left-preconditioner.
    max_cg_iter : int, optional
        Maximum number of CG iterations. The default is ``50``.
    cg_rtol : float, optional
        Tolerance at which to stop early (before ``max_iter``). The default is ``1e-3``.
    num_probes : int, optional
        Number of probes to use for the trace estimator. The default is ``1``.
    use_log_re_variant : bool, optional
        Whether to use the low precision arithmetic variant by Maddox et al.,
        :meth:`linear_log_cg_re`. The default is ``False``.
    use_preconditioned_probes : bool, optional
        Whether to use preconditioned probes, as described in Section 4.1 in [1]_.
        If ``True``, ``precon`` must not be ``None``. The default is ``False``.

        .. [1] J.R. Gardner, G. Pleiss, D. Bindel, K.Q. Weinberger, A.G. Wilson, 2018,
               "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU
               Acceleration". https://arxiv.org/pdf/1809.11165v6.pdf
    ignore_numerical_warning : bool, optional
        Not implemented yet. Should control whether numerical warnings are ignored.
        The default is ``False``.

    Returns
    -------
    grads : dict
        Gradient dictionary, with :class:`torch.nn.Parameter` instances as keys and gradient tensors
        as values.
    residual_norm : Tensor
        Residual norm of the CG solution.
    """
    # pylint: disable=too-many-locals

    trafo = observation_cov.trafo
    image_cov = observation_cov.image_cov
    log_noise_variance = observation_cov.log_noise_variance

    image_cov_parameters = list(image_cov.parameters())
    # observation_cov.parameters() should consist of image_cov_parameters and log_noise_variance
    assert len(list(observation_cov.parameters())) == len(image_cov_parameters) + 1

    assert isinstance(image_cov, LinearSandwichCov)
    # image_cov == image_cov.lin_op @ image_cov.inner_cov @ image_cov.lin_op_transposed
    # => d image_cov / d params ==
    #    image_cov.lin_op @ d image_cov.inner_cov / d params @ image_cov.lin_op_transposed

    def observation_cov_closure(v):
        return observation_cov(v.T.reshape(num_probes, 1, *observation_cov.trafo.obs_shape)
                ).view(num_probes, observation_cov.shape[0]).T

    precon_closure = None if precon is None else precon.get_closure()
    if not use_preconditioned_probes:
        v_flat = generate_probes_bernoulli(
            side_length=observation_cov.shape[0],
            num_probes=num_probes,
            device=observation_cov.device,
            jacobi_vector=None)  # (obs_numel, num_probes)
    else:
        assert precon is not None
        # The preconditioned BBMM is here used, proposed in [1]_. Refer to section 4.1 as well as
        # Appendix D.

        # .. [1] Gardner, J., Pleiss, G., Weinberger, K.Q., Bindel, D. and Wilson, A.G.,
        #         2018. "Gpytorch: Blackbox matrix-matrix gaussian process inference with
        #         gpu acceleration."
        #         Advances in neural information processing systems, 31.
        #         https://arxiv.org/pdf/1809.11165.pdf

        v_flat = precon.sample(num_samples=num_probes)

    grads = {}

    ## gradients for parameters in image_cov
    with torch.no_grad():
        v_obs_left_flat, residual_norm = cg(
                observation_cov_closure, v_flat, precon_closure=precon_closure,
                max_niter=max_cg_iter, rtol=cg_rtol, use_log_re_variant=use_log_re_variant,
                ignore_numerical_warning=ignore_numerical_warning
            )
        v_left = trafo.trafo_adjoint_flat(v_obs_left_flat)  # (im_numel, num_probes)
        v_left = v_left.T.reshape(num_probes, 1, *trafo.im_shape)
        v_left = image_cov.lin_op_transposed(v_left)  # (num_probes, nn_params_numel)
        # v_left = v.T @ observation_cov**-1 @ trafo @ lin_op

        if use_preconditioned_probes:
            v_flat = precon_closure(v_flat)
        v_right = trafo.trafo_adjoint_flat(v_flat)
        v_right = v_right.T.reshape(num_probes, 1, *trafo.im_shape)
        v_right = image_cov.lin_op_transposed(v_right) # (num_probes, nn_params_numel)
        # v_right = lin_op_transposed @ trafo_adjoint @ v

    # estimate expected value E(v_left @ d image_cov.inner_cov / d params @ v_right.T)
    v_scalar = 0.5 * torch.sum(image_cov.inner_cov(v_left) * v_right, dim=1).mean()
    # (scalar product over network params)
    image_cov_grads = torch.autograd.grad((v_scalar,), image_cov_parameters)
    grads.update(dict(zip(image_cov_parameters, image_cov_grads)))

    ## gradient for log_noise_variance

    # estimate expected value E(exp(v_obs_left_flat.T @ v_flat))
    noise_scalar = .5 * torch.sum(v_obs_left_flat * v_flat, dim=0).mean() * log_noise_variance.exp()

    # (scalar product over observation)
    grads[log_noise_variance] = torch.autograd.grad((noise_scalar,), (log_noise_variance,))[0]

    return grads, residual_norm
