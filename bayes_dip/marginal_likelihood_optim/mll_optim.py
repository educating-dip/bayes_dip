"""
Provides the marginal log-likelihood (MLL or Type-II-MAP) optimization routine for the prior
hyperparameters, :func:`marginal_likelihood_hyperparams_optim`.
"""
from typing import Iterable, Dict, Optional
import os
import socket
import datetime
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
import tensorboardX


from .observation_cov_log_det_grad import approx_observation_cov_log_det_grads
from .sample_based_predcp import sample_based_predcp_grads
from .utils import get_ordered_nn_params_vec
from ..probabilistic_models import (
        ObservationCov, MatmulObservationCov, BaseGaussPrior, GPprior, IsotropicPrior, NormalPrior)


def _get_prior_type_name(prior_type: type) -> Optional[str]:

    if issubclass(prior_type, GPprior):
        prior_type_name = 'GPprior'
    elif issubclass(prior_type, NormalPrior):
        prior_type_name = 'NormalPrior'
    elif issubclass(prior_type, IsotropicPrior):
        prior_type_name = 'GPrior'
    else:
        prior_type_name = None

    return prior_type_name


def _add_grads(params: Iterable, grad_dict: Dict) -> None:
    for param in params:
        if param.grad is None:
            param.grad = grad_dict[param]
        else:
            param.grad += grad_dict[param]


def _clamp_params_min(params: Iterable, min: float) -> None:
    # pylint: disable=redefined-builtin
    if min != -np.inf:
        for param in params:
            param.data.clamp_(min=min)


def marginal_likelihood_hyperparams_optim(
    observation_cov: ObservationCov,
    observation: Tensor,
    recon: Tensor,
    linearized_weights: Optional[Tensor] = None,
    optim_kwargs: Dict = None,
    log_path: str = './',
    comment: str = 'mll'
    ) -> None:
    """
    Optimize the prior hyperparameters by marginal log-likelihood (MLL or Type-II-MAP) optimization.

    Parameters
    ----------
    observation_cov : :class:`ObservationCov`
        Observation covariance.
    observation : Tensor
        Observation. Shape: ``(1, 1, *observation_cov.trafo.obs_shape)``.
    recon : Tensor
        Reconstruction. Shape: ``(1, 1, *observation_cov.trafo.im_shape)``.
    linearized_weights : Tensor, optional
        If specified, use these weights instead of the MAP weights (DIP network model weights).
        Useful to pass linearized weights like returned by
        :func:`bayes_dip.marginal_likelihood_optim.weights_linearization`.
        Shape: ``(observation_cov.image_cov.inner_cov.shape[0],)``.
    optim_kwargs : dict
        Optimization keyword arguments (most are required). The arguments are:

        ``'iterations'`` : int
            Number of iterations.
        ``'lr'`` : float
            Learning rate.
        ``'scheduler'`` : dict
            Scheduler keyword arguments.

            *Arguments in* ``optim_kwargs['scheduler']`` *are:*

            ``'use_scheduler'`` : bool
                Whether to use a :class:`torch.optim.lr_scheduler.StepLR` scheduler.
            ``'step_size'`` : int
                Step size of the scheduler.
            ``'gamma'`` : float
                Gamma of the scheduler.
        ``'min_log_variance'`` : float
            Minimum value for the logarithm of variance hyperparameters. The log variance
            hyperparameters are clamped by this value after each optimization step.
        ``'num_probes'`` : int
            Number of probes for estimating the observation covariance log determinant gradients
            if ``not isinstance(observation_cov, MatmulObservationCov)``; otherwise the gradients
            are calculated exactly from the assembled matrix via :func:`torch.linalg.slogdet`.
        ``'linear_cg'`` : dict
            Conjugate gradients keyword arguments for estimating the observation covariance log
            determinant gradients if ``not isinstance(observation_cov, MatmulObservationCov)``;
            otherwise the gradients are calculated exactly from the assembled matrix via
            :func:`torch.linalg.slogdet`.

            *Arguments in* ``optim_kwargs['linear_cg']`` *are:*

            ``'preconditioner'`` : :class:`BasePreconditioner` or None
                Left-preconditioner.
            ``'use_preconditioned_probes'`` : bool
                Whether to use preconditioned probes, as described in Section 4.1 in [1]_.
                If ``True``, the ``preconditioner`` must not be ``None``.

                .. [1] J.R. Gardner, G. Pleiss, D. Bindel, K.Q. Weinberger, A.G. Wilson, 2018,
                       "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU
                       Acceleration". https://arxiv.org/pdf/1809.11165v6.pdf
            ``'update_freq'`` : int
                Number of iterations between preconditioner updates.
            ``'max_iter'`` : int
                Maximum number of CG iterations.
            ``'rtol'`` : float
                Tolerance at which to stop early (before ``max_iter``).
            ``'use_log_re_variant'`` : bool
                Whether to use the low precision arithmetic variant by Maddox et al.,
                :meth:`linear_log_cg_re`.

        ``'include_predcp'`` : bool
            Whether to include the predictive complexity prior term.
        ``'predcp'`` : dict, optional
            PredCP keyword arguments, required if ``optim_kwargs['include_predcp']``.

            *Arguments in* ``optim_kwargs['predcp']`` *are:*

            ``'use_map_weights_mean'`` : bool
                If ``True``, use ``recon`` as the mean of the image samples;
                if ``False``, use ``recon - J @ map_weights`` instead, where
                ``J`` is the Jacobian of the network and ``map_weights`` are the network weights.
            ``'num_samples'`` : int
                Number of image samples for estimating the PredCP term gradients.
            ``'gamma'`` : float
                TV scaling factor, which is part of the scaling of the PredCP term.
                Should be the same as for the DIP optimization (the gamma values are comparable, as
                this function internally multiplies with ``observation.numel()`` because the
                likelihood objective uses the SSE instead of the MSE used for DIP optimization).
                See also ``optim_kwargs['predcp']['scale']``.
            ``'scale'`` : float
                Additional scaling factor for the PredCP term.
                See also ``optim_kwargs['predcp']['gamma']``.
    log_path : str, optional
        Path for saving tensorboard logs. This function creates a sub-folder in ``log_path``,
        starting with the current time. The default is ``'./'``.
    comment : str, optional
        Suffix for the tensorboard log sub-folder. The default is ``'mll'``.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements

    writer = tensorboardX.SummaryWriter(
            logdir=os.path.join(log_path, '_'.join((
                    datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    socket.gethostname(),
                    'marginal_likelihood_hyperparams_optim' + comment))))

    inner_cov = observation_cov.image_cov.inner_cov

    obs_sse = torch.sum((observation - observation_cov.trafo(recon)) ** 2)

    map_weights = get_ordered_nn_params_vec(inner_cov)

    weights_vec = linearized_weights if linearized_weights is not None else map_weights

    # f ~ N(predcp_recon_mean, image_cov)
    # let h be the linearization (first Taylor expansion) of nn_model around map_weights
    if optim_kwargs['include_predcp']:
        if optim_kwargs['predcp']['use_map_weights_mean']:
            # params ~ N(map_weights, parameter_cov)
            # E[f] == E[h(params)] == h(map_weights) == recon
            image_mean = recon
        else:
            # params ~ N(0, parameter_cov)
            # E[f] == E[h(params)] == h(0) == recon - J @ map_weights
            image_mean = recon - observation_cov.image_cov.lin_op(map_weights[None])

    optimizer = torch.optim.Adam(observation_cov.parameters(), lr=optim_kwargs['lr'])
    if optim_kwargs['scheduler']['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            step_size=optim_kwargs['scheduler']['step_size'],
            gamma=optim_kwargs['scheduler']['gamma'],
        )


    with tqdm(range(optim_kwargs['iterations']), desc='marginal_likelihood_hyperparams_optim',
            miniters=optim_kwargs['iterations']//100) as pbar:
        for i in pbar:

            optimizer.zero_grad()

            if optim_kwargs['include_predcp']:
                predcp_grads, predcp_shifted_loss = sample_based_predcp_grads(
                    image_cov=observation_cov.image_cov,
                    prior_list_under_predcp=inner_cov.priors_per_prior_type[GPprior],
                    image_mean=image_mean,
                    num_samples=optim_kwargs['predcp']['num_samples'],
                    scale=(optim_kwargs['predcp']['scale'] *
                            observation.numel() * optim_kwargs['predcp']['gamma']),
                    )

                params_under_predcp = []
                for prior_under_predcp in inner_cov.priors_per_prior_type[GPprior]:
                    params_under_predcp += list(prior_under_predcp.parameters())
                _add_grads(params=params_under_predcp, grad_dict=predcp_grads)
            else:
                predcp_shifted_loss = torch.zeros(1, device=observation_cov.device)

            loss = torch.zeros(1, device=observation_cov.device)

            if isinstance(observation_cov, MatmulObservationCov):
                sign, log_det = torch.linalg.slogdet(observation_cov.get_matrix(
                        apply_make_choleskable=True))
                assert sign > 0.

                loss = loss + 0.5 * log_det  # grads will be added in loss.backward() call
            else:
                # compute and add grads for post_hess_log_det manually
                log_det_grads, log_det_residual_norm = approx_observation_cov_log_det_grads(
                    observation_cov=observation_cov,
                    precon=optim_kwargs['linear_cg']['preconditioner'],
                    num_probes=optim_kwargs['num_probes'],
                    max_cg_iter=optim_kwargs['linear_cg']['max_iter'],
                    cg_rtol=optim_kwargs['linear_cg']['rtol'],
                    use_log_re_variant=optim_kwargs['linear_cg']['use_log_re_variant'],
                    use_preconditioned_probes=optim_kwargs['linear_cg']['use_preconditioned_probes']
                )

                _add_grads(params=observation_cov.parameters(), grad_dict=log_det_grads)

            observation_error_norm = obs_sse * torch.exp(-observation_cov.log_noise_variance)
            weights_prior_norm = (inner_cov(weights_vec[None], use_inverse=True) * weights_vec).sum(
                    dim=1).squeeze(0)
            loss = loss + 0.5 * (observation_error_norm + weights_prior_norm)

            loss.backward()
            optimizer.step()
            if optim_kwargs['scheduler']['use_scheduler']:
                scheduler.step()

            if (not isinstance(observation_cov, MatmulObservationCov) and
                    (optim_kwargs['linear_cg']['preconditioner'] is not None) and
                    ((i+1) % optim_kwargs['linear_cg']['update_freq']) == 0):
                optim_kwargs['linear_cg']['preconditioner'].update()

            _clamp_params_min(
                    params=inner_cov.log_variances, min=optim_kwargs['min_log_variance'])

            if (i+1) % 200 == 0:
                torch.save(optimizer.state_dict(),
                    f'optimizer_{comment}_iter_{i+1}.pt')
                torch.save(observation_cov.state_dict(),
                    f'observation_cov_{comment}_iter_{i+1}.pt')

            for prior_type, priors in inner_cov.priors_per_prior_type.items():

                prior_type_name = _get_prior_type_name(prior_type)

                for k, prior in enumerate(priors):
                    if issubclass(prior_type, (BaseGaussPrior, IsotropicPrior)):
                        writer.add_scalar(f'{prior_type_name}_variance_{k}',
                                torch.exp(prior.log_variance).item(), i)
                        if issubclass(prior_type, GPprior):
                            writer.add_scalar(f'{prior_type_name}_lengthscale_{k}',
                                    torch.exp(prior.log_lengthscale).item(), i)

            writer.add_scalar('observation_error_norm', observation_error_norm.item(), i)
            writer.add_scalar('weights_prior_norm', weights_prior_norm.item(), i)
            writer.add_scalar('predcp_shifted', -predcp_shifted_loss.item(), i)
            writer.add_scalar('observation_noise_variance',
                    torch.exp(observation_cov.log_noise_variance).item(), i)
            if not isinstance(observation_cov, MatmulObservationCov):
                writer.add_scalar('log_det_grad_cg_mean_residual',
                        log_det_residual_norm.mean().item(), i)
