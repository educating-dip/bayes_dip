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
    ):
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
                    ((i+1) % optim_kwargs['linear_cg']['update_freq']) == 0 and
                    (optim_kwargs['linear_cg']['preconditioner'] is not None)):
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
