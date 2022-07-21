from typing import Union, Dict
import os
import socket
import datetime
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
import tensorboardX
from .observation_cov_log_det_grad import approx_observation_cov_log_det_grads
from .sample_based_predcp import set_sample_based_predcp_grads
from .weights_linearization import weights_linearization
from .utils import get_ordered_nn_params_vec, get_params_list_under_GPpriors
from ..probabilistic_models import ObservationCov, BaseGaussPrior, GPprior, NormalPrior

def marginal_likelihood_hyperparams_optim(
    observation_cov: ObservationCov,
    observation: Tensor,
    recon: Tensor,
    ground_truth: Tensor = None, 
    use_linearized_weights: Union[bool, Tensor] = True,
    optim_kwargs: Dict = None,
    log_path: str = './',
    comment: str = ''
    ):

    writer = tensorboardX.SummaryWriter(
            logdir=os.path.join(log_path, '_'.join((
                    datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    socket.gethostname(),
                    'marginal_likelihood_hyperparams_optim' + comment))))

    proj_recon = observation_cov.trafo(recon).flatten()
    observation = observation.flatten()

    if use_linearized_weights:
        weights_vec, _ = weights_linearization(
            observation_cov=observation_cov,
            observation=observation,
            ground_truth=ground_truth, 
            optim_kwargs=optim_kwargs['linearize_weights']
        )
    else:
        weights_vec = get_ordered_nn_params_vec(observation_cov.image_cov.inner_cov)

    optimizer = torch.optim.Adam(observation_cov.parameters(), lr=optim_kwargs['lr'])
    if optim_kwargs['include_predcp']:
        params_list_under_predcp = get_params_list_under_GPpriors(observation_cov.image_cov.inner_cov)

    with tqdm(range(optim_kwargs['iterations']), desc='marginal_likelihood_hyperparams_optim', miniters=optim_kwargs['iterations']//100) as pbar:
        for i in pbar:

            optimizer.zero_grad()

            if optim_kwargs['include_predcp']:
                predcp_grads, predcp_loss = set_sample_based_predcp_grads(
                    observation_cov=observation_cov,
                    params_list_under_predcp=params_list_under_predcp,
                    num_samples=100,
                    scale=1.)
                
                for param in params_list_under_predcp:
                    if param.grad is None:
                        param.grad = predcp_grads[param]
                    else:
                        param.grad += predcp_grads[param]
            else:
                predcp_loss = torch.zeros(1)

            if optim_kwargs['compute_exact_logdet']:
                sign, log_det = torch.linalg.slogdet(observation_cov.observation_cov_matrix)
                assert sign > 0. 
            else: 
                # update grads for post_hess_log_det
                log_det_grads, log_det_residual_norm = approx_observation_cov_log_det_grads(
                    observation_cov=observation_cov,
                    precon=optim_kwargs['linear_cg']['preconditioner'],
                    num_probes=optim_kwargs['num_probes'], 
                    max_cg_iter=optim_kwargs['linear_cg']['max_iter'],
                    cg_rtol=optim_kwargs['linear_cg']['rtol'], 
                    ignore_numerical_warning=True
                )

                for param in observation_cov.parameters():
                    if param.grad is None:
                        param.grad = log_det_grads[param]
                    else:
                        param.grad += log_det_grads[param]

            observation_error_norm = torch.sum((observation-proj_recon) ** 2) * torch.exp(-observation_cov.log_noise_variance)
            weights_prior_norm = (observation_cov.image_cov.inner_cov(weights_vec[None], use_inverse=True) @ weights_vec[None].T)
            loss = 0.5 * (observation_error_norm + weights_prior_norm)
            if optim_kwargs['compute_exact_logdet']: 
                loss = loss + 0.5 * log_det

            loss.backward(retain_graph=True)
            optimizer.step()
            
            if not optim_kwargs['compute_exact_logdet']:
                if ((i+1) % optim_kwargs['linear_cg']['update_freq']) == 0 and (optim_kwargs['linear_cg']['preconditioner'] is not None):
                    optim_kwargs['linear_cg']['preconditioner'].update()

            if optim_kwargs['min_log_variance'] != -np.inf:
                for log_variance in observation_cov.image_cov.inner_cov.log_variances:
                    log_variance.data.clamp_(min=optim_kwargs['min_log_variance'])

            if (i+1) % 200 == 0:
                torch.save(optimizer.state_dict(),
                    './optimizer_{}_iter_{}.pt'.format(comment, i))
                torch.save(observation_cov.state_dict(),
                    './observation_cov_{}_iter_{}.pt'.format(comment, i))

            for prior_type, priors in observation_cov.image_cov.inner_cov.priors_per_prior_type.items():

                if issubclass(prior_type, GPprior):
                    prior_type_name = 'GPprior'
                elif issubclass(prior_type, NormalPrior):
                    prior_type_name = 'NormalPrior'
                else:
                    prior_type_name = None

                for k, prior in enumerate(priors):
                    if issubclass(prior_type, BaseGaussPrior):
                        writer.add_scalar(f'{prior_type_name}_variance_{k}', torch.exp(prior.log_variance).item(), i)
                        if issubclass(prior_type, GPprior):
                            writer.add_scalar(f'{prior_type_name}_lengthscale_{k}', torch.exp(prior.log_lengthscale).item(), i)

            writer.add_scalar('observation_error_norm', observation_error_norm.item(), i)
            writer.add_scalar('weights_prior_norm', weights_prior_norm.item(), i)
            writer.add_scalar('predcp', - predcp_loss.item(), i)
            writer.add_scalar('observation_noise_variance', torch.exp(observation_cov.log_noise_variance).item(), i)
            if not optim_kwargs['compute_exact_logdet']:
                writer.add_scalar('log_det_grad_cg_mean_residual', log_det_residual_norm.mean().item(), i)

