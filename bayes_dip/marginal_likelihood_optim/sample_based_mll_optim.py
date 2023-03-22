"""
Provides the kernelised sampling-based linearised NN inference routine for 
gprior hyperparameter, :func:``sample_based_marginal_likelihood_optim``.
"""
from typing import Dict, Optional
import os
import socket
import datetime
import torch
import numpy as np
import tensorboardX
from tqdm import tqdm
from torch import Tensor
from .sample_based_mll_optim_utils import (
        PCG_based_weights_linearization, sample_then_optim_weights_linearization,
        sample_then_optimise, estimate_effective_dimension, gprior_variance_mackay_update,
        debugging_loglikelihood_estimation, debugging_histogram_tensorboard,
        debugging_uqviz_tensorboard
    )
from bayes_dip.utils import get_mid_slice_if_3d
from bayes_dip.utils import PSNR, SSIM, normalize
from bayes_dip.inference import SampleBasedPredictivePosterior

def sample_based_marginal_likelihood_optim(
    predictive_posterior: SampleBasedPredictivePosterior,
    map_weights: Tensor, 
    observation: Tensor,
    nn_recon: Tensor,
    ground_truth: Tensor,
    optim_kwargs: Dict,
    log_path: str = './',
    em_start_step: int = 0,
    posterior_obs_samples_sq_sum: Optional[Dict] = None,
    ):

    '''
    Kernelised sampling-based linearised NN inference.
    ``sample_based_marginal_likelihood_optim`` implements Algo. 3 
    in https://arxiv.org/abs/2210.04994.
    '''

    writer = tensorboardX.SummaryWriter(
        logdir=os.path.join(log_path, '_'.join((
            datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            socket.gethostname(),
            'marginal_likelihood_sample_based_hyperparams_optim')))
        )

    writer.add_image('nn_recon.', normalize(get_mid_slice_if_3d(nn_recon)[0]), 0)
    writer.add_image('ground_truth', normalize(get_mid_slice_if_3d(ground_truth)[0]), 0)
    observation_cov = predictive_posterior.observation_cov

    with torch.no_grad():

        scale = observation_cov.image_cov.neural_basis_expansion.scale.pow(-1)
        scale_corrected_map_weights = scale*map_weights
        recon_offset = - nn_recon + observation_cov.image_cov.neural_basis_expansion.jvp(
            scale_corrected_map_weights[None, :])
        observation_offset = observation_cov.trafo(recon_offset)
        observation_for_lin_optim = observation + observation_offset
        
        linearized_weights = None
        weight_sample = None

        if posterior_obs_samples_sq_sum is not None:
            assert optim_kwargs['iterations'] == 1, 'Only one iteration is allowed when resuming from a checkpoint.'

        with tqdm(range(em_start_step, em_start_step + optim_kwargs['iterations']), desc='sample_based_marginal_likelihood_optim') as pbar:
            for i in pbar:
                if not optim_kwargs['use_sample_then_optimise']:
                    linearized_weights, linearized_observation, linearized_recon = PCG_based_weights_linearization(
                        observation_cov=observation_cov, 
                        observation=observation_for_lin_optim, 
                        cg_kwargs=optim_kwargs['sample_kwargs']['cg_kwargs'],
                    )
                else:
                    wd = observation_cov.image_cov.inner_cov.priors.gprior.log_variance.exp().pow(-1)
                    optim_kwargs['sample_kwargs']['weights_linearisation']['optim_kwargs'].update({'wd': wd})
                    use_warm_start = optim_kwargs['sample_kwargs']['weights_linearisation']['optim_kwargs']['use_warm_start']
                    with torch.enable_grad():
                        linearized_weights, linearized_recon = sample_then_optim_weights_linearization(
                            trafo=observation_cov.trafo, 
                            neural_basis_expansion=observation_cov.image_cov.neural_basis_expansion, 
                            map_weights=scale_corrected_map_weights, 
                            observation=observation_for_lin_optim, 
                            optim_kwargs=optim_kwargs['sample_kwargs']['weights_linearisation']['optim_kwargs'],
                            aux={'ground_truth': ground_truth, 'recon_offset': recon_offset},
                            init_at_previous_weights=linearized_weights if use_warm_start else None,
                            )

                linearized_observation = observation_cov.trafo.trafo(linearized_recon)
                linearized_recon = linearized_recon - recon_offset.squeeze(dim=0)
                linearized_observation = linearized_observation - observation_offset
                
                if posterior_obs_samples_sq_sum is None:
                    if not optim_kwargs['use_sample_then_optimise']:
                        image_samples = predictive_posterior.sample_zero_mean(
                            num_samples=optim_kwargs['num_samples'],
                            **optim_kwargs['sample_kwargs']
                            )
                    else:
                        use_warm_start = optim_kwargs['sample_kwargs']['hyperparams_update']['optim_kwargs']['use_warm_start']
                        weight_sample = sample_then_optimise(
                            observation_cov=observation_cov,
                            neural_basis_expansion=observation_cov.image_cov.neural_basis_expansion, 
                            noise_variance=observation_cov.log_noise_variance.exp().detach(), 
                            variance_coeff=observation_cov.image_cov.inner_cov.priors.gprior.log_variance.exp().detach(), 
                            num_samples=optim_kwargs['num_samples'],
                            optim_kwargs=optim_kwargs['sample_kwargs']['hyperparams_update']['optim_kwargs'],
                            init_at_previous_samples=weight_sample if use_warm_start else None,
                            )
                        
                        torch.save(weight_sample, f'weight_sample_iter_{i}.pt')
                        # Zero mean samples.
                        image_samples = observation_cov.image_cov.neural_basis_expansion.jvp(weight_sample).squeeze(dim=1)

                    obs_samples = observation_cov.trafo(image_samples)
                    posterior_obs_samples_sq_mean = obs_samples.pow(2).sum(dim=0) / obs_samples.shape[0]
                else:
                    posterior_obs_samples_sq_mean = posterior_obs_samples_sq_sum['value'] / posterior_obs_samples_sq_sum['num_samples']

                eff_dim = estimate_effective_dimension(posterior_obs_samples_sq_mean=posterior_obs_samples_sq_mean, 
                        noise_variance=observation_cov.log_noise_variance.exp().detach()
                        ).clamp(min=1, max=np.prod(observation_cov.trafo.obs_shape)-1)
                
                variance_coeff = gprior_variance_mackay_update(
                    eff_dim=eff_dim, map_linearized_weights=linearized_weights
                    )
                observation_cov.image_cov.inner_cov.priors.gprior.log_variance = variance_coeff.log()
                se_loss = (linearized_observation-observation).pow(2).sum()

                if not optim_kwargs['use_sample_then_optimise'] and optim_kwargs['cg_preconditioner'] is not None:
                    optim_kwargs['cg_preconditioner'].update()
                
                torch.save(
                    observation_cov.state_dict(), 
                    f'observation_cov_iter_{i}.pt'
                )
                torch.save(
                    variance_coeff, 
                    f'gprior_variance_iter_{i}.pt'
                )

                writer.add_scalar('variance_coeff', variance_coeff.item(), i)
                writer.add_scalar('noise_variance', observation_cov.log_noise_variance.data.exp().item(), i)
                writer.add_image('linearized_model_recon', normalize(get_mid_slice_if_3d(linearized_recon)[0]), i)
                writer.add_scalar('effective_dimension', eff_dim.item(), i)
                writer.add_scalar('se_loss', se_loss.item(), i)

                if optim_kwargs['activate_debugging_mode'] and posterior_obs_samples_sq_sum is None:
                    if optim_kwargs['use_sample_then_optimise']:
                        print('Log-likelihood is calculated using previous samples, and only mll_optim.num_samples are used.')
                    loglik_nn_model, image_samples_diagnostic = debugging_loglikelihood_estimation(
                        predictive_posterior=predictive_posterior,
                        mean=get_mid_slice_if_3d(nn_recon),
                        ground_truth=get_mid_slice_if_3d(ground_truth),
                        image_samples=None if not optim_kwargs['use_sample_then_optimise'] else image_samples,
                        sample_kwargs=optim_kwargs['sample_kwargs'],
                        loglikelihood_kwargs=optim_kwargs['debugging_mode_kwargs']['loglikelihood_kwargs']
                    )
                    loglik_lin_model, _ = debugging_loglikelihood_estimation(
                        predictive_posterior=predictive_posterior,
                        mean=get_mid_slice_if_3d(linearized_recon),
                        ground_truth=get_mid_slice_if_3d(ground_truth),
                        image_samples=get_mid_slice_if_3d(image_samples_diagnostic),
                        loglikelihood_kwargs=optim_kwargs['debugging_mode_kwargs']['loglikelihood_kwargs']
                    )
                    writer.add_image('debugging_histogram_nn_model', debugging_histogram_tensorboard(
                        get_mid_slice_if_3d(ground_truth), get_mid_slice_if_3d(nn_recon), 
                        get_mid_slice_if_3d(image_samples_diagnostic))[0], i)
                    writer.add_image('debugging_histogram_lin_model', debugging_histogram_tensorboard(
                        get_mid_slice_if_3d(ground_truth), get_mid_slice_if_3d(linearized_recon), 
                        get_mid_slice_if_3d(image_samples_diagnostic))[0], i)
                    writer.add_image('debugging_histogram_uqviz_nn_model', debugging_uqviz_tensorboard(
                        get_mid_slice_if_3d(ground_truth), get_mid_slice_if_3d(nn_recon), 
                        get_mid_slice_if_3d(image_samples_diagnostic))[0], i)
                    writer.add_scalar('loglik_nn_model',  loglik_nn_model.item(), i)
                    writer.add_scalar('loglik_lin_model', loglik_lin_model.item(), i)

                    if optim_kwargs['debugging_mode_kwargs']['verbose']:
                    
                        print('\n\033[1m' + f'iter: {i}, variance_coeff: {variance_coeff.item():.2E}, ',\
                            f'noise_variance: {observation_cov.log_noise_variance.data.exp().item():.2E}, ',\
                            f'eff_dim: {eff_dim.item():.2E}, se_loss: {se_loss.item():.2E} ',\
                            f'l2: {linearized_weights.pow(2).sum().item():.2E}' + '\033[0m')
                        print('\033[1m' + f'iter: {i}, linearized_recon PSNR: {PSNR(linearized_recon.cpu().numpy(), ground_truth.cpu().numpy()):.2E}, '\
                            f'SSIM: {SSIM(linearized_recon.cpu().numpy()[0, 0], ground_truth.cpu().numpy()[0, 0]):.2E}' + '\033[0m')
                        print('\033[1m' + f'iter: {i}, loglik_nn_model: {loglik_nn_model:.2E}, loglik_lin_model: {loglik_lin_model:.2E}\n' + '\033[0m')

    return linearized_weights, linearized_recon