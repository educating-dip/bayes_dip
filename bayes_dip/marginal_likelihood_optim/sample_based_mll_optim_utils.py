from typing import Dict, Optional, Tuple
import io
import os
import socket
import datetime
import PIL.Image
import torch
import tensorboardX
from torch import nn, Tensor
import matplotlib.pyplot as plt
from torch import Tensor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms import ToTensor
from tqdm import tqdm
from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from bayes_dip.utils import cg
from bayes_dip.probabilistic_models import ObservationCov, BaseNeuralBasisExpansion
from bayes_dip.inference import SampleBasedPredictivePosterior, get_image_patch_mask_inds
from bayes_dip.utils import eval_mode, get_mid_slice_if_3d, PSNR
from bayes_dip.utils.experiment_utils import get_predefined_patch_idx_list
from bayes_dip.utils.plot_utils import configure_matplotlib, plot_hist
import functorch

def PCG_based_weights_linearization(
    observation_cov: ObservationCov,
    observation: Tensor,
    cg_kwargs: Dict
    ) -> Tuple[Tensor, Tensor, Tensor]:

    '''
    It solves for the exact solution of linearised weights $\bar{\theta}$.
    .. math::
        \begin{align}
            \bar{\theta} &= \Sigma_\theta \tilde{J}^\top A^\top \left( A \tilde{J} \Sigma_\theta \tilde{J}^\top A^\top + \sigma^2 I \right)^{-1} \tilde{y} \\
            \bar{f} &= \tilde{J} \bar{\theta} \\,
            \bar{y} &= A \tilde{J} \bar{\theta}
        \end{align}
    where, 
    .. math::
        \begin{itemize}
            \item $\tilde{J} = sJ$ is the re-scaled Jacobian matrix. 
            \item $\Sigma_{\theta}^{-1} = g^{-1} I_{d \times d}$ is the weight prior precision matrix. 
            \item $B = \sigma^{-2} I$. 
            \item $\tilde{y} = y + A ( -f^* + \tilde{J} ( s^{-1} \theta^*) )$, where $f^*$ and $\theta^*$ are the NN reconstruction and weights. 
        \end{itemize}

    Refer to https://arxiv.org/abs/2210.04994 for an in-dept analysis.
    '''

    def observation_cov_closure(
        v: Tensor
        ):
        return observation_cov(v.T.reshape(
                batch_size, 1, *observation_cov.trafo.obs_shape)).view(
                        batch_size, observation_cov.shape[0]).T
    with torch.no_grad():
        batch_size = 1 # batch_size fixed to 1
        transposed_samples, _ = cg(observation_cov_closure, 
                observation.flatten()[:, None], **cg_kwargs
            )
        samples = transposed_samples.transpose(1, 0) # Sigma_{yy}^{-1} \tilde{y}
        samples = observation_cov.trafo.trafo_adjoint(
                samples.view(
                        batch_size, 1, *observation_cov.trafo.obs_shape)
                    ) # A^\top Sigma_{yy}^{-1} \tilde{y}
        samples = observation_cov.image_cov.lin_op_transposed(samples) # J^\top A^\top Sigma_{yy}^{-1} \tilde{y}
        # \bar{\theta} = \Sigma_{\theta\theta} J^\top A^\top Sigma_{yy}^{-1} \tilde{y}
        linearized_weights = observation_cov.image_cov.inner_cov(samples) 
        # \bar{f} =  \tilde{J} \bar{\theta}
        lin_recon = observation_cov.image_cov.lin_op(linearized_weights)
        # \bar{y} = A \bar{f}
        lin_observation = observation_cov.trafo(lin_recon)

    return linearized_weights.flatten(), lin_observation, lin_recon

def sample_then_optim_weights_linearization(
        trafo: BaseRayTrafo,
        neural_basis_expansion: BaseNeuralBasisExpansion,
        map_weights: Tensor,
        observation: Tensor,
        optim_kwargs: dict,
        aux,
        init_at_previous_weights: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:
    # pylint: disable=too-many-locals

    ''' 
    It solves for the linearised weights using the convex objective for solving the linear model. 
    
    .. math::
        \begin{align*}
            \mathcal{L}(\theta) &= \frac{1}{2} \left\| A \left(f^* + \tilde{J} (\theta - s^{-1} \theta^*) \right) - y \right\|^2_{B} + \frac{1}{2} \| \theta \|^2_{\Sigma_\theta^{-1}} \\
            &= \frac{1}{2} \left\| A \tilde{J} \theta - \tilde{y}  \right\|^2_{B} +  \frac{1}{2} \| \theta \|^2_{\Sigma_w^{-1}} 
        \end{align*}
    
    Taking the derivative to zero, and solving for the exact solution,
    
    .. math::
        \begin{align*}
            \frac{\partial{\mathcal{L}}}{\partial \theta} &= 2 B \tilde{J}^\top A^\top (A \tilde{J} \theta - \tilde{y}) + 2 \Sigma_\theta^{-1} \theta \\
            \bar{\theta} &= (B \tilde{J}^\top A^\top A \tilde{J} + \Sigma_\theta^{-1} )^{-1} B \tilde{J}^\top A^\top \tilde{y}
        \end{align*}
    '''
    def closure(
        lin_weights: Tensor,
        proj_lin_recon: Tensor,
        observation: Tensor,
        ):
        

        loss_fit = .5 * torch.nn.functional.mse_loss(
                proj_lin_recon, observation.view(*proj_lin_recon.shape), 
                reduction='sum'
            )
        loss_prior =  + .5 * optim_kwargs['wd'] * lin_weights.pow(2).sum()
        loss = loss_fit + loss_prior
        loss.backward()

        return loss, (loss_fit, loss_prior)


    writer = tensorboardX.SummaryWriter(
            logdir=os.path.join('./', '_'.join((
                datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                socket.gethostname(),
                'sample_then_optim_weights_linearization')))
            )

    nn_model = neural_basis_expansion.nn_model
    if init_at_previous_weights is not None:
        lin_weights = nn.Parameter(init_at_previous_weights)
    else:
        lin_weights = nn.Parameter(torch.zeros_like(map_weights))
    optimizer = torch.optim.SGD(
            [lin_weights], lr=optim_kwargs['lr'], weight_decay=0, momentum=optim_kwargs['momentum'], nesterov=True)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor = optim_kwargs['scheduler_kwargs']['start_factor'], 
        end_factor = optim_kwargs['scheduler_kwargs']['end_factor'],
        total_iters = int(optim_kwargs['iterations'] * optim_kwargs['scheduler_kwargs']['total_iters_red_pct']),
        )

    with tqdm(range(optim_kwargs['iterations']), miniters=optim_kwargs['iterations']//100) as pbar, \
            eval_mode(nn_model):
        for i in pbar:

            lin_recon = neural_basis_expansion.jvp(
                    lin_weights[None, :]).squeeze(dim=1)
            proj_lin_recon = trafo(lin_recon)
            observation = observation.view(*proj_lin_recon.shape)
            
            optimizer.zero_grad()
            _, (loss_fit, loss_prior) = closure(lin_weights=lin_weights, proj_lin_recon=proj_lin_recon, 
                    observation=observation
                    )
            
            writer.add_scalar('loss_fit', loss_fit.item(), i)
            writer.add_scalar('loss_prior', loss_prior.item(), i)

            if optim_kwargs['clip_grad_norm_value'] is not None:
                torch.nn.utils.clip_grad_norm_(
                    lin_weights, optim_kwargs['clip_grad_norm_value'])
            optimizer.step()
            scheduler.step()

            psnr = PSNR(lin_recon.detach().cpu().numpy() - aux['recon_offset'].cpu().numpy(), aux['ground_truth'].cpu().numpy())
            pbar.set_description(f'l2_norm lin_weights and PSNR: {lin_weights.pow(2).sum():.6f}, {psnr:.6f}', 
                    refresh=False
                )
    
    return lin_weights.detach(), lin_recon.detach()

def sample_then_optimise(
    observation_cov: ObservationCov,
    neural_basis_expansion: BaseNeuralBasisExpansion, 
    noise_variance: float,
    variance_coeff: float,
    num_samples: int,
    optim_kwargs: Dict,
    init_at_previous_samples: Optional[Tensor] = None,
    ):
    
    '''
    It samples from from the linear model's posterior using SGD. It samples following Eq. 7 in https://arxiv.org/abs/2210.04994. 
    '''
    
    def closure(
        trafo, 
        neural_basis_expansion, 
        weights_posterior_samples: Tensor,
        weights_sample_from_prior: Tensor,
        eps: Tensor,
        noise_variance: Tensor,
        variance_coeff: Tensor 
        ):
        proj_weights_posterior_samples = trafo(
            neural_basis_expansion.jvp(weights_posterior_samples).squeeze(dim=1)
            )
        loss_fit = .5 * (1 / noise_variance) * torch.nn.functional.mse_loss(
                proj_weights_posterior_samples, eps, 
                reduction='sum'
            ) 
        loss_prior =  .5 * variance_coeff * (weights_posterior_samples - weights_sample_from_prior).pow(2).sum()
        loss = loss_fit + loss_prior
    
        return loss, (loss_fit, loss_prior)

    
    writer = tensorboardX.SummaryWriter(
            logdir=os.path.join('./', '_'.join((
                datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                socket.gethostname(),
                'sample_then_optimise')))
            )

    if init_at_previous_samples is not None:
        weights_posterior_samples = nn.Parameter(init_at_previous_samples)
    else:
        weights_posterior_samples = nn.Parameter(
            torch.zeros(
                num_samples, neural_basis_expansion.num_params, 
                device=observation_cov.device
                )
            )

    factor = optim_kwargs['polyak_averaging_factor']
    if factor is not None:
        polyak_weights_posterior_samples = weights_posterior_samples.clone()

    optimizer = torch.optim.SGD(
        [weights_posterior_samples], lr=optim_kwargs['lr'], weight_decay=0, momentum=optim_kwargs['momentum'], nesterov=True)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor = optim_kwargs['scheduler_kwargs']['start_factor'], 
        end_factor = optim_kwargs['scheduler_kwargs']['end_factor'],
        total_iters = int(optim_kwargs['iterations'] * optim_kwargs['scheduler_kwargs']['total_iters_red_pct']),
        )
    
    weights_sample_from_prior = torch.randn(num_samples, neural_basis_expansion.num_params, 
            device=observation_cov.device) * torch.sqrt(variance_coeff) # prior sample
    eps = torch.randn(num_samples, 1, *observation_cov.trafo.obs_shape, 
            device=observation_cov.device) * torch.sqrt(noise_variance)

    theta_n = weights_sample_from_prior + (1. / variance_coeff) * noise_variance * neural_basis_expansion.vjp(
            observation_cov.trafo.trafo_adjoint(eps)[:, None, ...])
    
    with tqdm(range(optim_kwargs['iterations']), miniters=optim_kwargs['iterations']//100) as pbar:
        for i in pbar:
            
            # We compute gradients exactly, since functorch.vmap and autograd have issues with sparse trafo.
            sample_linear_recon = observation_cov.trafo.trafo(
                neural_basis_expansion.jvp(weights_posterior_samples))
            grad_1 = noise_variance * neural_basis_expansion.vjp(
                observation_cov.trafo.trafo_adjoint(sample_linear_recon)[:, None, ...])
            grad_2 = (variance_coeff) * (weights_posterior_samples - theta_n)

            optimizer.zero_grad()
            weights_posterior_samples.grad = grad_1 + grad_2

            # We only use closure to calculate loss for logging purposes and not to compute the gradients.
            _, (loss_fit, loss_prior) = closure(
                trafo=observation_cov.trafo, 
                neural_basis_expansion=neural_basis_expansion,
                weights_posterior_samples=weights_posterior_samples, 
                weights_sample_from_prior=weights_sample_from_prior, 
                eps=eps,
                noise_variance=noise_variance,
                variance_coeff=variance_coeff
                )
            
            writer.add_scalar('loss_fit', loss_fit.item(), i)
            writer.add_scalar('loss_prior', loss_prior.item(), i)

            if optim_kwargs['clip_grad_norm_value'] is not None:
                torch.nn.utils.clip_grad_norm_(
                    weights_posterior_samples, optim_kwargs['clip_grad_norm_value'])
            
            optimizer.step()
            if factor is not None:
                polyak_weights_posterior_samples = (1. - factor) * polyak_weights_posterior_samples + factor * weights_posterior_samples
            scheduler.step()

            if optim_kwargs['verbose']:
                image_sample = neural_basis_expansion.jvp(weights_posterior_samples).squeeze(dim=1)
                obs_samples = observation_cov.trafo(image_sample)
                posterior_obs_samples_sq_mean = obs_samples.pow(2).sum(dim=0) / obs_samples.shape[0]
                eff_dim = estimate_effective_dimension(
                    posterior_obs_samples_sq_mean=posterior_obs_samples_sq_mean,
                    noise_variance=noise_variance
                    )
                pbar.set_description(f'approx_eff_dim: {eff_dim.detach().cpu().numpy():.4f}', refresh=False)

    if factor is not None:
        return polyak_weights_posterior_samples.detach()
    else:
        return weights_posterior_samples.detach()
            

def estimate_effective_dimension(
    posterior_obs_samples_sq_mean: Tensor,
    noise_variance: float,
    ) -> float:

    return posterior_obs_samples_sq_mean.sum()*(1/noise_variance) 


def gprior_variance_mackay_update(
    eff_dim: float, 
    map_linearized_weights: Tensor
    ) -> float:

    weight_norm = map_linearized_weights.pow(2).sum()
    return weight_norm / eff_dim

# ------------------------------------------------------------------------------
# |                       Tensorboard debugging utilities                        |
# ------------------------------------------------------------------------------

def debugging_loglikelihood_estimation(
    predictive_posterior: SampleBasedPredictivePosterior,
    mean: Tensor,
    ground_truth: Tensor,
    loglikelihood_kwargs: Dict,
    image_samples: Optional[Tensor] = None,
    sample_kwargs: Optional[Dict] = None
    ):

    if image_samples is None:
        assert sample_kwargs is not None

        image_samples = predictive_posterior.sample_zero_mean(
            num_samples=loglikelihood_kwargs['num_samples'],
            **sample_kwargs
        )

    image_samples = get_mid_slice_if_3d(image_samples)
    all_patch_mask_inds = get_image_patch_mask_inds(
        image_samples.shape[2:], 
        patch_size=loglikelihood_kwargs['patch_kwargs']['patch_size'])
    patch_idx_list = loglikelihood_kwargs['patch_kwargs']['patch_idx_list']
    if patch_idx_list is None:
        patch_idx_list = list(range(len(all_patch_mask_inds)))
    elif isinstance(patch_idx_list, str):
        patch_idx_list = get_predefined_patch_idx_list(
            name=patch_idx_list, patch_size=loglikelihood_kwargs['patch_kwargs']['patch_size'])
    loglikelihood_kwargs['patch_kwargs']['patch_idx_list'] = patch_idx_list
    
    loglik = predictive_posterior.log_prob(
                mean=mean,
                ground_truth=ground_truth,
                samples=image_samples,
                patch_kwargs=loglikelihood_kwargs['patch_kwargs'],
                noise_x_correction_term=loglikelihood_kwargs['noise_x_correction_term'],
                verbose=loglikelihood_kwargs['verbose'],
                return_patch_diags=loglikelihood_kwargs['return_patch_diags'],
                unscaled=loglikelihood_kwargs['unscaled']
            )
    return loglik, image_samples

def plot_to_image(figure):

    '''
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    '''

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    return image

def debugging_uqviz_tensorboard(
    ground_truth: Tensor,
    recon: Tensor,
    samples: Tensor
    ): 
    
    configure_matplotlib()

    diff_abs = (ground_truth - recon).abs().cpu().numpy()[0, 0]
    stddev = (samples.pow(2).sum(dim=0) / samples.shape[0]).pow(0.5).cpu().numpy()[0]
    ratio = diff_abs / stddev
    _, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    ax_0 = axs[0].imshow(diff_abs)
    axs[0].set_title('$|x-x^*|$')
    axs[0].get_xaxis().set_ticks([])
    axs[0].get_yaxis().set_ticks([])
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(ax_0, cax=cax)
    ax_1 = axs[1].imshow(stddev)
    axs[1].set_title('std-dev')
    axs[1].get_xaxis().set_ticks([])
    axs[1].get_yaxis().set_ticks([])
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(ax_1,cax=cax)
    ax_2 = axs[2].imshow(ratio)
    axs[2].get_xaxis().set_ticks([])
    axs[2].get_yaxis().set_ticks([])
    axs[2].set_title('$|x-x^*|$ / std-dev')
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(ax_2, cax=cax)
    return plot_to_image(
            plt.gcf()
        )

def debugging_histogram_tensorboard(
    ground_truth: Tensor,
    recon: Tensor,
    samples: Tensor
    ):

    configure_matplotlib()
    def _get_xlim(data):
        return (0, max((d.max() for d in data)))
    def _get_ylim(n_list, ylim_min_fct=0.5):
        ylim_min = ylim_min_fct * min(n[n > 0].min() for n in n_list)
        ylim_max = max(n.max() for n in n_list)
        return (ylim_min, ylim_max)

    abs_diff = (recon - ground_truth).abs()
    stddev = (samples.pow(2).sum(dim=0) / samples.shape[0]).pow(0.5)
    data = [d.flatten().cpu().numpy() for d in [abs_diff, stddev]]
    label_list = ['$|x-x^*|$','std-dev (MLL)']
    ax, n_list, _ = plot_hist(data=data, label_list=label_list, remove_ticks=False)
    ax.set_xlim(_get_xlim(data))
    ax.set_ylim(_get_ylim(n_list))

    return plot_to_image(
            plt.gcf()
        )