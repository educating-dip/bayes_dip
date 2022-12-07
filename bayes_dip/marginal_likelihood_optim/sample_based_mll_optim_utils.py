from typing import Dict, Optional
import io
import PIL.Image
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms import ToTensor

from bayes_dip.utils import cg
from bayes_dip.probabilistic_models import ObservationCov
from bayes_dip.inference import SampleBasedPredictivePosterior, get_image_patch_mask_inds
from bayes_dip.utils import get_mid_slice_if_3d
from bayes_dip.utils.experiment_utils import get_predefined_patch_idx_list
from bayes_dip.utils.plot_utils import configure_matplotlib, plot_hist

def PCG_based_weights_linearization(
    observation_cov: ObservationCov,
    observation: Tensor,
    cg_kwargs: Dict
    ):
    
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
        samples = transposed_samples.transpose(1, 0)
        samples = observation_cov.trafo.trafo_adjoint(
                samples.view(
                        batch_size, 1, *observation_cov.trafo.obs_shape)
                    )
        samples = observation_cov.image_cov.lin_op_transposed(samples)
        linearized_weights = observation_cov.image_cov.inner_cov(samples)
        lin_recon = observation_cov.image_cov.lin_op(linearized_weights)
        lin_observation = observation_cov.trafo(lin_recon)

    return linearized_weights.flatten(), lin_observation, lin_recon

def estimate_effective_dimension(
    posterior_obs_samples: Tensor,
    noise_variance: float,
    ) -> float:

    return posterior_obs_samples.pow(2).mean(dim=0).sum()*(1/noise_variance) 

def gprior_variance_mackay_update(
    eff_dim: float, 
    map_linerized_weights: Tensor
    ) -> float:

    weight_norm = map_linerized_weights.pow(2).sum()
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