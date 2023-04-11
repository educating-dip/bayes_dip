from typing import Dict, Optional, Any, Callable
import os
import io
import torch
import socket
import datetime
import numpy as np
import tensorboardX
import matplotlib.pyplot as plt

from copy import deepcopy
from math import ceil
from tqdm import tqdm
from torch import Tensor

from .sample_observations import sample_observations_shifted_bayes_exp_design
from .acq_criterions import find_optimal_proj
from .acq_state_tracker import AcqStateTracker
from .utils import eval_recon_on_new_acq_observation_data

from bayes_dip.utils import normalize, PSNR

# note: observation needs to have shape (len(init_angle_inds), num_projs_per_angle)
def bed_optimal_angles_search(
        acq_state_tracker: AcqStateTracker, 
        init_state_dict: Any,
        observation_full: Tensor,
        filtbackproj: Tensor,
        ground_truth: Tensor,
        bed_kwargs: Dict,
        dip_kwargs: Dict,
        criterion: str = 'EIG',
        model_basename: str = 'refined_dip_model',
        log_path: str = './',
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
        hyperparam_fun: Optional[Callable] = None, # hyperparam_fun(num_acq) -> (gamma, iterations)
        callbacks=(),
        logged_plot_callbacks: Optional[Any] = None,
        return_recons: bool = True,
    ):

    assert not (bed_kwargs['acquisition']['update_network_params'] and bed_kwargs['use_alternative_recon']), "cannot update network parameters when using alternative reconstruction method"

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'bayesian_experimental_design'
    logdir = os.path.join(log_path,
            current_time + '_' + socket.gethostname() + comment
        )
    writer = tensorboardX.SummaryWriter(logdir=logdir)

    if logged_plot_callbacks is None:
        logged_plot_callbacks = {}

    if bed_kwargs['use_precomputed_best_inds'] is not None:
        use_precomputed_best_inds = list(use_precomputed_best_inds)

    all_acq_angle_inds = list(acq_state_tracker.angles_tracker.acq_angle_inds)
    dip_kwargs = deepcopy(dip_kwargs)

    recons = []
    num_batches = ceil(
        acq_state_tracker.angles_tracker.total_num_acq_projs / acq_state_tracker.angles_tracker.acq_projs_batch_size)
    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='bed_optimal_angles_search'):
        
        if not bed_kwargs['use_precomputed_best_inds']:

            cov_obs_mat_eps_abs = acq_state_tracker.observation_cov.get_stabilizing_eps(
                observation_cov_mat = acq_state_tracker.cov_obs_mat_no_noise + torch.exp(
                        acq_state_tracker.observation_cov.log_noise_variance) * torch.eye(
                                acq_state_tracker.cov_obs_mat_no_noise.shape[0], device=device),
                    **bed_kwargs['bayes_exp_design_inference']['cov_obs_mat'])
            samples, images_samples = sample_observations_shifted_bayes_exp_design(  # pylint: disable=possibly-unused-variable
                acq_state_tracker.observation_cov, acq_state_tracker.ray_trafo_obj, acq_state_tracker.ray_trafo_comp_obj,
                torch.linalg.cholesky(
                    acq_state_tracker.cov_obs_mat_no_noise + ( torch.exp(
                        acq_state_tracker.observation_cov.log_noise_variance) + cov_obs_mat_eps_abs ) * torch.eye(
                            acq_state_tracker.cov_obs_mat_no_noise.shape[0], device=device) ),
                mc_samples=bed_kwargs['bayes_exp_design_inference']['mc_samples'],
                batch_size=bed_kwargs['bayes_exp_design_inference']['batch_size'],
                device=device
                )
            top_projs_idx, obj = find_optimal_proj(  # pylint: disable=possibly-unused-variable
                    samples=samples,
                    log_noise_variance=acq_state_tracker.observation_cov.log_noise_variance, 
                    acq_projs_batch_size=acq_state_tracker.angles_tracker.acq_projs_batch_size, 
                    criterion=criterion,
                    return_obj=True)
        else:
            top_projs_idx = [
                    acq_state_tracker.angles_tracker.acq_angle_inds.index(ind)  # absolute to relative in acq_angle_inds
                    for ind in bed_kwargs['use_precomputed_best_inds'][i*acq_state_tracker.angles_tracker.acq_projs_batch_size:(i+1)*acq_state_tracker.angles_tracker.acq_projs_batch_size]]

        for callback in callbacks:
            callback(
                all_acq_angle_inds, 
                acq_state_tracker.angles_tracker.init_angle_inds,
                acq_state_tracker.angles_tracker.cur_angle_inds,
                acq_state_tracker.angles_tracker.acq_angle_inds, 
                top_projs_idx,
                local_vars=locals()
                )

        for name, plot_callback in logged_plot_callbacks.items():
            fig = plot_callback(
                all_acq_angle_inds,
                acq_state_tracker.angles_tracker.init_angle_inds, 
                acq_state_tracker.angles_tracker.cur_angle_inds, 
                acq_state_tracker.angles_tracker.acq_angle_inds, 
                top_projs_idx, local_vars=locals()
                )
            # writer.add_figure(name, fig, i)  # includes a log of margin
            with io.BytesIO() as buff:
                fig.savefig(buff, format='png', bbox_inches='tight')
                buff.seek(0)
                im = plt.imread(buff)
                im = im.transpose((2, 0, 1))
            writer.add_image(name, im, i)
        
        acq_state_tracker.state_update(top_projs_idx=top_projs_idx,
            batch_size=bed_kwargs['bayes_exp_design_inference']['batch_size'],
            use_precomputed_best_inds=bed_kwargs['use_precomputed_best_inds']   ) 
            # updates ray_trafo_obj, ray_trafo_comp_obj, 

        refined_model = None 
        if ( bed_kwargs['acquisition']['reconstruct_every_k_step'] is not None
                and (i+1) % bed_kwargs['acquisition']['reconstruct_every_k_step'] == 0  ):
        
            acq_noisy_observation = observation_full.flatten()[np.concatenate(
                                acq_state_tracker.angles_tracker.cur_proj_inds_list)].view(
                                                    1, 1, *acq_state_tracker.ray_trafo_obj.obs_shape)

            acq_noisy_observation = acq_noisy_observation.to(dtype=dtype)

            recon, refined_model = eval_recon_on_new_acq_observation_data(
                ray_trafo=acq_state_tracker.ray_trafo_obj,
                angles_tracker=acq_state_tracker.angles_tracker,
                acq_noisy_observation=acq_noisy_observation,
                filtbackproj=filtbackproj,
                ground_truth=ground_truth,
                net_kwargs=dip_kwargs['net'],
                optim_kwargs=dip_kwargs['optim'],
                init_state_dict=init_state_dict,
                use_alternative_recon=bed_kwargs['use_alternative_recon'],
                alternative_recon_kwargs=bed_kwargs['alternative_recon_kwargs'],
                hyperparam_fun=hyperparam_fun,
                dtype=torch.float32,
                device=device
                )
            recons.append(recon.cpu().numpy()[0, 0])
            if refined_model is not None:
                refined_model.to(dtype=dtype)
            if bed_kwargs['acquisition']['update_network_params']:
                acq_state_tracker.model_update(refined_model=refined_model)

            if refined_model is not None:
                torch.save(
                        refined_model.state_dict(), 
                        './{}_acq_{}.pt'.format(model_basename, i+1)
                    )
            
            writer.add_image('reco', normalize(recon[0]), i)
            writer.add_image('abs(reco-gt)', normalize(np.abs(recon[0].cpu().numpy() - ground_truth[0].cpu().numpy())), i)
            print('\nPSNR with {:d} acquisitions: {}'.format(
                    len(acq_state_tracker.angles_tracker.cur_proj_inds_list), 
                    PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()),
                    '\n')
                )
    
    writer.add_image('gt', normalize(ground_truth[0].cpu().numpy()), i)
    writer.close()

    return (acq_state_tracker.angles_tracker.best_inds_acquired, 
                recons) if return_recons else acq_state_tracker.angles_tracker.best_inds_acquired
