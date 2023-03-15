from typing import Dict, Optional, Any 
import os
import gc
import io
import torch
import socket
import datetime
import numpy as np
import scipy.sparse
import tensorboardX
import matplotlib.pyplot as plt

from copy import deepcopy
from math import ceil
from tqdm import tqdm
from torch import Tensor

from .sample_observations import sample_observations_shifted_bayes_exp_design
from .update_cov_obs_mat import update_cov_obs_mat_no_noise
from .tvadam import TVAdamReconstructor
from .base_angles_tracker import BaseAnglesTracker
from .acq_criterions import find_optimal_proj
from .acq_state_tracker import AcqStateTracker
from .utils import eval_dip_on_newly_acquired_observation_data

from bayes_dip.data import MatmulRayTrafo
from bayes_dip.probabilistic_models import ObservationCov
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.marginal_likelihood_optim import marginal_likelihood_hyperparams_optim, get_preconditioner
from bayes_dip.utils import normalize, PSNR

# note: observation needs to have shape (len(init_angle_inds), num_projs_per_angle)
def bed_optimal_angles_search(
        acq_state_tracker: AcqStateTracker, 
        observation_full: Tensor,
        filtbackproj: Tensor,
        ground_truth: Tensor,
        bed_kwargs: Dict,
        dip_kwargs: Dict,
        criterion: str = 'EIG',
        init_state_dict: Optional[Any] = None,
        model_basename: str = 'refined_dip_model',
        log_path: str = './',
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
        hyperparam_fun=None, # hyperparam_fun(num_acq) -> (gamma, iterations)
        callbacks=(),
        logged_plot_callbacks: Optional[Any]  = None,
        return_recons: bool = False,
    ):

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

    cov_obs_mat_no_noise = acq_state_tracker.observation_cov.assemble_observation_cov(
                    use_noise_variance=False
                    )
    all_acq_angle_inds = list(acq_state_tracker.angles_tracker.acq_angle_inds)
    dip_kwargs = deepcopy(dip_kwargs)

    if init_state_dict is None:
        random_init_reconstructor = DeepImagePriorReconstructor(
                acq_state_tracker.ray_trafo_obj, torch_manual_seed=bed_kwargs['torch_manual_seed'],
                device=device, dip_kwargs=dip_kwargs,
            )
        init_state_dict = random_init_reconstructor.nn_model.state_dict()

    recons = []
    num_batches = ceil(
        acq_state_tracker.angles_tracker.total_num_acq_projs / acq_state_tracker.angles_tracker.acq_projs_batch_size)
    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='bed_optimal_angles_search'):
        
        if not bed_kwargs['use_precomputed_best_inds']:

            cov_obs_mat_eps_abs = acq_state_tracker.observation_cov.get_stabilizing_eps(
                observation_cov_mat = cov_obs_mat_no_noise + torch.exp(
                        acq_state_tracker.observation_cov.log_noise_variance) * torch.eye(
                            cov_obs_mat_no_noise.shape[0], device=device),
                    **bed_kwargs['bayes_exp_design_inference']['cov_obs_mat'])

            samples, images_samples = sample_observations_shifted_bayes_exp_design(
                acq_state_tracker.observation_cov, acq_state_tracker.ray_trafo_obj, acq_state_tracker.ray_trafo_comp_obj,
                torch.linalg.cholesky(
                    cov_obs_mat_no_noise + ( torch.exp(
                        acq_state_tracker.observation_cov.log_noise_variance) + cov_obs_mat_eps_abs ) * torch.eye(
                            cov_obs_mat_no_noise.shape[0], device=device) ),
                mc_samples=bed_kwargs['bayes_exp_design_inference']['mc_samples'],
                batch_size=bed_kwargs['bayes_exp_design_inference']['batch_size'],
                device=device
                )

            top_projs_idx, obj = find_optimal_proj(
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

        # update lists of acquired and not yet acquired projections
        top_k_acq_proj_inds_list = acq_state_tracker.angles_tracker.update(
            top_projs_idx=top_projs_idx
            )
        ray_trafo_top_k_obj = MatmulRayTrafo(
                im_shape=acq_state_tracker.ray_trafo_full.im_shape, 
                obs_shape=(acq_state_tracker.angles_tracker.acq_projs_batch_size, 
                                acq_state_tracker.angles_tracker.num_projs_per_angle), 
                matrix=scipy.sparse.csr_matrix(
                            acq_state_tracker.ray_trafo_full.matrix[np.concatenate(
                                    top_k_acq_proj_inds_list)].cpu().numpy()
                        )
            ).to(dtype=dtype, device=device)

        if not bed_kwargs['use_precomputed_best_inds']:
            cov_obs_mat_no_noise = update_cov_obs_mat_no_noise(
                observation_cov=acq_state_tracker.observation_cov, 
                ray_trafo_obj=acq_state_tracker.ray_trafo_obj, 
                ray_trafo_top_k_obj=ray_trafo_top_k_obj,
                cov_obs_mat_no_noise=cov_obs_mat_no_noise,
                batch_size=bed_kwargs['bayes_exp_design_inference']['batch_size']
                )

        acq_state_tracker.state_update()

        refined_model = None 
        if ( bed_kwargs['acquisition']['reconstruct_every_k_step'] is not None
                and (i+1) % bed_kwargs['acquisition']['reconstruct_every_k_step'] == 0):

            newly_acq_noisy_observation = observation_full.flatten()[np.concatenate(
                    acq_state_tracker.angles_tracker.cur_proj_inds_list)].view(
                        1, 1, *acq_state_tracker.ray_trafo_obj.obs_shape)
            newly_acq_noisy_observation.to(dtype=torch.float32)

            if not bed_kwargs['use_alternative_recon']:
                if hyperparam_fun is not None:
                    dip_kwargs['optim']['gamma'], dip_kwargs['optim']['iterations'] = hyperparam_fun(
                            len(acq_state_tracker.angles_tracker.cur_proj_inds_list)
                        )
                recon, refined_model = eval_dip_on_newly_acquired_observation_data(
                    acq_state_tracker=acq_state_tracker,
                    noisy_observation=newly_acq_noisy_observation,
                    filtbackproj=filtbackproj.to(dtype=dtype),
                    ground_truth=ground_truth.to(dtype=dtype),
                    net_kwargs=dip_kwargs['net'],
                    optim_kwargs=dip_kwargs['optim'],
                    init_state_dict=init_state_dict,
                    dtype=torch.float32, 
                    device=device
                )
                torch.save(
                    refined_model.state_dict(), 
                    './{}_acq_{}.pt'.format(model_basename, i+1)
                    )
                recons.append(recon)

                if bed_kwargs['acquisition']['update_network_params'] and not bed_kwargs['use_precomputed_best_inds']:
                    
                    marglik_update_kwargs = (bed_kwargs['marginal_lik_kwargs'], bed_kwargs['update_preconditioner_kwargs'])
                    acq_state_tracker.state_update(
                        update_neural_basis_from_refined_model=refined_model,
                        scale_update_kwargs=bed_kwargs['scale_update_kwargs'],
                        marglik_update_kwargs=marglik_update_kwargs, 
                        noisy_observation=newly_acq_noisy_observation, 
                        recon=recon,
                        )
                    cov_obs_mat_no_noise = acq_state_tracker.observation_cov.assemble_observation_cov(
                            use_noise_variance=False)
                            
            elif bed_kwargs['use_alternative_recon'] == 'tvadam':

                if bed_kwargs['alternative_recon_kwargs']['tvadam_hyperparam_fun'] is not None:
                    bed_kwargs['alternative_recon_kwargs']['tvadam_kwargs']['gamma'], bed_kwargs['alternative_recon_kwargs']['tvadam_kwargs']['iterations'] = tvadam_hyperparam_fun(
                            len(angles_tracker.cur_proj_inds_list)
                        )
                tvadam_reconstructor = TVAdamReconstructor(
                        deepcopy(ray_trafo_obj).to(dtype=torch.float32), 
                    )
                recon = tvadam_reconstructor.reconstruct(
                        newly_acq_noisy_observation,
                        filtbackproj=filtbackproj.to(dtype=torch.float32), 
                        ground_truth=ground_truth.to(dtype=torch.float32),
                        optim_kwargs=bed_kwargs['alternative_recon_kwargs']['tvadam_kwargs'],
                        log=True)
                recons.append(recon)
            else:
                raise ValueError
            
            writer.add_image('reco', normalize(recon[0]), i)
            writer.add_image('abs(reco-gt)', normalize(np.abs(recon[0].cpu().numpy() - ground_truth[0].cpu().numpy())), i)
            print('\nPSNR with {:d} acquisitions: {}'.format(
                len(acq_state_tracker.angles_tracker.cur_proj_inds_list), 
                PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()),
                '\n')
                )
    
    writer.add_image('gt', normalize(ground_truth[0].cpu().numpy()), i)
    writer.close()

    best_inds_acquired = angles_tracker.get_best_inds_acquired()

    del cov_obs_mat_no_noise
    gc.collect(); torch.cuda.empty_cache()

    return best_inds_acquired, recons if return_recons else best_inds_acquired
