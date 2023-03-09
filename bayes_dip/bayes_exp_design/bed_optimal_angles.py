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
import tensorly as tl
import matplotlib.pyplot as plt

from copy import deepcopy
tl.set_backend('pytorch')
from math import ceil
from tqdm import tqdm
from torch import Tensor

from .sample_observations import sample_observations_shifted_bayes_exp_design
from .update_cov_obs_mat import update_cov_obs_mat_no_noise
from .tvadam import TVAdamReconstructor
from .base_angles_tracker import BaseAnglesTracker
from .utils import get_ray_trafo_modules_exp_design

from bayes_dip.data import MatmulRayTrafo
from bayes_dip.probabilistic_models import ObservationCov
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.marginal_likelihood_optim import marginal_likelihood_hyperparams_optim, get_preconditioner
from bayes_dip.utils import normalize, PSNR

# criterion 
def sampled_diagonal_EIG(
            samples_per_detector_per_angle: Tensor, 
            noise_obs_std: float
        ):
    
    # y_samples_per_detector_per_angle -> (angles, deterctors, samples)
    log_var_per_detector_per_angle = torch.log( 
        (samples_per_detector_per_angle ** 2).mean(axis=-1) + (noise_obs_std ** 2) ) # log variance per detector pixel per angle
    diag_EIG_per_angle = log_var_per_detector_per_angle.sum(axis=1) # sum across detector pixels
    return diag_EIG_per_angle

def sampled_EIG(
            samples_per_detector_per_angle: Tensor, 
            noise_obs_std: float
        ):
    
    # y_samples_per_detector_per_angle -d> (angles, detectors, samples)
    mc_samples = samples_per_detector_per_angle.shape[-1]
    angle_cov = torch.bmm(
            samples_per_detector_per_angle, samples_per_detector_per_angle.transpose(1,2)
        ) / mc_samples # (angles, detectors, detectors)
    angle_cov += (noise_obs_std ** 2) * torch.eye(angle_cov.shape[1], device=angle_cov.device)[None, :, :]
    s, EIG = torch.linalg.slogdet(angle_cov)
    assert all(s == 1)
    return EIG

def find_optimal_proj(
        samples: Tensor, 
        log_noise_model_variance_obs: float, 
        acq_projs_batch_size: int, 
        criterion: str = 'EIG', 
        return_obj: bool = False):
    
    # mc_samples x 1 x num_acq x num_projs_per_angle
    if criterion == 'diagonal_EIG':
        obj = sampled_diagonal_EIG(
                samples.squeeze(1).moveaxis(0,-1), 
                torch.exp(log_noise_model_variance_obs)**.5
            ).cpu().numpy()
    elif criterion == 'EIG':
        obj = sampled_EIG(
                samples.squeeze(1).moveaxis(0,-1), 
                torch.exp(log_noise_model_variance_obs)**.5
            ).cpu().numpy()
    elif criterion == 'var':
        obj = torch.mean(
                samples.pow(2), dim=(0, -1)
            ).squeeze(0).cpu().numpy()
    else:
        raise ValueError
    top_projs_idx = np.argpartition(obj, -acq_projs_batch_size)[-acq_projs_batch_size:]

    return top_projs_idx, obj if return_obj else top_projs_idx

# note: observation needs to have shape (len(init_angle_inds), num_projs_per_angle)
def bed_optimal_angles_search(
        observation_cov: ObservationCov,
        ray_trafo_full: MatmulRayTrafo,
        angles_tracker: BaseAnglesTracker, # proj_inds_per_angle, init_angle_inds, acq_angle_inds, total_num_acq_projs, acq_projs_batch_size
        observation_full: Tensor,
        filtbackproj: Tensor,
        ground_truth: Tensor,
        init_cov_obs_mat_no_noise: Tensor,
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

    cov_obs_mat_no_noise = init_cov_obs_mat_no_noise

    all_acq_angle_inds = list(angles_tracker.acq_angle_inds)
    ray_trafo_obj, ray_trafo_comp_obj = get_ray_trafo_modules_exp_design(
        ray_trafo_full=ray_trafo_full,
        angles_tracker=angles_tracker, 
        dtype=dtype,
        device=device
    )

    dip_kwargs = deepcopy(dip_kwargs)
    if init_state_dict is None:
        random_init_reconstructor = DeepImagePriorReconstructor(
                ray_trafo_obj, torch_manual_seed=bed_kwargs['torch_manual_seed'],
                device=device, dip_kwargs=dip_kwargs,
            )
        init_state_dict = random_init_reconstructor.nn_model.state_dict()
    observation_cov_state_dict = observation_cov.state_dict()

    recons = []
    num_batches = ceil(angles_tracker.total_num_acq_projs / angles_tracker.acq_projs_batch_size)
    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='bed_optimal_angles_search'):
        if not bed_kwargs['use_precomputed_best_inds']:
            cov_obs_mat_eps_abs = observation_cov.get_stabilizing_eps(
                observation_cov_mat = cov_obs_mat_no_noise + torch.exp(observation_cov.log_noise_variance) * torch.eye(
                            cov_obs_mat_no_noise.shape[0], device=device),
                    **bed_kwargs['bayes_exp_design_inference']['cov_obs_mat']
                )
            samples, images_samples = sample_observations_shifted_bayes_exp_design(
                observation_cov, ray_trafo_obj, ray_trafo_comp_obj,
                torch.linalg.cholesky(
                    cov_obs_mat_no_noise + ( torch.exp(observation_cov.log_noise_variance) + cov_obs_mat_eps_abs ) * torch.eye(
                        cov_obs_mat_no_noise.shape[0], device=device) ),
                mc_samples=bed_kwargs['bayes_exp_design_inference']['mc_samples'],
                batch_size=bed_kwargs['bayes_exp_design_inference']['batch_size'],
                device=device
                )
            top_projs_idx, obj = find_optimal_proj(
                    samples, 
                    observation_cov.log_noise_variance, 
                    angles_tracker.acq_projs_batch_size, 
                    criterion=criterion, 
                    return_obj=True
                )
        else:
            top_projs_idx = [
                    angles_tracker.acq_angle_inds.index(ind)  # absolute to relative in acq_angle_inds
                    for ind in bed_kwargs['use_precomputed_best_inds'][i*angles_tracker.acq_projs_batch_size:(i+1)*angles_tracker.acq_projs_batch_size]]

        for callback in callbacks:
            callback(
                all_acq_angle_inds, 
                angles_tracker.init_angle_inds, 
                angles_tracker.cur_angle_inds, 
                angles_tracker.acq_angle_inds, 
                top_projs_idx,
                local_vars=locals()
                )

        for name, plot_callback in logged_plot_callbacks.items():
            fig = plot_callback(
                all_acq_angle_inds,
                angles_tracker.init_angle_inds, 
                angles_tracker.cur_angle_inds, 
                angles_tracker.acq_angle_inds, 
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
        top_k_acq_proj_inds_list = angles_tracker.update(top_projs_idx_to_be_added=top_projs_idx)
        ray_trafo_top_k_obj = MatmulRayTrafo(
                im_shape=ray_trafo_full.im_shape, 
                obs_shape=(angles_tracker.acq_projs_batch_size, angles_tracker.num_projs_per_angle), 
                matrix=
                scipy.sparse.csr_matrix(
                            ray_trafo_full.matrix[np.concatenate(top_k_acq_proj_inds_list)].cpu().numpy()
                        )
            ).to(dtype=dtype, device=device)

        if not bed_kwargs['use_precomputed_best_inds']:
            cov_obs_mat_no_noise = update_cov_obs_mat_no_noise(
                observation_cov=observation_cov, 
                ray_trafo_obj=ray_trafo_obj, 
                ray_trafo_top_k_obj=ray_trafo_top_k_obj,
                cov_obs_mat_no_noise=cov_obs_mat_no_noise, 
                batch_size=bed_kwargs['bayes_exp_design_inference']['batch_size']
                )

        # update transforms
        ray_trafo_obj, ray_trafo_comp_obj = get_ray_trafo_modules_exp_design(
            ray_trafo_full=ray_trafo_full,
            angles_tracker=angles_tracker, 
            dtype=dtype,
            device=device
            )
        observation_cov.trafo = ray_trafo_obj
        observation_cov.image_cov.neural_basis_expansion.trafo = ray_trafo_obj

        if bed_kwargs['acquisition']['reconstruct_every_k_step'] is not None and (i+1) % bed_kwargs['acquisition']['reconstruct_every_k_step'] == 0:
            if not bed_kwargs['use_alternative_recon']:
                if hyperparam_fun is not None:
                    dip_kwargs['optim']['gamma'], dip_kwargs['optim']['iterations'] = hyperparam_fun(
                            len(angles_tracker.cur_proj_inds_list)
                        )
                refine_reconstructor = DeepImagePriorReconstructor(
                    deepcopy(ray_trafo_obj),
                    net_kwargs=dip_kwargs['net'],
                    device=device
                    )
                refine_reconstructor.nn_model.to(dtype=torch.float32)
                obs = observation_full.flatten()[np.concatenate(
                    angles_tracker.cur_proj_inds_list)].view(1, 1, *ray_trafo_obj.obs_shape)
                refine_reconstructor.nn_model.load_state_dict(init_state_dict)
                recon = refine_reconstructor.reconstruct(
                    noisy_observation=obs.to(dtype=torch.float32),
                    filtbackproj=filtbackproj.to(dtype=torch.float32),
                    ground_truth=ground_truth.to(dtype=torch.float32),
                    optim_kwargs=dip_kwargs['optim']
                    )
                torch.save(refine_reconstructor.nn_model.state_dict(),
                            './{}_acq_{}.pt'.format(model_basename, i+1)
                                )
                recons.append(recon)
                if bed_kwargs['acquisition']['update_network_params'] and not bed_kwargs['use_precomputed_best_inds']:
                    observation_cov.load_state_dict(observation_cov_state_dict)
                    observation_cov.image_cov.neural_basis_expansion.nn_model = refine_reconstructor.nn_model
                    if bed_kwargs['acquisition']['update_prior_hyperparams'] == 'mrglik':
                        # update preconditioner 
                        updated_cg_preconditioner = get_preconditioner(
                                observation_cov=observation_cov,
                                kwargs=bed_kwargs['update_preconditioner_kwargs']
                            )
                        bed_kwargs['marginal_lik_kwargs']['linear_cg']['preconditioner'] = updated_cg_preconditioner
        
                        marginal_likelihood_hyperparams_optim(
                            observation_cov=observation_cov,
                            observation=obs,
                            recon=recon,
                            linearized_weights=None, # TODO: 
                            optim_kwargs=bed_kwargs['marginal_lik_kwargs'],
                            log_path=os.path.join(bed_kwargs['log_path'], f'mrglik_optim_{i}'),
                        )
                        observation_cov_state_dict = observation_cov.state_dict()
                    else: 
                        pass
                    if bed_kwargs['use_gprior']:
                        observation_cov.load_state_dict(observation_cov_state_dict)
                        # TODO: update scale vector
                        observation_cov.image_cov.neural_basis_expansion.update_scale()
                    # compute cov_obs_mat via closure 
                    cov_obs_mat_no_noise = observation_cov.assemble_observation_cov(
                                                use_noise_variance=False)
                                                    
            elif bed_kwargs['use_alternative_recon'] == 'tvadam':
                obs = observation_full.flatten()[np.concatenate(angles_tracker.cur_proj_inds_list)].view(1, 1, *ray_trafo_obj.im_shape)
                if bed_kwargs['alternative_recon_kwargs']['tvadam_hyperparam_fun'] is not None:
                    bed_kwargs['alternative_recon_kwargs']['tvadam_kwargs']['gamma'], bed_kwargs['alternative_recon_kwargs']['tvadam_kwargs']['iterations'] = tvadam_hyperparam_fun(
                            len(angles_tracker.cur_proj_inds_list)
                        )
                tvadam_reconstructor = TVAdamReconstructor(
                        deepcopy(ray_trafo_obj).to(dtype=torch.float32), 
                        cfg=bed_kwargs['alternative_recon_kwargs']['tvadam_kwargs']
                    )
                recon = tvadam_reconstructor.reconstruct(
                        obs.to(dtype=torch.float32), 
                        filtbackproj=filtbackproj.to(dtype=torch.float32), 
                        ground_truth=ground_truth.to(dtype=torch.float32),
                        log=True)
                recons.append(recon)
            else:
                raise ValueError
            
            writer.add_image('reco', normalize(recon[None]), i)
            writer.add_image('abs(reco-gt)', normalize(np.abs(recon[None] - ground_truth[0, 0].cpu().numpy())), i)
            print('\nPSNR with {:d} acquisitions: {}'.format(len(angles_tracker.cur_proj_inds_list), PSNR(recon, ground_truth[0, 0].cpu().numpy()), '\n'))
    
    writer.add_image('gt', normalize(ground_truth[0].cpu().numpy()), i)
    writer.close()

    best_inds_acquired = angles_tracker.get_best_inds_acquired()

    del cov_obs_mat_no_noise
    gc.collect(); torch.cuda.empty_cache()

    return best_inds_acquired, recons if return_recons else best_inds_acquired

# note: observation needs to have shape (len(init_angle_inds), num_projs_per_angle)
def bed_eqdist_angles_baseline(
    ray_trafo_full: MatmulRayTrafo, 
    angles_tracker: BaseAnglesTracker, 
    observation_full: Tensor, 
    filtbackproj: Tensor,
    ground_truth: Tensor,
    bed_kwargs: Dict,
    dip_kwargs: Dict,
    init_state_dict: Optional[Any] = None,
    hyperparam_fun: Optional[Any]  = None, 
    model_basename: str = '.',
    callbacks=(),
    logged_plot_callbacks: Optional[Any]  = None,
    log_path: str = './',
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
    ):

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'bayesian_experimental_design_baseline'
    logdir = os.path.join(
    log_path,
    current_time + '_' + socket.gethostname() + comment)
    writer = tensorboardX.SummaryWriter(logdir=logdir)

    if logged_plot_callbacks is None:
        logged_plot_callbacks = {}

    dip_kwargs = deepcopy(dip_kwargs)
    if init_state_dict is None:
        random_init_reconstructor = DeepImagePriorReconstructor(
                ray_trafo_obj, torch_manual_seed=bed_kwargs['torch_manual_seed'],
                device=device, net_kwargs=dip_kwargs['net'],
            )
        init_state_dict = random_init_reconstructor.nn_model.state_dict()

    ray_trafo_obj, _ = get_ray_trafo_modules_exp_design(
        ray_trafo_full=ray_trafo_full,
        angles_tracker=angles_tracker,
        dtype=dtype,
        device=device
    )

    recons = []
    all_acq_angle_inds = list(angles_tracker.acq_angle_inds)
    num_batches = ceil(angles_tracker.total_num_acq_projs / angles_tracker.acq_projs_batch_size)
    for i in tqdm(range(num_batches), miniters=num_batches//100, desc='bed_eqdist_angles_baseline'):
        num_add_acq = (i + 1) *angles_tracker.acq_projs_batch_size
        num_cur = len(angles_tracker.init_angle_inds) + num_add_acq

        if (angles_tracker.full_num_angles % num_cur == 0 and num_cur % len(angles_tracker.init_angle_inds) == 0):
            baseline_step = angles_tracker.full_num_angles // num_cur
            new_cur_angle_inds = np.arange(
                    0, 
                    angles_tracker.full_num_angles, 
                    baseline_step
                )
            cur_proj_inds_list = [angles_tracker.proj_inds_per_angle[a_ind] for a_ind in new_cur_angle_inds]
            add_acq_angle_inds = np.setdiff1d(
                    new_cur_angle_inds, 
                    angles_tracker.init_angle_inds
                ).tolist()
            assert len(add_acq_angle_inds) == len(new_cur_angle_inds) - len(angles_tracker.init_angle_inds)
            for callback in callbacks:
                callback(
                    all_acq_angle_inds, 
                    angles_tracker.init_angle_inds, 
                    angles_tracker.init_angle_inds,
                    add_acq_angle_inds, 
                    np.arange(len(add_acq_angle_inds)), 
                    local_vars=locals()
                )
            for name, plot_callback in logged_plot_callbacks.items():
                fig = plot_callback(
                    all_acq_angle_inds, 
                    angles_tracker.init_angle_inds, 
                    angles_tracker.init_angle_inds,
                    add_acq_angle_inds, 
                    np.arange(len(add_acq_angle_inds)), 
                    local_vars=locals()
                )
                # writer.add_figure(name, fig, i)  # includes a log of margin
                with io.BytesIO() as buff:
                    fig.savefig(buff, format='png', bbox_inches='tight')
                    buff.seek(0)
                    im = plt.imread(buff)
                    im = im.transpose((2, 0, 1))
                writer.add_image(name, im, i)
        
            angles_tracker.cur_proj_inds_list = cur_proj_inds_list # manual update cur_proj_inds_list
            # update transform
            ray_trafo_obj, _ = get_ray_trafo_modules_exp_design(
                ray_trafo_full=ray_trafo_full,
                angles_tracker=angles_tracker, 
                dtype=dtype,
                device=device
            )
        else:
            new_cur_angle_inds = None

        if bed_kwargs['acquisition']['reconstruct_every_k_step'] is not None and (i+1) % bed_kwargs['acquisition']['reconstruct_every_k_step'] == 0:
            if new_cur_angle_inds is not None:
                if not bed_kwargs['use_alternative_recon']:
                    if hyperparam_fun is not None:
                        dip_kwargs['optim']['gamma'], dip_kwargs['optim']['iterations'] = hyperparam_fun(
                            len(cur_proj_inds_list)
                        )
                    refine_reconstructor = DeepImagePriorReconstructor(
                            deepcopy(ray_trafo_obj).to(
                                        device=device, 
                                        dtype=torch.float32),
                            net_kwargs=dip_kwargs['net'],
                            device=device
                        )
                    refine_reconstructor.nn_model.load_state_dict(init_state_dict)
                    refine_reconstructor.nn_model.to(dtype=torch.float32)
                    obs = observation_full.flatten()[np.concatenate(
                                    cur_proj_inds_list)].view(
                                        1, 1, *ray_trafo_obj.obs_shape)

                    recon = refine_reconstructor.reconstruct(
                            noisy_observation=obs.to(dtype=torch.float32), 
                            filtbackproj=filtbackproj.to(dtype=torch.float32), 
                            ground_truth=ground_truth.to(dtype=torch.float32),
                            optim_kwargs=dip_kwargs['optim']
                        )
                    torch.save(refine_reconstructor.nn_model.state_dict(),
                                './{}_acq_{}.pt'.format(model_basename, i+1)
                            )
                    
                    recons.append(recon.cpu().numpy())
        
                elif bed_kwargs['use_alternative_recon'] == 'tvadam':

                    obs = observation_full.flatten()[np.concatenate(
                        cur_proj_inds_list)].view(1, 1, *ray_trafo_obj.obs_shape)
                    if bed_kwargs['alternative_recon_kwargs']['tvadam_hyperparam_fun'] is not None:
                        tvadam_hyperparam_fun = bed_kwargs['alternative_recon_kwargs']['tvadam_hyperparam_fun']
                        bed_kwargs['alternative_recon_kwargs']['tvadam_kwargs']['gamma'], bed_kwargs['alternative_recon_kwargs']['tvadam_kwargs']['iterations'] = tvadam_hyperparam_fun(
                                len(cur_proj_inds_list)
                            )
                    tvadam_reconstructor = TVAdamReconstructor(
                            deepcopy(ray_trafo_obj).to(dtype=torch.float32), 
                            )
                    recon = tvadam_reconstructor.reconstruct(
                                obs.to(dtype=torch.float32), 
                                filtbackproj=filtbackproj.to(dtype=torch.float32), 
                                ground_truth=ground_truth.to(dtype=torch.float32),
                                optim_kwargs=bed_kwargs['alternative_recon_kwargs']['tvadam_kwargs'],
                                log=True
                            )
                    recons.append(recon.cpu().numpy())
                else:
                    raise ValueError

                writer.add_image('reco', normalize(recon[0].cpu().numpy()), i)
                writer.add_image('abs_reco_gt', normalize(np.abs(recon[0].cpu().numpy() - ground_truth[0].cpu().numpy())), i)
                print('\nPSNR with {:d} acquisitions: {}'.format(len(cur_proj_inds_list), 
                                    PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
                                )
                            )
            else:
                print('\ncould not find baseline angle inds with {:d} acquisitions (init={:d}, full={:d})'.format(
                        num_cur, len(angles_tracker.init_angle_inds), 
                        angles_tracker.full_num_angles)
                    )
                recons.append(None)

    writer.add_image('gt', normalize(ground_truth[0].cpu().numpy()), 0)
    writer.close()

    return recons
