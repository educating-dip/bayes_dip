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
from math import ceil
from tqdm import tqdm
from torch import Tensor

from .sample_observations import sample_observations_shifted_bayes_exp_design
from .update_cov_obs_mat import update_cov_obs_mat_no_noise
from .tvadam import TVAdamReconstructor
from .base_angles_tracker import BaseAnglesTracker
from .acq_state_tracker import _get_ray_trafo_modules

from bayes_dip.data import MatmulRayTrafo
from bayes_dip.probabilistic_models import ObservationCov
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.marginal_likelihood_optim import marginal_likelihood_hyperparams_optim, get_preconditioner
from bayes_dip.utils import normalize, PSNR

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

    ray_trafo_obj, _ = _get_ray_trafo_modules(
        ray_trafo_full=ray_trafo_full,
        angles_tracker=angles_tracker,
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
            ray_trafo_obj, _ = _get_ray_trafo_modules(
                ray_trafo_full=ray_trafo_full,
                angles_tracker=angles_tracker, 
                device=device
            )
        else:
            new_cur_angle_inds = None

        if bed_kwargs['reconstruct_every_k_step'] is not None and (i+1) % bed_kwargs['reconstruct_every_k_step'] == 0:
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
