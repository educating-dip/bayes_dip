from typing import Dict, Optional, Any 
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

from .base_angles_tracker import BaseAnglesTracker
from .acq_state_tracker import _get_ray_trafo_modules
from .utils import eval_recon_on_new_acq_observation_data

from bayes_dip.data import MatmulRayTrafo
from bayes_dip.dip import DeepImagePriorReconstructor
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
        num_add_acq = (i + 1) * angles_tracker.acq_projs_batch_size
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

                acq_noisy_observation = observation_full.flatten()[np.concatenate(
                                    angles_tracker.cur_proj_inds_list)].view(
                                                        1, 1, *ray_trafo_obj.obs_shape)

                recon, refined_model = eval_recon_on_new_acq_observation_data(
                    ray_trafo=ray_trafo_obj,
                    angles_tracker=angles_tracker,
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
                    torch.save(
                            refined_model.state_dict(), 
                            './{}_acq_{}.pt'.format(model_basename, i+1)
                        )

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
