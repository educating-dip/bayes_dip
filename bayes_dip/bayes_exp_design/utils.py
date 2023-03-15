from typing import Optional, Any, Dict
import yaml
import scipy
import numpy as np 
import matplotlib.pyplot as plt

from torch import Tensor
from .base_angles_tracker import BaseAnglesTracker
from .acq_state_tracker import AcqStateTracker

from bayes_dip.data import MatmulRayTrafo
from bayes_dip.dip import DeepImagePriorReconstructor

def eval_dip_on_newly_acquired_observation_data(
    acq_state_tracker: AcqStateTracker,
    noisy_observation: Tensor,
    filtbackproj: Tensor, 
    ground_truth: Tensor,
    net_kwargs: Dict,
    optim_kwargs: Dict,
    init_state_dict: Optional[Any] = None,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None
    ): 

    refine_reconstructor = DeepImagePriorReconstructor(
        acq_state_tracker.ray_trafo_obj,
        net_kwargs=net_kwargs,
        device=device
        )

    if init_state_dict is not None: 
        refine_reconstructor.nn_model.load_state_dict(init_state_dict)
    refine_reconstructor.nn_model.to(
        device=device, dtype=dtype
        )

    recon = refine_reconstructor.reconstruct(
        noisy_observation=noisy_observation,
        filtbackproj=filtbackproj,
        ground_truth=ground_truth,
        optim_kwargs=optim_kwargs
        )

    return recon, refine_reconstructor.nn_model

def get_hyperparam_fun_from_yaml(path, data, noise_stddev):
    
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    d_per_angle = {n_a: d[data][n_a][noise_stddev] for n_a in d[data].keys() if noise_stddev in d[data][n_a]}
    def hyperparam_fun(num_angles):
        num_angles_provided = list(d_per_angle.keys())
        nearest_num_angles_i = min(enumerate(abs(n_a - num_angles) for n_a in num_angles_provided), key=lambda x: x[1])[0]
        nearest_num_angles = num_angles_provided[nearest_num_angles_i]
        if nearest_num_angles != num_angles:
            print('did not find hyperparameters for {:d} angles, using hyperparameters from {:d} angles instead'.format(num_angles, nearest_num_angles))
        h = d_per_angle[nearest_num_angles]
        return float(h['gamma']), int(h['iterations'])
    return hyperparam_fun

def get_save_obj_callback(obj_list):

    def save_obj_callback(all_acq_angle_inds, init_angle_inds, cur_angle_inds, acq_angle_inds, top_projs_idx, local_vars):
        if 'obj' in local_vars:
            images_samples = local_vars['images_samples']
            obj = local_vars['obj']
            obj_list.append({'obj': obj,
                'acq_angle_inds': acq_angle_inds,
                'var_images_samples': images_samples.pow(2).mean(dim=0).squeeze().detach().cpu().numpy()
                }
            )
        else:
            obj_list.append(None)
    
    return save_obj_callback

def plot_obj_callback(all_acq_angle_inds, init_angle_inds, cur_angle_inds, acq_angle_inds, top_projs_idx, local_vars):
    fig, ax = plt.subplots()
    top_k_acq_angle_inds = [acq_angle_inds[idx] for idx in top_projs_idx]
    for a in init_angle_inds:
        ax.axvline(a, color='gray')
    ax.plot(acq_angle_inds, local_vars['obj'], 'x', color='tab:blue')
    ax.plot(top_k_acq_angle_inds, local_vars['obj'][top_projs_idx], 'o', color='tab:red')
    ax.set_xlabel('angle')
    ax.set_ylabel('mean variance')
    return fig

def plot_angles_callback(all_acq_angle_inds, init_angle_inds, cur_angle_inds, acq_angle_inds, top_projs_idx, local_vars):
    
    full_angles = local_vars['acq_state_tracker'].ray_trafo_full.angles
    top_k_acq_angle_inds = [acq_angle_inds[idx] for idx in top_projs_idx]
    if (len(full_angles) % (len(cur_angle_inds) + len(top_k_acq_angle_inds)) == 0
            and (len(cur_angle_inds) + len(top_k_acq_angle_inds)) % len(init_angle_inds) == 0):
        baseline_step = len(full_angles) // (len(cur_angle_inds) + len(top_k_acq_angle_inds))
        baseline_angle_inds = np.arange(0, len(full_angles), baseline_step)
    else:
        baseline_angle_inds = None
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # for theta in full_angles[init_angle_inds]:
    #     ax.plot([theta, theta], [0.1, 1.], color='gray')
    for theta in full_angles[acq_angle_inds]:
        ax.plot([theta, theta], [0.1, 1.], color='gray', alpha=0.025)
    for theta in full_angles[cur_angle_inds]:
        ax.plot([theta, theta], [0.1, 1.], color='tab:red', alpha=0.425)
    for theta in full_angles[top_k_acq_angle_inds]:
        ax.plot([theta, theta], [0.1, 1.], color='tab:red')
    if baseline_angle_inds is not None:
        for theta in full_angles[baseline_angle_inds]:
            ax.plot([theta, theta], [0.1, 1.], color='gray', linestyle='dotted')
    ax.set_yticks([])
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_thetagrids(full_angles[init_angle_inds]/np.pi*180.)
    ax.grid(linewidth=1.5, color='black')

    return fig
