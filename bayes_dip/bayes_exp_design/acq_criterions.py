import torch
import numpy as np

from torch import Tensor

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
        log_noise_variance: float, 
        acq_projs_batch_size: int, 
        criterion: str = 'EIG', 
        return_obj: bool = False
        ):

    # mc_samples x 1 x num_acq x num_projs_per_angle
    stddev = torch.exp(log_noise_variance)**.5
    if criterion == 'diagonal_EIG':
        obj = sampled_diagonal_EIG(
                samples.squeeze(1).moveaxis(0,-1),
                stddev
            ).cpu().numpy()
    elif criterion == 'EIG':
        obj = sampled_EIG(
                samples.squeeze(1).moveaxis(0,-1),
                stddev
            ).cpu().numpy()
    elif criterion == 'var':
        obj = torch.mean(
                samples.pow(2),
                dim=(0, -1)
            ).squeeze(0).cpu().numpy()
    else:
        raise ValueError
    
    top_projs_idx = np.argpartition(
            obj, 
            -acq_projs_batch_size
        )[-acq_projs_batch_size:]

    return top_projs_idx, obj if return_obj else top_projs_idx
