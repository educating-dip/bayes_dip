import torch
import numpy as np
from torch import Tensor
from bayes_dip.data import MatmulRayTrafo
from bayes_dip.probabilistic_models import ObservationCov

def update_cov_obs_mat_no_noise(
        observation_cov: ObservationCov, 
        ray_trafo_obj: MatmulRayTrafo, 
        ray_trafo_top_k_obj: MatmulRayTrafo,
        cov_obs_mat_no_noise: Tensor,
        batch_size: int
    ):
    
    row_numel = ray_trafo_top_k_obj.matrix.shape[0]
    top_k_cov_obs_diag = []
    top_k_cov_obs_off_diag = []
    for i in range(0, row_numel, batch_size):
        if ray_trafo_top_k_obj.matrix.is_sparse:
            v = torch.stack(
                    [ray_trafo_top_k_obj.matrix[j] for j in range(i, min(i+batch_size, row_numel))]
                ).to_dense()
        else:
            v = ray_trafo_top_k_obj.matrix[i:i+batch_size, :]
        eff_batch_size = v.shape[0]
        if eff_batch_size < batch_size:
            v = torch.nn.functional.pad(v, (0, 0, 0, batch_size-eff_batch_size))
        v = observation_cov.image_cov(
                v.view(-1, 1, *ray_trafo_top_k_obj.im_shape)
            )
        v = v.view(v.shape[0], -1)
        if eff_batch_size < batch_size:
            v = v[:eff_batch_size]

        top_k_cov_obs_diag.append(
                ray_trafo_top_k_obj.trafo_flat(v.T).T
            )
        top_k_cov_obs_off_diag.append(
                ray_trafo_obj.trafo_flat(v.T).T
            )
    top_k_cov_obs_diag = torch.cat(top_k_cov_obs_diag, dim=0)
    top_k_cov_obs_diag = 0.5*(top_k_cov_obs_diag + top_k_cov_obs_diag.T)  # numerical stability
    top_k_cov_obs_off_diag = torch.cat(top_k_cov_obs_off_diag, dim=0)
    updated_top_cov_obs_mat = torch.cat(
            [cov_obs_mat_no_noise, top_k_cov_obs_off_diag.T], 
            dim=1
        )
    updated_bottom_cov_obs_mat = torch.cat(
            [top_k_cov_obs_off_diag, top_k_cov_obs_diag], 
            dim=1
        )
    updated_cov_obs_mat = torch.cat(
            [updated_top_cov_obs_mat, updated_bottom_cov_obs_mat], 
            dim=0
        )
    
    return updated_cov_obs_mat
