from typing import List
import numpy as np
from bayes_dip.data import MatmulRayTrafo

class BaseAnglesTracker:

    def __init__(self, 
            ray_trafo: MatmulRayTrafo, 
            angular_sub_sampling: int = 1,
            total_num_acq_projs: int = 100,
            acq_projs_batch_size: int = 5
        ):
        
        self.full_num_angles = len(ray_trafo.angles)
        self.total_num_acq_projs = total_num_acq_projs
        self.acq_projs_batch_size = acq_projs_batch_size

        self.init_angle_inds = np.arange(
                0, self.full_num_angles, angular_sub_sampling
            )
        self.init_angles = ray_trafo.angles[self.init_angle_inds]

        self.cur_angle_inds = list(self.init_angle_inds)
        self.acq_angle_inds = np.setdiff1d(
                np.arange(self.full_num_angles), 
                np.arange(0, self.full_num_angles, angular_sub_sampling)
            )
        self.proj_inds_per_angle = np.arange(
                np.prod(ray_trafo.obs_shape)
            ).reshape(ray_trafo.obs_shape)
        self.cur_proj_inds_list = [self.proj_inds_per_angle[a_ind] for a_ind in self.cur_angle_inds]

        assert self.proj_inds_per_angle.shape[0] == self.full_num_angles
        self.num_projs_per_angle = len(
                self.proj_inds_per_angle[self.acq_angle_inds[0]]
            )

        assert all(
            len(self.proj_inds_per_angle[a_ind]) == self.num_projs_per_angle for a_ind in self.init_angle_inds)

        assert all(
            len(self.proj_inds_per_angle[a_ind]) == self.num_projs_per_angle for a_ind in self.acq_angle_inds)

        self.acq_angle_inds = list(self.acq_angle_inds)
        self.acq_proj_inds_list = [self.proj_inds_per_angle[a_ind] for a_ind in self.acq_angle_inds]

    def get_best_inds_acquired(self, ) -> List:

        return [int(ind) for ind in self.cur_angle_inds if ind not in self.init_angle_inds]
    
    def update(self, top_projs_idx_to_be_added: List) -> List:

        self.top_k_acq_angle_inds = [self.acq_angle_inds[idx] for idx in top_projs_idx_to_be_added]
        top_k_acq_proj_inds_list = [self.proj_inds_per_angle[a_ind] for a_ind in self.top_k_acq_angle_inds]
        self.cur_angle_inds += self.top_k_acq_angle_inds
        self.cur_proj_inds_list = [self.proj_inds_per_angle[a_ind] for a_ind in self.cur_angle_inds]

        _reduced_acq_inds = np.setdiff1d(
            np.arange(len(self.acq_proj_inds_list)), 
            top_projs_idx_to_be_added
            )

        self.acq_angle_inds = [self.acq_angle_inds[idx] for idx in _reduced_acq_inds]
        self.acq_proj_inds_list = [self.proj_inds_per_angle[a_ind] for a_ind in self.acq_angle_inds]

        return top_k_acq_proj_inds_list

