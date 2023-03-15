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
        
        self.ray_trafo = ray_trafo # full ray trafo (all possible acquirable angles)
        self.full_num_angles = len(self.ray_trafo.angles)
        self.total_num_acq_projs = total_num_acq_projs
        self.acq_projs_batch_size = acq_projs_batch_size

        self.init_angle_inds = np.arange( # init angle indices at the start of acquisition procedure
                0, self.full_num_angles, angular_sub_sampling)
        # init angles at the start of acquisition procedure
        self.init_angles = self.ray_trafo.angles[self.init_angle_inds]
        # angles acquired initilised to init_angle_inds
        self.cur_angle_inds = list(self.init_angle_inds) 
        # angles that can be acquired not in init_angle_inds
        self.acq_angle_inds = np.setdiff1d(np.arange(self.full_num_angles), self.init_angle_inds)
        self.proj_inds_per_angle = np.arange( # proj_inds = angles * det_pixel_dim
                    np.prod(self.ray_trafo.obs_shape)
            ).reshape(self.ray_trafo.obs_shape) # list of abs indices that defines the det_pixels for this angle
        self.cur_proj_inds_list = [self.proj_inds_per_angle[a_ind] for a_ind in self.cur_angle_inds]

        assert self.proj_inds_per_angle.shape[0] == self.full_num_angles
        self.num_projs_per_angle = len(
                self.proj_inds_per_angle[self.acq_angle_inds[0]])
        assert all(
            len(self.proj_inds_per_angle[a_ind]) == self.num_projs_per_angle for a_ind in self.init_angle_inds)
        assert all(
            len(self.proj_inds_per_angle[a_ind]) == self.num_projs_per_angle for a_ind in self.acq_angle_inds)

        self.acq_angle_inds = list(self.acq_angle_inds)
        self.acq_proj_inds_list = [self.proj_inds_per_angle[a_ind] for a_ind in self.acq_angle_inds]

    def get_best_inds_acquired(self, ) -> List:

        return [int(ind) for ind in self.cur_angle_inds if ind not in self.init_angle_inds]
    
    def update(self, 
        top_projs_idx: List
        ) -> List:

        self.top_k_acq_angle_inds = [self.acq_angle_inds[idx] for idx in top_projs_idx]
        top_k_acq_proj_inds_list = [self.proj_inds_per_angle[a_ind] for a_ind in self.top_k_acq_angle_inds]
        
        self.cur_angle_inds += self.top_k_acq_angle_inds
        self.cur_proj_inds_list = [self.proj_inds_per_angle[a_ind] for a_ind in self.cur_angle_inds]

        _reduced_acq_inds = np.setdiff1d(
            np.arange(len(self.acq_proj_inds_list)), 
            top_projs_idx
            )

        self.acq_angle_inds = [self.acq_angle_inds[idx] for idx in _reduced_acq_inds]
        self.acq_proj_inds_list = [self.proj_inds_per_angle[a_ind] for a_ind in self.acq_angle_inds]

        return top_k_acq_proj_inds_list

