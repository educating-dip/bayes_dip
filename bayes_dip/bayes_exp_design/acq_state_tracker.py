from typing import Optional, Any, Tuple, Dict, Sequence
import gc
import numpy as np
import scipy
import functorch as ftch
import torch 

from torch import Tensor
from .base_angles_tracker import BaseAnglesTracker
from .update_cov_obs_mat import update_cov_obs_mat_no_noise

from bayes_dip.data import MatmulRayTrafo, BaseRayTrafo
from bayes_dip.probabilistic_models import (
        get_neural_basis_expansion, ParameterCov, ImageCov, ObservationCov, LowRankNeuralBasisExpansion)
from bayes_dip.marginal_likelihood_optim import marginal_likelihood_hyperparams_optim, get_preconditioner

def _get_observation_cov_module(
        nn_model: torch.nn.Module,
        ray_trafo: BaseRayTrafo,
        nn_input: Tensor,
        prior_assignment_dict: Dict,
        hyperparams_init_dict: Dict,
        use_gprior: bool = True, 
        scale_kwargs: Optional[Dict] = None,
        device: Optional[Any] = None, 
    ): 

    parameter_cov = ParameterCov(
        nn_model,
        prior_assignment_dict,
        hyperparams_init_dict,
        device=device
        )
    neural_basis_expansion = get_neural_basis_expansion(
        nn_model=nn_model,
        nn_input=nn_input,
        ordered_nn_params=parameter_cov.ordered_nn_params,
        nn_out_shape=nn_input.shape,
        use_gprior=use_gprior,
        trafo=ray_trafo,
        scale_kwargs=scale_kwargs
        )
    image_cov = ImageCov(
        parameter_cov=parameter_cov,
        neural_basis_expansion=neural_basis_expansion
        )
    observation_cov = ObservationCov(
        trafo=ray_trafo,
        image_cov=image_cov,
        device=device   
        )
    
    return observation_cov

def _get_ray_trafo_modules(
            ray_trafo_full: MatmulRayTrafo,
            angles_tracker: BaseAnglesTracker,
            device: Optional[Any] = None
        ):

    ray_trafo_module = MatmulRayTrafo(
            # reshaping of matrix rows to (len(cur_proj_inds_list), num_projs_per_angle) is row-major
            im_shape=ray_trafo_full.im_shape, 
            obs_shape=(len(angles_tracker.cur_proj_inds_list), angles_tracker.num_projs_per_angle),
            matrix=
                scipy.sparse.csr_matrix(
                ray_trafo_full.matrix[np.concatenate(angles_tracker.cur_proj_inds_list)].cpu().numpy()
            )
        ).to(device=device)
    if len(angles_tracker.acq_proj_inds_list) > 0:
        ray_trafo_comp_module = MatmulRayTrafo(
                # reshaping of matrix rows to (len(acq_proj_inds_list), num_projs_per_angle) is row-major
                im_shape=ray_trafo_full.im_shape,
                obs_shape=(len(angles_tracker.acq_proj_inds_list), angles_tracker.num_projs_per_angle),
                matrix=
                scipy.sparse.csr_matrix(
                    ray_trafo_full.matrix[np.concatenate(angles_tracker.acq_proj_inds_list)].cpu().numpy()
                )
            ).to(device=device)
    else:
        ray_trafo_comp_module = None
    return ray_trafo_module, ray_trafo_comp_module

class AcqStateTracker:
    def __init__(self,
            angles_tracker: BaseAnglesTracker, # proj_inds_per_angle, init_angle_inds, acq_angle_inds, total_num_acq_projs, acq_projs_batch_size
            nn_model: torch.nn.Module,
            observation_cov_kwargs: Dict,
            device: Optional[Any] = None
        ):

        self.device = device
        self.angles_tracker = angles_tracker
        self.ray_trafo_full = self.angles_tracker.ray_trafo
        self.ray_trafo_obj, self.ray_trafo_comp_obj = self._get_ray_trafo_modules(
                device=device
            )

        self.observation_cov_kwargs = observation_cov_kwargs
        self.observation_cov = _get_observation_cov_module(
            nn_model=nn_model,
            ray_trafo=self.ray_trafo_obj,
            device=self.device,
            **self.observation_cov_kwargs
        )

        assert not isinstance(
            self.observation_cov.image_cov.neural_basis_expansion, LowRankNeuralBasisExpansion)
        self.cov_obs_mat_no_noise = self.observation_cov.assemble_observation_cov(
                use_noise_variance=False
        )

    def _get_ray_trafo_modules(self,
            device: Optional[Any] = None
        ) -> Tuple[MatmulRayTrafo, MatmulRayTrafo]:

        return _get_ray_trafo_modules(ray_trafo_full=self.ray_trafo_full, 
                    angles_tracker=self.angles_tracker, device=device)
    
    def state_update(self,
            top_projs_idx: Sequence, 
            batch_size: int, 
            use_precomputed_best_inds: bool = False
        ) -> None:

        # update lists of acquired and not yet acquired projections
        self.angles_tracker.update( top_projs_idx=top_projs_idx   ) 

        # only updates attributes: cur_angle_inds and acq_angle_inds and lists 
        if not use_precomputed_best_inds:
            ray_trafo_top_k_obj = MatmulRayTrafo(
                    im_shape=self.ray_trafo_full.im_shape, 
                    obs_shape=(self.angles_tracker.acq_projs_batch_size, 
                        self.angles_tracker.num_projs_per_angle), 
                    matrix=scipy.sparse.csr_matrix(
                        self.ray_trafo_full.matrix[np.concatenate(
                                self.angles_tracker.top_k_acq_proj_inds_list)].cpu().numpy()
                            )
                ).to(device=self.device)
            self.cov_obs_mat_no_noise = update_cov_obs_mat_no_noise(
                    observation_cov=self.observation_cov, 
                    ray_trafo_obj=self.ray_trafo_obj, 
                    ray_trafo_top_k_obj=ray_trafo_top_k_obj,
                    cov_obs_mat_no_noise=self.cov_obs_mat_no_noise,
                    batch_size=batch_size
                )
        # update transform
        self.ray_trafo_obj, self.ray_trafo_comp_obj = self._get_ray_trafo_modules(
                device=self.device
            )
        # update observation_cov
        self.observation_cov.update_trafo(  self.ray_trafo_obj  )

    def model_update(self,
            refined_model: torch.nn.Module, 
        ):

        self.observation_cov = _get_observation_cov_module(
            nn_model=refined_model,
            ray_trafo=self.ray_trafo_obj,
            device=self.device, 
            **self.observation_cov_kwargs
            )

        del self.cov_obs_mat_no_noise
        gc.collect(); torch.cuda.empty_cache()
        self.cov_obs_mat_no_noise = self.observation_cov.assemble_observation_cov(
                use_noise_variance=False    )