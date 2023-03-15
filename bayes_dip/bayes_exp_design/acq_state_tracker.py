from typing import Optional, Any, Tuple, Dict
import numpy as np
import scipy
import functorch as ftch

from torch import Tensor
from .base_angles_tracker import BaseAnglesTracker

from bayes_dip.data import MatmulRayTrafo
from bayes_dip.marginal_likelihood_optim import marginal_likelihood_hyperparams_optim, get_preconditioner
from bayes_dip.probabilistic_models import (ObservationCov, 
        GpriorNeuralBasisExpansion, LowRankNeuralBasisExpansion)

def _get_ray_trafo_modules(
        ray_trafo_full: MatmulRayTrafo,
        angles_tracker: BaseAnglesTracker,
        dtype: Optional[Any] = None,
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
        ).to(dtype=dtype, device=device)
    if len(angles_tracker.acq_proj_inds_list) > 0:
        ray_trafo_comp_module = MatmulRayTrafo(
                # reshaping of matrix rows to (len(acq_proj_inds_list), num_projs_per_angle) is row-major
                im_shape=ray_trafo_full.im_shape,
                obs_shape=(len(angles_tracker.acq_proj_inds_list), angles_tracker.num_projs_per_angle),
                matrix=
                scipy.sparse.csr_matrix(
                    ray_trafo_full.matrix[np.concatenate(angles_tracker.acq_proj_inds_list)].cpu().numpy()
                )
            ).to(dtype=dtype, device=device)
    else:
        ray_trafo_comp_module = None
    return ray_trafo_module, ray_trafo_comp_module

def _update_hyperparams_via_marglik(
        observation_cov: ObservationCov, 
        noisy_observation: Tensor,
        recon: Tensor,
        marglik_kwargs: Dict,
        preconditioner_kwargs: Optional[Dict] = None) -> None:

        # update preconditioner 
        if preconditioner_kwargs is not None: 
            updated_preconditioner = get_preconditioner(
                observation_cov=observation_cov,
                kwargs=preconditioner_kwargs
            )
            marglik_kwargs['linear_cg']['preconditioner'] = updated_preconditioner

        marginal_likelihood_hyperparams_optim(
            observation_cov=observation_cov,
            observation=noisy_observation,
            recon=recon,
            linearized_weights=None, # TODO:
            optim_kwargs=marglik_kwargs,
            log_path='./',
        )
    

class AcqStateTracker:
    def __init__(self,
        angles_tracker: BaseAnglesTracker, # proj_inds_per_angle, init_angle_inds, acq_angle_inds, total_num_acq_projs, acq_projs_batch_size
        observation_cov: ObservationCov,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None
        ):

        self.device = device
        self.dtype = dtype

        self.angles_tracker = angles_tracker
        self.ray_trafo_full = self.angles_tracker.ray_trafo
        self.ray_trafo_obj, self.ray_trafo_comp_obj = self._get_ray_trafo_modules(
            dtype=dtype, 
            device=device
            )

        self.observation_cov = observation_cov
        self.observation_cov_init_state_dict = self.observation_cov.state_dict()
        self.trafo = observation_cov.trafo
        self.neural_basis_expansion = self.observation_cov.image_cov.neural_basis_expansion
        assert not isinstance(
            self.neural_basis_expansion, LowRankNeuralBasisExpansion)

    def _get_ray_trafo_modules(self,
            dtype: Optional[Any] = None, 
            device: Optional[Any] = None
        ) -> Tuple[MatmulRayTrafo, MatmulRayTrafo]:

        return _get_ray_trafo_modules(ray_trafo_full=self.ray_trafo_full, 
                    angles_tracker=self.angles_tracker, dtype=dtype, device=device)
    
    def state_update(self,
        update_neural_basis_from_refined_model: Optional[Any] = None,
        marglik_update_kwargs: Optional[Tuple[Dict, Dict]] = None,
        scale_update_kwargs: Optional[Dict] = None,
        noisy_observation: Optional[Tensor] = None, 
        recon: Optional[Tensor] = None
        ) -> None:
        
        # update transform
        self.ray_trafo_obj, self.ray_trafo_comp_obj =_get_ray_trafo_modules(
            ray_trafo_full=self.ray_trafo_full,
            angles_tracker=self.angles_tracker, 
            dtype=self.dtype, 
            device=self.device
        )
        # update observation_cov
        self.observation_cov.trafo = self.ray_trafo_obj
        self.observation_cov.shape = (
                np.prod(self.ray_trafo_obj.obs_shape),
            ) * 2
        
        if hasattr(self.neural_basis_expansion, 'trafo'): 
            self.neural_basis_expansion.trafo = self.ray_trafo_obj
        
        if update_neural_basis_from_refined_model is not None:
            self.neural_basis_expansion.nn_model = update_neural_basis_from_refined_model
            self.neural_basis_expansion.func_model_with_input, self.neural_basis_expansion.func_params = ftch.make_functional(
                    update_neural_basis_from_refined_model
                )
            if isinstance(self.neural_basis_expansion, GpriorNeuralBasisExpansion):
                self.neural_basis_expansion.update_scale(**scale_update_kwargs)
        
        if marglik_update_kwargs is not None:
            
            marglik_kwargs, preconditioner_kwargs = marglik_update_kwargs
            _update_hyperparams_via_marglik(
                observation_cov=self.observation_cov,
                marglik_kwargs=marglik_kwargs, 
                preconditioner_kwargs=preconditioner_kwargs, 
                noisy_observation=noisy_observation, 
                recon=recon
            )

