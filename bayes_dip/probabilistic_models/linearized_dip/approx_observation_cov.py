"""Provides :class:`LowRankObservationCov`"""
from typing import Optional
from math import ceil
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm

from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from ..base_image_cov import BaseImageCov
from ..observation_cov import ObservationCov


class LowRankObservationCov(ObservationCov):

    def __init__(self,
        trafo: BaseRayTrafo,
        image_cov: BaseImageCov,
        init_noise_variance: float = 1.,
        low_rank_rank_dim: int = 100, 
        oversampling_param: int = 5,
        vec_batch_size: int = 1,
        load_approx_basis_from: Optional['str'] = None,
        device=None,
        ) -> None:
        """
        Parameters
        ----------
        TODO
        """

        super().__init__(trafo=trafo,
            image_cov=image_cov,
            init_noise_variance=init_noise_variance,
            device=device
        )

        self.low_rank_rank_dim = low_rank_rank_dim
        self.oversampling_param = oversampling_param
        self.vec_batch_size = vec_batch_size
        self.random_matrix = self._assemble_random_matrix()

        if load_approx_basis_from:
            #TODO: load U, S and Vh
            self.load_low_rank_observation_cov(load_approx_basis_from)
            raise NotImplementedError

    def load_low_rank_observation_cov(self, path):
        pass

    def save_low_rank_observation_cov(self, ):
        pass

    def _assemble_random_matrix(self, ) -> Tensor:
        low_rank_rank_dim = self.low_rank_rank_dim + self.oversampling_param
        random_matrix = torch.randn(
            (low_rank_rank_dim, np.prod(self.trafo.obs_shape) 
                ),
            device=self.device
            )
        return random_matrix
    
    def get_batched_low_rank_observation_cov_basis(self,
        use_cpu: bool = False,
        eps: float = 1e-3,
        ):
        
        num_batches = ceil((self.low_rank_rank_dim + self.oversampling_param ) / self.vec_batch_size)
        v_cov_obs_mat = []
        for i in tqdm(range(num_batches), miniters=num_batches//100, desc='get_cov_obs_low_rank'):
            rnd_vect = self.random_matrix[i * self.vec_batch_size:(i * self.vec_batch_size) + self.vec_batch_size, :].unsqueeze(dim=1)
            eff_batch_size = rnd_vect.shape[0]
            if eff_batch_size < self.vec_batch_size:
                rnd_vect = torch.cat(
                    [rnd_vect, torch.zeros(
                            (self.vec_batch_size-eff_batch_size, *rnd_vect.shape[1:]), 
                                dtype=rnd_vect.dtype,
                                device=rnd_vect.device)
                        ]
                    )
            v = super().forward( 
                rnd_vect,
                use_noise_variance=False)
            if eff_batch_size < self.vec_batch_size:
                v = v[:eff_batch_size]
            v_cov_obs_mat.append(v)
        v_cov_obs_mat = torch.cat(v_cov_obs_mat)
        v_cov_obs_mat = v_cov_obs_mat.view(*v_cov_obs_mat.shape[:1], -1).T
        Q, _ = torch.linalg.qr(v_cov_obs_mat.detach().cpu() if use_cpu else v_cov_obs_mat)
        Q = Q if not use_cpu else Q.to(self.device)
        random_matrix = self.random_matrix.view(self.random_matrix.shape[0], -1)
        B = torch.linalg.solve(random_matrix @ Q, v_cov_obs_mat.T @ Q)
        L, V = torch.linalg.eig(B)
        U = Q @ V.real
        return U[:, :self.low_rank_rank_dim], L.real[:self.low_rank_rank_dim].clamp_(min=eps)

