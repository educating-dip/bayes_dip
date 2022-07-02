import torch
import numpy as np
from torch import Tensor, nn
from tqdm import tqdm
from functools import lru_cache

from .linearized_dip.image_cov import ImageCov
from ..data.trafo.base_ray_trafo import BaseRayTrafo
from bayes_dip.utils import bisect_left  # for python >= 3.10 one can use instead: from bisect import bisect_left

class ObservationCov(nn.Module):

    def __init__(self,
        trafo: BaseRayTrafo,
        image_cov: ImageCov,
        init_noise_variance: float = 1., 
        device=None, 
        ) -> None:

        super().__init__()

        self.trafo = trafo
        self.image_cov = image_cov
        self.device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

        self.log_noise_variance = nn.Parameter(
                torch.tensor(np.log(init_noise_variance), device=self.device),
            )

    def forward(self, 
                v: Tensor,
                use_noise_variance: bool = True, 
                **kwargs
            ) -> Tensor:

        v_ = self.trafo.trafo_adjoint(v)
        v_ = self.image_cov(v_)
        v_ = self.trafo(v_)
        
        v = v_ +  v * self.log_noise_variance.exp() if use_noise_variance else v_

        return v
    
    def assemble_observation_cov(self,
        vec_batch_size: int = 1, 
        use_noise_variance: bool = True,
        return_on_cpu: bool = False, 
        sub_slice_batches = None, 
        ) -> Tensor:
        
        obs_shape = (1, 1,) + self.trafo.obs_shape
        obs_numel = np.prod(obs_shape)
        if sub_slice_batches is None:
            sub_slice_batches = slice(None)
        rows = []
        v = torch.empty((vec_batch_size,) + obs_shape, device=self.device)

        for i in tqdm(np.array(range(0, obs_numel, vec_batch_size))[sub_slice_batches], 
                    desc='get_prior_cov_obs_mat', miniters=obs_numel//vec_batch_size//100
                ):

            v[:] = 0.
            # set v.view(vec_batch_size, -1) to be a subset of rows of torch.eye(obs_numel); in last batch, it may contain some additional (zero) rows
            v.view(vec_batch_size, -1)[:, i:i+vec_batch_size].fill_diagonal_(1.)

            rows_batch = self.forward(
                v, 
                use_noise_variance=use_noise_variance)

            rows_batch = rows_batch.view(vec_batch_size, -1)

            if i+vec_batch_size > obs_numel:  # last batch
                rows_batch = rows_batch[:obs_numel%vec_batch_size]
            rows_batch = rows_batch.cpu()  # collect on CPU (saves memory while running the closure)
            rows.append(rows_batch)

        observation_cov_mat = torch.cat(rows, dim=0)

        return observation_cov_mat if return_on_cpu else observation_cov_mat.to(self.device)
    
    @classmethod
    def get_stabilizing_eps(cls, 
        observation_cov_mat: Tensor, 
        eps_mode: str, 
        eps: float, 
        eps_min_for_auto: float = 1e-6, 
        include_zero_for_auto: bool = True
        ) -> float:

        observation_cov_mat_diag_mean = observation_cov_mat.diag().mean().item()

        if eps_mode == 'abs':
            observation_cov_mat_eps = eps or 0.
        elif eps_mode == 'rel_mean_diag':
            observation_cov_mat_eps = (eps or 0.) * observation_cov_mat.diag().mean().item()
        elif eps_mode == 'auto_abs' or eps_mode == 'auto_rel_mean_diag':
            @lru_cache(maxsize=None)
            def observation_cov_mat_cholesky_decomposable(eps_value):
                try:
                    _ = torch.linalg.cholesky(
                            observation_cov_mat + eps_value * torch.eye(
                                observation_cov_mat.shape[0],
                                device=observation_cov_mat.device
                        )
                        )
                except RuntimeError:
                    return False
                return True
            fct = 1. if eps_mode == 'auto_abs' else observation_cov_mat_diag_mean
            assert eps >= eps_min_for_auto, "eps_min_for_auto must be lower than eps"
            # both eps and eps_min_for_auto are relative to fct
            eps_to_search = list(np.logspace(np.log10(eps_min_for_auto / eps), 0, 1000) * eps * fct)
            if include_zero_for_auto: 
                eps_to_search = [0.] + eps_to_search
            i_eps = bisect_left(eps_to_search, True, key=observation_cov_mat_cholesky_decomposable)

            assert i_eps < len(eps_to_search), ('failed to make Kyy cholesky decomposable,' 
                f' max eps is {eps_to_search[-1]} == {eps_to_search[-1] / fct} * Kyy.diag().mean()')

            observation_cov_mat_eps = eps_to_search[i_eps]
        elif eps_mode is None or eps_mode.lower() == 'none':
            observation_cov_mat_eps = 0.
        else:
            raise NotImplementedError

        return observation_cov_mat_eps