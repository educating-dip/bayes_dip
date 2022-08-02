
import torch
from abc import ABC, abstractmethod
from torch import Tensor
from ..probabilistic_models import LowRankObservationCov

class BasePreC(ABC):

    def __init__(self,
        use_cpu: bool = False
        ) -> None:

        super().__init__()
        self.update(use_cpu=use_cpu)

    @abstractmethod
    def sample(self,
            num_samples: int
            ) -> Tensor:

        raise NotImplementedError

    @abstractmethod
    def matmul(self,
            v: Tensor,
            use_inverse: bool,
            ) -> Tensor:

        raise NotImplementedError

    @abstractmethod
    def update(self,
        use_cpu: bool
        ) -> Tensor:

        raise NotImplementedError

class LowRankPreC(BasePreC):

    def __init__(self,
        pre_con_obj: LowRankObservationCov
        ):

        self.pre_con_obj = pre_con_obj
        super().__init__()

    def sample(self,
        num_samples: int = 10,
        ):

        normal_std = torch.randn(
            self.pre_con_obj.shape[0], num_samples,
            device=self.pre_con_obj.device
            )
        normal_low_rank = torch.randn(
            self.pre_con_obj.low_rank_rank_dim, num_samples,
            device=self.pre_con_obj.device
            )
        samples = normal_std * torch.exp(self.pre_con_obj.log_noise_variance).pow(0.5) + (
            self.pre_con_obj.U * self.pre_con_obj.L.pow(0.5) ) @ normal_low_rank
        return samples

    def matmul(self,
        v: Tensor,
        use_inverse: bool = False,
        ):

        if not use_inverse:
            return self.pre_con_obj.U @ ( self.L[:, None] * (self.U.T @ v) ) + self.noise_model_variance_obs_and_eps * v
        else:
            return ( v / self.noise_model_variance_obs_and_eps) - ( self.U @ torch.linalg.solve(
                self.sysmat, self.U.T @ v.T / (self.noise_model_variance_obs_and_eps ** 2) ) ).T

    def update(self,
        eps: float = 1e-3,
        full_diag_eps: float = 1e-6,
        use_cpu: bool = False
        ):

        self.U, self.L = self.pre_con_obj.get_batched_low_rank_observation_cov_basis(
            eps=eps,
            use_cpu=use_cpu
        )
        self.noise_model_variance_obs_and_eps = self.pre_con_obj.log_noise_variance.exp() + full_diag_eps
        self.sysmat = torch.diag( 1 / (self.L) ) + self.U.T @ self.U  / self.noise_model_variance_obs_and_eps