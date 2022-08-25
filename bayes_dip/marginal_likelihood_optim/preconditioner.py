
from abc import ABC, abstractmethod
from typing import Callable, Optional
from functools import partial
import torch
from torch import Tensor
from ..probabilistic_models import LowRankObservationCov

class BasePreconditioner(ABC):

    def __init__(self,
        **update_kwargs,
        ) -> None:

        self.update(**update_kwargs)

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
        **kwargs
        ) -> Tensor:
        raise NotImplementedError

    def get_closure(self) -> Callable[[Tensor], Tensor]:
        return partial(self.matmul, use_inverse=True)

class LowRankObservationCovPreconditioner(BasePreconditioner):

    def __init__(self,
        low_rank_observation_cov: LowRankObservationCov,
        default_update_kwargs: Optional[dict] = None,
        ):  # pylint: disable=super-init-not-called

        self.low_rank_observation_cov = low_rank_observation_cov
        self.default_update_kwargs = default_update_kwargs or {}
        # do not call super().__init__(), low_rank_observation_cov should be updated already

    def sample(self,
        num_samples: int = 10,
        ) -> Tensor:
        return self.low_rank_observation_cov.sample(num_samples=num_samples)

    def matmul(self,
        v: Tensor,
        use_inverse: bool = False,
        ) -> Tensor:

        with torch.no_grad():
            v = self.low_rank_observation_cov.matmul(v, use_inverse=use_inverse)
        return v

    def update(self, **kwargs) -> None:
        merged_kwargs = self.default_update_kwargs.copy()
        merged_kwargs.update(kwargs)
        self.low_rank_observation_cov.update(**merged_kwargs)
