from abc import ABC
from torch import Tensor

class BasePredictivePosterior(ABC):
    def __init__(self, observation_cov):
        self.observation_cov = observation_cov
        self.shape = self.observation_cov.image_cov.shape

    def sample(self, observation: Tensor, num_samples: int) -> Tensor:
        raise NotImplementedError

    def covariance(self, observation: Tensor, num_samples: int, patch_size: int = 1) -> Tensor:
        raise NotImplementedError

    def log_prob(self) -> float:
        raise NotImplementedError
