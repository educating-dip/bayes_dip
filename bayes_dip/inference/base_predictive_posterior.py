"""
Provides :class:`BasePredictivePosterior`.
"""
from abc import ABC, abstractmethod
from torch import Tensor
from bayes_dip.probabilistic_models import BaseObservationCov

class BasePredictivePosterior(ABC):
    def __init__(self,
            observation_cov: BaseObservationCov):
        self.observation_cov = observation_cov
        self.shape = self.observation_cov.image_cov.shape

    def sample(self, num_samples: int, *args, **kwargs) -> Tensor:
        # child classes may add arguments
        raise NotImplementedError

    def covariance(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self,
            mean: Tensor,
            ground_truth: Tensor,
            *args,
            noise_x_correction_term: float = 0.,
            **kwargs) -> float:
        # child classes may add arguments
        raise NotImplementedError
