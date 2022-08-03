from abc import ABC, abstractmethod
from torch import Tensor
from bayes_dip.probabilistic_models import BaseObservationCov

class BasePredictivePosterior(ABC):
    def __init__(self,
            observation_cov: BaseObservationCov):
        self.observation_cov = observation_cov
        self.shape = self.observation_cov.image_cov.shape

    def sample(self, observation: Tensor, num_samples: int) -> Tensor:
        raise NotImplementedError

    def covariance(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self) -> float:
        raise NotImplementedError
