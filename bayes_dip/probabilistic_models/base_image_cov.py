"""Provides :class:`BaseImageCov`"""
from typing import Tuple
from abc import ABC, abstractmethod
from torch import Tensor, nn

class BaseImageCov(nn.Module, ABC):
    """
    Base class for covariance in image space.
    """

    @abstractmethod
    def forward(self,
            v: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self,
            num_samples: int) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self
            ) -> Tuple[int, int]:
        raise NotImplementedError
