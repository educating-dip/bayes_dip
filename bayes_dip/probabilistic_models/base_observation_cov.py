"""Provides :class:`BaseObservationCov`"""
from typing import Tuple
from abc import ABC, abstractmethod
from torch import Tensor, nn

class BaseObservationCov(nn.Module, ABC):
    """
    Base class for covariance in observation space.
    """

    @abstractmethod
    def forward(self,
            v: Tensor
            ) -> Tensor:

        raise NotImplementedError
    
    @property
    @abstractmethod
    def shape(self,
            ) -> Tuple[int, int]:

        raise NotImplementedError
