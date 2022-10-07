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
        """
        Multiply with the covariance "matrix".

        I.e., evaluate ``(cov @ v.view(v.shape[0], -1).T).T.view(*v.shape)`` where ``cov`` is a
        matrix representation of ``self``.

        Parameters
        ----------
        v : Tensor
            Images. Shape: ``(batch_size, 1, H, W)``.

        Returns
        -------
        Tensor
            Products. Shape: same as ``v``.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self,
            num_samples: int) -> Tensor:
        """
        Sample from a Gaussian with this covariance.

        Unless specified using an additional argument in a sub-class, the mean should be zero.

        Parameters
        ----------
        num_samples : int
            Number of samples.

        Returns
        -------
        Tensor
            Samples. Shape: ``(num_samples, 1, H, W)``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self
            ) -> Tuple[int, int]:
        """Shape of the (theoretical) matrix representation."""
        raise NotImplementedError
