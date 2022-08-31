"""Provides :class:`BaseObservationCov`"""
from typing import Tuple
from abc import ABC, abstractmethod
from torch import Tensor, nn
import torch
import numpy as np
from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from .base_image_cov import BaseImageCov

class BaseObservationCov(nn.Module, ABC):
    """
    Base class for covariance in observation space.
    """

    def __init__(self,
        trafo: BaseRayTrafo,
        image_cov: BaseImageCov,
        init_noise_variance: float = 1.,
        device=None,
        ) -> None:
        """
        Parameters
        ----------
        trafo : :class:`bayes_dip.data.BaseRayTrafo`
            Ray transform.
        image_cov : :class:`bayes_dip.probabilistic_models.BaseImageCov`
            Image space covariance module.
        init_noise_variance : float, optional
            Initial value for noise variance parameter. The default is `1.`.
        device : str or torch.device, optional
            Device. If `None` (the default), `'cuda:0'` is chosen if available or `'cpu'` otherwise.
        """

        super().__init__()

        self.trafo = trafo
        self.image_cov = image_cov
        self.device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

        self.log_noise_variance = nn.Parameter(
                torch.tensor(float(np.log(init_noise_variance)), device=self.device),
            )

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the (theoretical) matrix representation."""
        return (np.prod(self.trafo.obs_shape),) * 2

    @abstractmethod
    def forward(self,
            v: Tensor
            ) -> Tensor:
        """
        Parameters
        ----------
        v : Tensor
            Input. Shape: ``(batch_size, 1, *self.trafo.obs_shape)``
        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, 1, *self.trafo.obs_shape)``
        """
        raise NotImplementedError
