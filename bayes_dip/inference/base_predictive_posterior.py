"""
Provides :class:`BasePredictivePosterior`.
"""
from typing import Optional, Union
from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np
from bayes_dip.probabilistic_models import BaseObservationCov

class BasePredictivePosterior(ABC):
    """
    Base predictive posterior class.
    """
    def __init__(self,
            observation_cov: BaseObservationCov):
        """
        Parameters
        ----------
        observation_cov : :class:`BaseObservationCov`
            Observation covariance. Usually, the its parameters (i.e., the prior hyperparameters)
            should be optimized before.
        """
        self.observation_cov = observation_cov
        self.shape = self.observation_cov.image_cov.shape

    def sample(self,
            num_samples: int,
            mean: Tensor,
            noise_x_correction_term: Optional[float] = None
            ) -> Tensor:
        """
        Sample from the predictive posterior with the given ``mean``.

        Optional to implement by sub-classes.

        Parameters
        ----------
        num_samples : int
            Number of samples.
        mean : Tensor
            Predictive posterior mean. Shape: ``(1, 1, *self.observation_cov.trafo.im_shape)``.
        noise_x_correction_term : float, optional
            Noise amount that is assumed to be present in ground truth. Can help to stabilize
            computations.

        Returns
        -------
        samples : Tensor
            Samples. Shape: ``(num_samples, 1, *self.observation_cov.trafo.im_shape)``.
        """
        # child classes may add arguments
        raise NotImplementedError

    def covariance(self) -> Tensor:
        """
        Return the covariance matrix of the predictive posterior.

        Optional to implement by sub-classes.

        Returns
        -------
        cov : Tensor
            Covariance matrix. Shape: ``(np.prod(self.observation_cov.trafo.im_shape),) * 2``.
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self,
            mean: Tensor,
            ground_truth: Tensor,
            noise_x_correction_term: Optional[float] = None
            ) -> Union[float, np.float64]:
        """
        Return the log probability of ``ground_truth`` under the predictive posterior with the given
        ``mean``.

        Parameters
        ----------
        mean : Tensor
            Predictive posterior mean.
        ground_truth : Tensor
            Ground truth.
        noise_x_correction_term : float, optional
            Noise amount that is assumed to be present in ground truth. Can help to stabilize
            computations.

        Returns
        -------
        log_probability : float-like
            Log probability.
        """
        # child classes may add arguments
        raise NotImplementedError
