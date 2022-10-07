"""
Provides :class:`BaseRayTrafo`.
"""

from typing import Union, Tuple
from abc import ABC, abstractmethod
from torch import nn
from torch import Tensor
import numpy as np


class BaseRayTrafo(nn.Module, ABC):
    """
    Abstract base ray transform.

    Attributes
    ----------
    im_shape : 2-tuple or 3-tuple of int
        Image shape.
        For 2D geometries: ``(im_0, im_1)``.
        For 3D geometries: ``(im_0, im_1, im_2)``.
    obs_shape : 2-tuple or 3-tuple of int
        Observation shape.
        For 2D geometries: ``(angles, det_cols)``.
        For 3D geometries: ``(det_rows, angles, det_cols)``.
    """

    def __init__(self,
            im_shape: Union[Tuple[int, int], Tuple[int, int, int]],
            obs_shape: Union[Tuple[int, int], Tuple[int, int, int]]):
        """
        Parameters
        ----------
        im_shape : 2-tuple or 3-tuple of int
            Image shape.
            For 2D geometries: ``(im_0, im_1)``.
            For 3D geometries: ``(im_0, im_1, im_2)``.
        obs_shape : 2-tuple or 3-tuple of int
            Observation shape.
            For 2D geometries: ``(angles, det_cols)``.
            For 3D geometries: ``(det_rows, angles, det_cols)``.
        """
        super().__init__()  # nn.Module.__init__()
        self.im_shape = im_shape
        self.obs_shape = obs_shape

    @property
    def angles(self) -> np.ndarray:
        """:class:`np.ndarray` : The angles (in radian)."""
        raise NotImplementedError

    @abstractmethod
    def trafo(self, x: Tensor) -> Tensor:
        """
        Apply the forward projection.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Image of attenuation.
            Shape for 2D geometries: ``(batch, channels, im_0, im_1)``.
            Shape for 3D geometries: ``(batch, channels, im_0, im_1, im_2)``.

        Returns
        -------
        observation : :class:`torch.Tensor`
            Forward projection.
            Shape for 2D geometries: ``(batch, channels, angles, det_cols)``.
            Shape for 3D geometries: ``(batch, channels, det_rows, angles, det_cols)``.
        """
        raise NotImplementedError

    # sub-classes use this function for trafo() if trafo_flat() is implemented
    def _trafo_via_trafo_flat(self, x: Tensor) -> Tensor:
        batch_dim, channel_dim = x.shape[:2]
        x_flat = x.reshape(
                batch_dim * channel_dim, np.prod(self.im_shape)).T
        observation_flat = self.trafo_flat(x_flat)
        return observation_flat.T.reshape(batch_dim, channel_dim, *self.obs_shape)

    @abstractmethod
    def trafo_flat(self, x: Tensor) -> Tensor:
        """
        Apply the forward projection; flat version of :meth:`trafo`.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Image of attenuation.
            1D or 2D tensor of shape ``(im_numel,)`` or ``(im_numel, batch)``.

        Returns
        -------
        observation : :class:`torch.Tensor`
            Forward projection.
            1D or 2D tensor of shape ``(obs_numel,)`` or ``(obs_numel, batch)``.
        """
        raise NotImplementedError

    # sub-classes use this function for trafo_flat() if trafo() is implemented
    def _trafo_flat_via_trafo(self, x: Tensor) -> Tensor:
        batch_dim = x.shape[1]
        x_reshaped = x.T.reshape(
                1, batch_dim, *self.im_shape)
        observation = self.trafo(x_reshaped)
        return observation.reshape(batch_dim, np.prod(self.obs_shape)).T

    @abstractmethod
    def trafo_adjoint(self, observation: Tensor) -> Tensor:
        """
        Apply the adjoint operation (sometimes implemented as a back-projection;
        for an accurately matching adjoint of the discrete forward projection
        operation :meth:`trafo`, one might consider assembling the matrix
        describing :meth:`trafo` and multiply with the transposed matrix in this
        function).

        Parameters
        ----------
        observation : :class:`torch.Tensor`
            Projection values.
            Shape for 2D geometries: ``(batch, channels, angles, det_cols)``.
            Shape for 3D geometries: ``(batch, channels, det_rows, angles, det_cols)``.

        Returns
        -------
        x : :class:`torch.Tensor`
            Result of the adjoint operation (back-projection).
            Shape for 2D geometries: ``(batch, channels, im_0, im_1)``.
            Shape for 3D geometries: ``(batch, channels, im_0, im_1, im_2)``.
        """
        raise NotImplementedError

    # sub-classes use this function for trafo_adjoint() if trafo_adjoint_flat()
    # is implemented
    def _trafo_adjoint_via_trafo_adjoint_flat(
            self, observation: Tensor) -> Tensor:

        batch_dim, channel_dim = observation.shape[:2]
        observation_flat = observation.reshape(
                batch_dim * channel_dim, np.prod(self.obs_shape)).T
        x_flat = self.trafo_adjoint_flat(observation_flat)
        return x_flat.T.reshape(batch_dim, channel_dim, *self.im_shape)

    @abstractmethod
    def trafo_adjoint_flat(self, observation: Tensor) -> Tensor:
        """
        Apply the adjoint operation; flat version of :meth:`trafo_adjoint`.

        Parameters
        ----------
        observation : :class:`torch.Tensor`
            Projection values.
            1D or 2D tensor of shape ``(obs_numel,)`` or ``(obs_numel, batch)``.

        Returns
        -------
        x : :class:`torch.Tensor`
            Result of the adjoint operation (back-projection).
            1D or 2D tensor of shape ``(im_numel,)`` or ``(im_numel, batch)``.
        """
        raise NotImplementedError

    # sub-classes use this function for trafo_adjoint_flat() if trafo_adjoint()
    # is implemented
    def _trafo_adjoint_flat_via_trafo_adjoint(
            self, observation: Tensor) -> Tensor:

        batch_dim = observation.shape[1]
        observation_reshaped = observation.T.reshape(
                1, batch_dim, *self.obs_shape)
        x = self.trafo_adjoint(observation_reshaped)
        return x.reshape(batch_dim, np.prod(self.im_shape)).T

    def fbp(self, observation: Tensor) -> Tensor:
        """
        Apply a filtered back-projection.

        Parameters
        ----------
        observation : :class:`torch.Tensor`
            Projection values.
            Shape for 2D geometries: ``(batch, channels, angles, det_cols)``.
            Shape for 3D geometries: ``(batch, channels, det_rows, angles, det_cols)``.

        Returns
        -------
        x : :class:`torch.Tensor`
            Filtered back-projection.
            Shape for 2D geometries: ``(batch, channels, im_0, im_1)``.
            Shape for 3D geometries: ``(batch, channels, im_0, im_1, im_2)``.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """See :meth:`trafo`."""
        return self.trafo(x)
