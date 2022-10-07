"""
Provides base classes for neural basis expansion.
"""
from typing import Optional, Sequence, Tuple
from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
import numpy as np

from bayes_dip.utils import eval_mode
from ..utils import get_inds_from_ordered_params, get_slices_from_ordered_params


class BaseNeuralBasisExpansion(ABC):
    """
    Class for Jacobian vector products and vector Jacobian products.

    This class stores all the stateful information needed for these operations as attributes
    and exposes just the :meth:`jvp` and :meth:`vjp` methods.
    """

    def __init__(self,
            nn_model: nn.Module,
            nn_input: Tensor,
            ordered_nn_params: Sequence[nn.Parameter],
            nn_out_shape: Optional[Tuple[int, int, int, int]] = None,
            ) -> None:
        """
        Parameters
        ----------
        nn_model : :class:`nn.Module`
            Network.
        nn_input : Tensor
            Network input.
        ordered_nn_params : sequence of nn.Parameter
            Sequence of parameters that should be included in this expansion.
        nn_out_shape : 4-tuple of int, optional
            Shape of the network output. If not specified, it is determined by a forward call
            (performed in eval mode).
        """

        self.nn_model = nn_model
        self.nn_input = nn_input

        self.ordered_nn_params = ordered_nn_params
        self.inds_from_ordered_params = get_inds_from_ordered_params(
                self.nn_model, self.ordered_nn_params
            )
        self.slices_from_ordered_params = get_slices_from_ordered_params(
                self.ordered_nn_params
            )
        self.num_params = sum(param.data.numel() for param in self.ordered_nn_params)

        self.nn_out_shape = nn_out_shape
        if self.nn_out_shape is None:
            with torch.no_grad(), eval_mode(self.nn_model):
                self.nn_out_shape = self.nn_model(nn_input).shape

    @property
    def jac_shape(self) -> Tuple[int, int]:
        """
        2-tuple of int
            Shape of the Jacobian matrix, i.e. ``(np.prod(self.nn_out_shape), self.num_params)``.
        """
        return (np.prod(self.nn_out_shape), self.num_params)

    @abstractmethod
    def jvp(self, v: Tensor) -> Tensor:
        """
        Evaluate a batch of Jacobian vector products.

        I.e., evaluate ``(J @ v.T).T`` where ``J`` is the Jacobian matrix.

        Parameters
        ----------
        v : Tensor
            Input. Shape: ``(batch_size, self.num_params)``

        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, *self.nn_out_shape)``
        """
        raise NotImplementedError

    @abstractmethod
    def vjp(self, v: Tensor) -> Tensor:
        """
        Evaluate a batch of vector Jacobian products.

        I.e., evaluate ``v.view(v.shape[0], -1) @ J`` where ``J`` is the Jacobian matrix.

        Parameters
        ----------
        v : Tensor
            Input. Shape: ``(batch_size, *self.nn_out_shape)``

        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, self.num_params)``
        """
        raise NotImplementedError

class BaseMatmulNeuralBasisExpansion(BaseNeuralBasisExpansion):
    """
    Partial implementation of :class:`BaseNeuralBasisExpansion` using matmul with the assembled
    Jacobian matrix for the :meth:`jvp` and :meth:`vjp` methods.

    The Jacobian matrix property :attr:`matrix` and methods assembling it (i.e., :meth:`get_matrix`
    and :meth:`update_matrix`) are abstract in this class.
    """

    @property
    @abstractmethod
    def matrix(self):
        """
        Tensor
            Jacobian matrix. Shape: ``self.jac_shape``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_matrix(self) -> Tensor:
        """
        Assemble and return the Jacobian matrix.

        Returns
        -------
        matrix : Tensor
            Jacobian matrix. Shape: ``self.jac_shape``.
        """
        raise NotImplementedError

    @abstractmethod
    def update_matrix(self) -> None:
        """
        Assemble and update the Jacobian matrix :attr:`matrix`.
        """
        raise NotImplementedError

    def jvp(self, v: Tensor, sub_slice: slice = None) -> Tensor:
        """
        Parameters
        ----------
        v : Tensor
            Input. Shape: ``(batch_size, self.num_params)``

        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, *self.nn_out_shape)``
        """
        matrix = self.matrix if sub_slice is None else self.matrix[:, sub_slice]
        jvp = (matrix @ v.T).T
        return jvp.view(v.shape[0], *self.nn_out_shape)

    def vjp(self, v: Tensor, sub_slice: slice = None) -> Tensor:
        """
        Parameters
        ----------
        v : Tensor
            Input. Shape: ``(batch_size, *self.nn_out_shape)``

        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, self.num_params)``
        """
        matrix = self.matrix if sub_slice is None else self.matrix[:, sub_slice]
        return v.view(v.shape[0], -1) @ matrix
