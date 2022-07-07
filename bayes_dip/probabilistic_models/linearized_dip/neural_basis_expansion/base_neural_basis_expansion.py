from typing import Sequence, Tuple
from abc import ABC, abstractmethod
import torch
from torch import nn
import numpy as np

from bayes_dip.utils.utils import eval_mode
from ..utils import get_inds_from_ordered_params, get_slices_from_ordered_params


class BaseNeuralBasisExpansion(ABC):

    # pylint: disable=too-few-public-methods
    # self.jvp and self.vjp act as main public "methods"

    def __init__(self,
            nn_model: nn.Module,
            nn_input: torch.Tensor,
            ordered_nn_params: Sequence,
            nn_out_shape: Tuple[int, int] = None,
            ) -> None:

        """
        Wrapper class for Jacobian vector products and vector Jacobian products.
        This class stores all the statefull information needed for these operations as attributes
        and exposes just the JvP and vJP methods.
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
        self.num_params = sum([param.data.numel() for param in self.ordered_nn_params])

        self.nn_out_shape = nn_out_shape
        if self.nn_out_shape is None:
            with torch.no_grad(), eval_mode(self.nn_model):
                self.nn_out_shape = self.nn_model(nn_input).shape

    @property
    def jac_shape(self) -> Tuple[int, int]:
        return (np.prod(self.nn_out_shape), self.num_params)

    @abstractmethod
    def jvp(self, v):
        raise NotImplementedError

    @abstractmethod
    def vjp(self, v):
        raise NotImplementedError
