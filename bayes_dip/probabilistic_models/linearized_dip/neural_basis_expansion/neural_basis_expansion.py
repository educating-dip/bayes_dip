from typing import Callable, Sequence, Tuple
import torch
from torch import nn
import numpy as np
import functorch as ftch

from bayes_dip.utils.utils import eval_mode
from .functorch_utils import unflatten_nn_functorch, flatten_grad_functorch
from ..utils import get_inds_from_ordered_params, get_slices_from_ordered_params

class NeuralBasisExpansion:

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
        self._func_model_with_input, self.func_params = ftch.make_functional(self.nn_model)

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

        self._single_jvp_fun = self._get_single_jvp_fun(return_out=True)
        self._single_vjp_fun = self._get_single_vjp_fun(return_out=False)

        # jvp takes inputs of size (K, 1, D) where K is number of vectors to perform jvp with and D is size of those vectors which should match number of non-normed parameters
        self.jvp = ftch.vmap(self._single_jvp_fun, in_dims=0)

        # vjp takes inputs of size (K, 1, O) where K is number of vectors to perform jvp with and O is size of the NN outputs
        self.vjp = ftch.vmap(self._single_vjp_fun, in_dims=(0))

    @property
    def jac_shape(self) -> Tuple[int, int]:
        return (np.prod(self.nn_out_shape), self.num_params)

    def _func_model(self,
            func_params):

        """
        Closure that hardcodes the input "nn_input", leaving only a function of the NN weights.
        Args:
            func_params: functorch wrapped NN weights, exposed as to comply with signature of ftch.jvp
        """
        return self._func_model_with_input(func_params, self.nn_input)

    def _get_single_jvp_fun(self,
                    return_out: bool = False) -> Callable:

        """
            Generate closure that performs J_{params}(x) .

        Args:
            laplace_model: instance of Laplace class from which to extract metadata.
            params: weights at which to evaluate Jacobian, in functorch format.
            params_model: NN model in wrapped in functorch func_model.
            include_biases: whether to give a Bayesian treatment to model biases."""

        def f(v):

            unflat_v = unflatten_nn_functorch(
                self.nn_model,
                self.inds_from_ordered_params,
                self.slices_from_ordered_params,
                v.detach(),
               )

            single_out, single_jvp = ftch.jvp(
                self._func_model, (self.func_params,), (unflat_v,))

            return (single_out, single_jvp) if return_out else single_jvp

        return f

    def _get_single_vjp_fun(self,
                    return_out: bool = False) -> Callable:

        single_out, vjp_fn = ftch.vjp(self._func_model, self.func_params)

        def f(v):
            # Calculate v.J using vJP
            # v is vector of size N_outputs
            unflat_w_grad = vjp_fn(v)[0] # we index 0th element, as vjp return tuple

            single_w_grad = flatten_grad_functorch(
                self.inds_from_ordered_params,
                unflat_w_grad,
               )  # (D,)

            return (single_out, single_w_grad) if return_out else single_w_grad

        return f
