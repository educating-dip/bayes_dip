from typing import Callable, Sequence, Tuple
import torch
from torch import nn
import functorch as ftch

from bayes_dip.utils.utils import CustomAutogradModule
from .base_neural_basis_expansion import BaseNeuralBasisExpansion
from .functorch_utils import unflatten_nn_functorch, flatten_grad_functorch


class NeuralBasisExpansion(BaseNeuralBasisExpansion):

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

        super().__init__(
                nn_model=nn_model, nn_input=nn_input, ordered_nn_params=ordered_nn_params,
                nn_out_shape=nn_out_shape)

        self._func_model_with_input, self.func_params = ftch.make_functional(self.nn_model)

        _single_jvp_fun_with_out = self._get_single_jvp_fun(return_out=True)
        _single_jvp_fun = self._get_single_jvp_fun(return_out=False)
        _single_vjp_fun = self._get_single_vjp_fun(return_out=False)

        # jvp takes inputs of size (K, 1, D) where K is number of vectors to perform jvp with and
        # D is size of those vectors which should match number of non-normed parameters
        self.jvp_with_out = ftch.vmap(_single_jvp_fun_with_out, in_dims=0)
        jvp = ftch.vmap(_single_jvp_fun, in_dims=0)
        # vjp takes inputs of size (K, 1, O) where K is number of vectors to perform jvp with and
        # O is size of the NN outputs
        vjp = ftch.vmap(_single_vjp_fun, in_dims=0)

        self._jvp = CustomAutogradModule(jvp, vjp)
        self._vjp = CustomAutogradModule(vjp, jvp)

    def _func_model(self,
            func_params):
        """
        Closure that hardcodes the input `nn_input`, leaving only a function of the NN weights.

        Parameters
        ----------
        func_params
            functorch wrapped NN weights, exposed as to comply with signature of ``ftch.jvp``
        """
        return self._func_model_with_input(func_params, self.nn_input)

    def _get_single_jvp_fun(self,
                    return_out: bool = False) -> Callable:
        """
        Generate closure that performs ``J_{params}(x)``.

        Parameters
        ----------
        return_out : bool, optional
            If `True`, let the closure return ``(out, jvp)``, i.e., also the output, not just `jvp`.
            The default is `False`.
        """

        def f(v):

            unflat_v = unflatten_nn_functorch(
                self.nn_model,
                self.inds_from_ordered_params,
                self.slices_from_ordered_params,
                v.detach(),
               )

            single_out, single_jvp = ftch.jvp(  # pylint: disable=unbalanced-tuple-unpacking
                self._func_model, (self.func_params,), (unflat_v,))

            return (single_out, single_jvp) if return_out else single_jvp

        return f

    def _get_single_vjp_fun(self,
                    return_out: bool = False) -> Callable:

        single_out, vjp_fn = ftch.vjp(  # pylint: disable=unbalanced-tuple-unpacking
                self._func_model, self.func_params)

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

    def jvp(self, v):
        return self._jvp(v)

    def vjp(self, v):
        return self._vjp(v)
