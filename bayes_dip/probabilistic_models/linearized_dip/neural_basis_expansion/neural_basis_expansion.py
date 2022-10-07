"""
Provides a neural basis expansion implementation using functorch.
"""
from typing import Callable
import functorch as ftch
import torch
from torch import nn, Tensor
from torch import autograd

from bayes_dip.utils.utils import CustomAutogradModule
from .base_neural_basis_expansion import BaseNeuralBasisExpansion, BaseMatmulNeuralBasisExpansion
from .functorch_utils import unflatten_nn_functorch, flatten_grad_functorch
from ..utils import get_inds_from_ordered_params


class NeuralBasisExpansion(BaseNeuralBasisExpansion):
    """
    Implementation of Jacobian vector products (:meth:`jvp`) and vector Jacobian products
    (:meth:`vjp`) via functorch. Both methods support autograd.
    """

    def __init__(self, *args, functional_forward_kwargs=None, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.func_model_with_input, self.func_params = ftch.make_functional(self.nn_model)
        self.functional_forward_kwargs = functional_forward_kwargs or {}

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

    def _func_model(self, func_params) -> Callable:
        """
        Closure that hardcodes the input ``nn_input``, leaving only a function of the NN weights.

        Parameters
        ----------
        func_params
            functorch wrapped NN weights, exposed as to comply with signature of ``ftch.jvp``
        """
        return self.func_model_with_input(
                func_params, self.nn_input, **self.functional_forward_kwargs)

    def _get_single_jvp_fun(self,
                    return_out: bool = False) -> Callable:
        """
        Generate closure that performs ``J_{params}(x)``.

        Parameters
        ----------
        return_out : bool, optional
            If ``True``, let the closure return ``(out, jvp)``, i.e., also the output, not just
            ``jvp``.
            The default is ``False``.
        """

        def f(v: Tensor):

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

    def jvp(self, v: Tensor) -> Tensor:
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
        return self._jvp(v)

    def vjp(self, v: Tensor) -> Tensor:
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
        return self._vjp(v)

class MatmulNeuralBasisExpansion(BaseMatmulNeuralBasisExpansion):
    """
    Implementation of Jacobian vector products (:meth:`jvp`) and vector Jacobian products
    (:meth:`vjp`) using matmul with the Jacobian matrix assembled via
    :func:`autograd.functional.jacobian`.
    """

    def __init__(self, nn_model: nn.Module, *args, **kwargs) -> None:

        self.func_model_with_input, _ = ftch.make_functional(nn_model)
        super().__init__(nn_model, *args, **kwargs)
        self.update_matrix()

    @property
    def matrix(self):
        return self._matrix

    def update_matrix(self) -> None:
        self._matrix = self.get_matrix()

    def get_matrix(self) -> Tensor:

        inds_from_ordered_params = get_inds_from_ordered_params(
            self.nn_model,
            self.ordered_nn_params
        )
        func_params = list(self.nn_model.parameters())
        def _func_model(*func_params_under_prior):
            for i, func_param in zip(
                    inds_from_ordered_params, func_params_under_prior):
                func_params[i] = func_param
            return self.func_model_with_input(func_params, self.nn_input)

        matrix = autograd.functional.jacobian(
            _func_model, tuple(self.ordered_nn_params)
            )
        matrix = torch.cat(
            [jac_i.view(self.nn_input.numel(), -1) for jac_i in matrix], dim=1
            )

        return matrix
