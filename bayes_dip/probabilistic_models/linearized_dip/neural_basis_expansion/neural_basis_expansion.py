from warnings import warn
from typing import Callable, Optional
import functorch as ftch
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor
import torch.autograd as autograd

from bayes_dip.utils.utils import CustomAutogradModule
from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from .base_neural_basis_expansion import BaseNeuralBasisExpansion
from .functorch_utils import unflatten_nn_functorch, flatten_grad_functorch
from ..utils import get_inds_from_ordered_params


class NeuralBasisExpansion(BaseNeuralBasisExpansion):
    """
    Implementation of Jacobian vector products (:meth:`jvp`) and vector Jacobian products
    (:meth:`vjp`) via functorch. Both methods support autograd.
    """

    def __init__(self, *args, functional_forward_kwargs=None, **kwargs) -> None:
        """
        Parameters are the same as for :class:`BaseNeuralBasisExpansion`.
        """

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
        Closure that hardcodes the input `nn_input`, leaving only a function of the NN weights.

        Parameters
        ----------
        func_params
            functorch wrapped NN weights, exposed as to comply with signature of ``ftch.jvp``
        """
        return self.func_model_with_input(func_params, self.nn_input, **self.functional_forward_kwargs)

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

class MatmulNeuralBasisExpansion(BaseNeuralBasisExpansion):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.func_model_with_input, _ = ftch.make_functional(self.nn_model)
        self.matrix = self.get_matrix()

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

        v_transpose = v.T
        jvp = (self.matrix @ v_transpose).T
        return jvp.view(
            v.shape[0], *self.nn_out_shape)

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
        return v.view(v.shape[0], -1) @ self.matrix

    def get_matrix(self) -> Tensor:

        inds_from_ordered_params = get_inds_from_ordered_params(
            self.nn_model,
            self.ordered_nn_params
        )
        func_params = [param for param in self.nn_model.parameters()]
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

    def update_matrix(self) -> None:
        self.matrix = self.get_matrix()

class GpriorNeuralBasisExpansion(NeuralBasisExpansion):
    def __init__(self, 
            trafo: BaseRayTrafo,
            scale_kwargs, 
            *args,
            **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)

        self.trafo = trafo
        self.neural_basis_expansion = NeuralBasisExpansion(
            *args, **kwargs,
        )
        self.scale = self.compute_scale(**scale_kwargs)
    
    def update_scale(self, **scale_kwargs): 
        self.scale = self.compute_scale(**scale_kwargs)

    def compute_scale(self,
            reduction: str = 'mean',
            batch_size: int = 1,
            eps: float = 1e-6,
            max_scale_thresh: float = 1e5,
            verbose: bool = True,
        ) -> Tensor:
            
            def forward(v):
                return self.neural_basis_expansion.vjp(
                    self.trafo.trafo_adjoint(v).unsqueeze(dim=1)
                ).pow(2) 
                
            obs_numel = np.prod(self.trafo.obs_shape)
            v = torch.empty((batch_size, 1, *self.trafo.obs_shape), device=self.trafo.matrix.device)
            rows = torch.zeros((self.num_params), device=self.trafo.matrix.device)
            with torch.no_grad():
                for i in tqdm(np.array(range(0, obs_numel, batch_size)),
                            desc='compute_scale', miniters=obs_numel//batch_size//100
                        ):
                    v[:] = 0.
                    # set v.view(batch_size, -1) to be a subset of rows of torch.eye(obs_numel);
                    # in last batch, it may contain some additional (zero) rows
                    v.view(batch_size, -1)[:, i:i+batch_size].fill_diagonal_(1.)
                    rows_batch = forward(
                        v,
                    )
                    rows_batch = rows_batch.view(batch_size, -1)
                    if i+batch_size > obs_numel:  # last batch
                        rows_batch = rows_batch[:obs_numel%batch_size]
                    rows += rows_batch.sum(dim=0)
                if verbose:
                    print(f'scale.min: {rows.min()}, scale.max: {rows.max()}, ' 
                        f'scale.num_vals_below_{eps}:{(rows < eps).sum()}\n'
                    )
                if rows.max() > max_scale_thresh: 
                    warn('max scale values reached.')
                scale_vec = (rows.clamp_(min=eps) / obs_numel).pow(0.5) if reduction == 'mean' \
                     else rows.clamp_(min=eps).pow(0.5) # num_obs, num_params

            return scale_vec

    def _get_single_jvp_fun(self,
                    return_out: bool = False) -> Callable:
        """
        Generate closure that performs ``J_{params*scale}(x)``.
        """

        def f(v: Tensor):
            scaled_v = v*self.scale
            scaled_unflat_v = unflatten_nn_functorch(
                self.nn_model,
                self.inds_from_ordered_params,
                self.slices_from_ordered_params,
                scaled_v.detach()
               )

            single_out, single_jvp = ftch.jvp(  # pylint: disable=unbalanced-tuple-unpacking
                self._func_model, (self.func_params,), (scaled_unflat_v,))

            return (single_out, single_jvp) if return_out else single_jvp

        return f

    def _get_single_vjp_fun(self,
                    return_out: bool = False) -> Callable:

        single_out, vjp_fn = ftch.vjp(  # pylint: disable=unbalanced-tuple-unpacking
                self._func_model, self.func_params)

        def f(v):
            # Calculate v.J*s using vJp
            # v is vector of size N_outputs
            unflat_w_grad = vjp_fn(v)[0] # we index 0th element, as vjp return tuple

            single_w_grad = flatten_grad_functorch(
                self.inds_from_ordered_params,
                unflat_w_grad,
               )  # (D,)

            return (single_out, single_w_grad*self.scale) if return_out else single_w_grad*self.scale

        return f