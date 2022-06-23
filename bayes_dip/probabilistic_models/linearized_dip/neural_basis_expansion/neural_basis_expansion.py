from typing import Callable, Sequence
import torch
from torch import nn
import functorch as ftch
from bayes_dip.utils import list_norm_layers, count_parameters
from .functorch_utils import unflatten_nn_functorch, flatten_grad_functorch

class NeuralBasisExpansion:

    def __init__(self, 
            model: nn.Module, 
            nn_input: torch.Tensor, 
            include_biases: bool, 
            exclude_nn_layers: Sequence = () ) -> None:

        """
        Wrapper class for Jacobian vector products and vector Jacobian products.
        This class stores all the statefull information needed for these operations as attributes
        and exposes just the JvP and vJP methods.
        """
        
        self.torch_model = model 
        self.exclude_layers = list_norm_layers(self.torch_model) + list(exclude_nn_layers)
        self.include_biases = include_biases
        self.nn_input = nn_input
        self._func_model_with_input, self.func_params = ftch.make_functional(self.torch_model)
        self.num_params = count_parameters(self.torch_model, self.exclude_layers, self.include_biases)

        self._single_jvp_fun = self._get_single_jvp_fun(return_out=True)
        self._single_vjp_fun = self._get_single_vjp_fun(return_out=False)

        # jvp takes inputs of size (K, 1, D) where K is number of vectors to perform jvp with and D is size of those vectors which should match number of non-normed parameters
        self.jvp = ftch.vmap(self._single_jvp_fun, in_dims=0)
        
        # vjp takes inputs of size (K, 1, O) where K is number of vectors to perform jvp with and O is size of the NN outputs
        self.vjp = ftch.vmap(self._single_vjp_fun, in_dims=(0))

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
                self.torch_model,
                self.exclude_layers,
                v.detach(),
                include_biases=self.include_biases)
            
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
            unflat_w_grad = vjp_fn(v)

            single_w_grad = flatten_grad_functorch(
                self.torch_model,
                self.exclude_layers,
                unflat_w_grad[0],  # we index 0th element, as vjp return tuple
                include_biases=self.include_biases)  # (D,)

            return (single_out, single_w_grad) if return_out else single_w_grad

        return f
