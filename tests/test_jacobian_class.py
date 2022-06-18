from builtins import breakpoint
import torch
import functorch as ftch 
import torch.nn as nn 
from typing import Callable

def flatten_grad_functorch(model, batchnorm_layers, grads, include_biases=False):
    jacs = []
    for (name, param), grad in zip(model.named_parameters(), grads):
        name = name.replace('module.', '')
        if "weight" in name and name not in batchnorm_layers:
            jacs.append(grad.detach().reshape(-1))

        if include_biases and 'bias' in name and name not in batchnorm_layers:
            jacs.append(grad.detach().flatten())

    jacs = torch.cat(jacs, dim=0)
    return jacs

def unflatten_nn_functorch(model, batchnorm_layers, weights, include_biases=False):
    w_pointer = 0
    weight_list = []
    for name, param in model.named_parameters():
        name = name.replace('module.', '')
        if 'weight' in name and name not in batchnorm_layers:
            len_w = param.data.numel()
            weight_list.append(weights[w_pointer:w_pointer +len_w].view(param.shape).float().to(weights.device, non_blocking=True))
            w_pointer += len_w
        if include_biases and 'bias' in name and name not in batchnorm_layers:
            len_w = param.data.numel()
            weight_list.append(weights[w_pointer:w_pointer + 
                                       len_w].view(param.shape).float().to(weights.device, non_blocking=True))
            w_pointer += len_w
        
        # Append zero batchnorm parameters, so that unflattened weights have
        # exactly same shape as func_model params, but ensure that batchnorm
        # parameters don't affect JvPs or vJPs.
        if name in batchnorm_layers or (not include_biases and 'bias' in name and name not in batchnorm_layers):
            len_w = param.data.numel()
            weight_list.append(torch.zeros_like(param).float().to(weights.device, non_blocking=True))

    # functorch.make_functional returns tuple(params)
    return tuple(weight_list)

def list_norm_layers(model):

    """ compute list of names of all GroupNorm (or BatchNorm2d) layers in the model """
    norm_layers = []
    for (name, module) in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, torch.nn.GroupNorm) or isinstance(module,
                torch.nn.BatchNorm2d) or isinstance(module, torch.nn.InstanceNorm2d):
            norm_layers.append(name + '.weight')
            norm_layers.append(name + '.bias')
    return norm_layers

class DummyNetwork(nn.Module):
    def __init__(self) -> None:
        super(DummyNetwork, self).__init__()

        self.layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10)
        )

    def forward(self, x):
        
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class NeuralBasisExpansions:

    def __init__(self, model: nn.Module, NN_input: torch.Tensor, include_biases: bool) -> None:

        """
        Wrapper class for Jacobian vector products and vector Jacobian products.
        This class stores all the statefull information needed for these operations as attributes
        and exposes just the JvP and vJP methods.
        """
        
        self.torch_model = model 
        self.norm_layers = list_norm_layers(self.torch_model)
        self.include_biases = include_biases
        self.NN_input = NN_input
        self._func_model_with_input, self.func_params = ftch.make_functional(self.torch_model)

        self._single_jvp_fun = self._get_single_jvp_fun(return_out=True)
        self._single_vjp_fun = self._get_single_vjp_fun(return_out=False)

        # jvp takes inputs of size (K, 1, D) where K is number of vectors to perform jvp with and D is size of those vectors which should match number of non-normed parameters
        self.jvp = ftch.vmap(self._single_jvp_fun, in_dims=0)
        
        # vjp takes inputs of size (K, 1, O) where K is number of vectors to perform jvp with and O is size of the NN outputs
        self.vjp = ftch.vmap(self._single_vjp_fun, in_dims=(0))


    def _func_model(self, func_params):

        """
        Closure that hardcodes the input "NN_input", leaving only a function of the NN weights.
        Args:
            func_params: functorch wrapped NN weights, exposed as to comply with signature of ftch.jvp
        """
        return self._func_model_with_input(func_params, self.NN_input)
        
    def _get_single_jvp_fun(self, return_out: bool = False) -> Callable:

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
                self.norm_layers,
                v.detach(),
                include_biases=self.include_biases)
            
            single_out, single_jvp = ftch.jvp(
                self._func_model, (self.func_params,), (unflat_v,))

            if return_out:
                return single_out, single_jvp
            else:
                return single_jvp

        return f

    def _get_single_vjp_fun(self, return_out: bool = False) -> Callable:
            
        single_out, vjp_fn = ftch.vjp(self._func_model, self.func_params)

        def f(v):
            # Calculate v.J using vJP
            # v is vector of size N_outputs
            unflat_w_grad = vjp_fn(v)

            single_w_grad = flatten_grad_functorch(
                self.torch_model,
                self.norm_layers,
                unflat_w_grad[0],  # we index 0th element, as vjp return tuple
                include_biases=self.include_biases)  # (D,)

            if return_out:
                return single_out, single_w_grad
            else:
                return single_w_grad

        return f

nn = DummyNetwork()
include_biases = True
n_params = 2140 if not include_biases else 2158
input = torch.randn((1, 1, 28, 28))
neural_basis_expansion = NeuralBasisExpansions(
    model=nn,
    NN_input=input,
    include_biases=include_biases
)
v_out = torch.randn((3, 1, 10))
v_params = torch.randn((3, n_params))

out = neural_basis_expansion.vjp(v_out)
print(out.shape)

_, out = neural_basis_expansion.jvp(v_params)
print(out.shape)