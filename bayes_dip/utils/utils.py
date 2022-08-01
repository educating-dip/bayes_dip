from typing import List, Optional, Callable
import os
from functools import reduce
import numpy as np
from skimage.metrics import structural_similarity
import torch
from torch import nn
from torch import Tensor
from .linear_cg_gpytorch import linear_cg
try:
    import hydra.utils
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False


def get_original_cwd():
    cwd = None
    if HYDRA_AVAILABLE:
        try:
            cwd = hydra.utils.get_original_cwd()
        except ValueError:  # raised if hydra is not initialized
            pass
    if cwd is None:
        cwd = os.getcwd()
    return cwd

def list_norm_layer_params(nn_model):

    """ compute list of names of all GroupNorm (or BatchNorm2d) layers in the model """
    norm_layer_params = []
    for (name, module) in nn_model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module,
                (torch.nn.GroupNorm, torch.nn.BatchNorm2d,
                torch.nn.InstanceNorm2d)):
            norm_layer_params.append(name + '.weight')
            norm_layer_params.append(name + '.bias')
    return norm_layer_params

def get_params_from_nn_module(nn_model, exclude_norm_layers=True, include_bias=False):

    norm_layer_params = []
    if exclude_norm_layers:
        norm_layer_params = list_norm_layer_params(nn_model)

    params = []
    for (name, param) in nn_model.named_parameters():
        if name not in norm_layer_params:
            if name.endswith('.weight') or (name.endswith('.bias') and include_bias):
                params.append(param)

    return params

def get_modules_by_names(
        nn_model: nn.Module,
        layer_names: List[str]
        ) -> List[nn.Module]:
    layers = [
        reduce(getattr, layer_name.split(sep='.'), nn_model)
        for layer_name in layer_names]
    return layers

def PSNR(reconstruction, ground_truth, data_range=None):
    gt = np.asarray(ground_truth)
    mse = np.mean((np.asarray(reconstruction) - gt)**2)
    if mse == 0.:
        return float('inf')
    if data_range is None:
        data_range = np.max(gt) - np.min(gt)
    return 20*np.log10(data_range) - 10*np.log10(mse)

def SSIM(reconstruction, ground_truth, data_range=None):
    gt = np.asarray(ground_truth)
    if data_range is None:
        data_range = np.max(gt) - np.min(gt)
    return structural_similarity(reconstruction, gt, data_range=data_range)

def normalize(x, inplace=False):
    if inplace:
        x -= x.min()
        x /= x.max()
    else:
        x = x - x.min()
        x = x / x.max()
    return x

def cg(
        closure: Callable,
        v: Tensor,
        precon_closure: Optional[Callable] = None,
        max_niter: int = 10,
        rtol: float = 1e-6,
        ignore_numerical_warning: bool = False,
        ) -> Tensor:

    if ignore_numerical_warning:
        raise NotImplementedError

    v_norm = torch.norm(v, 2, dim=0, keepdim=True)
    v_scaled = v.div(v_norm)

    scaled_solve, residual_norm = linear_cg(closure, v_scaled, n_tridiag=0, tolerance=rtol,
                eps=1e-10, stop_updating_after=1e-10, max_iter=max_niter,
                max_tridiag_iter=max_niter-1, preconditioner=precon_closure,
            )

    solve = scaled_solve * v_norm
    return solve, residual_norm

# adapted from python 3.10's bisect module
def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
    return lo


class eval_mode:
    def __init__(self, nn_model):
        self.nn_model = nn_model
        self.training = None  # will be set in __enter__

    def __enter__(self):
        self.training = self.nn_model.training
        self.nn_model.eval()

    def __exit__(self, *exc):
        self.nn_model.train(self.training)


class CustomAutogradFunction(torch.autograd.Function):
    # pylint: disable=abstract-method

    @staticmethod
    def forward(ctx, x, forward_fun, backward_fun):  # pylint: disable=arguments-differ
        ctx.backward_fun = backward_fun
        y = forward_fun(x)
        return y

    @staticmethod
    def backward(ctx, y):  # pylint: disable=arguments-differ
        x = ctx.backward_fun(y)
        return x, None, None


class CustomAutogradModule(nn.Module):
    def __init__(self, forward_fun, backward_fun):
        super().__init__()
        self.forward_fun = forward_fun
        self.backward_fun = backward_fun

    def forward(self, x):
        return CustomAutogradFunction.apply(x, self.forward_fun, self.backward_fun)
