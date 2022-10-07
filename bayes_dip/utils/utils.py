"""
General utilities.
"""

from __future__ import annotations  # postponed evaluation, to make ArrayLike look good in docs
from typing import List, Tuple, Optional, Callable, Union, Any
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any
import os
from functools import reduce
import numpy as np
from skimage.metrics import structural_similarity
import torch
from torch import nn
from torch import Tensor
from .linear_cg_gpytorch import linear_cg
from .linear_cg_gpytorch_log_cg_re import linear_log_cg_re
try:
    import hydra.utils
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False


def get_original_cwd() -> str:
    """
    Return the original current working directory. This is a wrapper for
    :func:`hydra.utils.get_original_cwd()` that also works if not using hydra.
    """
    cwd = None
    if HYDRA_AVAILABLE:
        try:
            cwd = hydra.utils.get_original_cwd()
        except ValueError:  # raised if hydra is not initialized
            pass
    if cwd is None:
        cwd = os.getcwd()
    return cwd

def list_norm_layer_params(nn_model: nn.Module) -> List[str]:
    """
    Return a list of names of all parameters from ``GroupNorm``, ``BatchNorm2d`` and
    ``InstanceNorm2d`` layers in a model.
    """
    norm_layer_params = []
    for (name, module) in nn_model.named_modules():
        if isinstance(module,
                (torch.nn.GroupNorm, torch.nn.BatchNorm2d,
                torch.nn.InstanceNorm2d)):
            norm_layer_params.extend(f'{name}.{param_name}'
                    for param_name, _ in module.named_parameters())
    return norm_layer_params

def get_params_from_nn_module(
        nn_model: nn.Module, exclude_norm_layers: bool = True, exclude_bias: bool = True
        ) -> List[str]:
    """
    Return names of parameters in a model, optionally excluding norm layers' and/or bias parameters.
    """
    exclude_params = []
    if exclude_norm_layers:
        exclude_params = list_norm_layer_params(nn_model)

    params = []
    for (name, param) in nn_model.named_parameters():
        if name not in exclude_params and not (exclude_bias and name.endswith('.bias')):
            params.append(param)

    return params

def get_modules_by_names(
        nn_model: nn.Module,
        layer_names: List[str]
        ) -> List[nn.Module]:
    """
    Return a list of modules by a list of names in a model.
    """
    layers = [
        reduce(getattr, layer_name.split(sep='.'), nn_model)
        for layer_name in layer_names]
    return layers

def PSNR(
        reconstruction: ArrayLike,
        ground_truth: ArrayLike,
        data_range: Optional[float] = None) -> np.number:
    """
    Return the peak signal-to-noise ratio (PSNR) of a reconstruction given the ground truth.

    If ``data_range is None``, it is computed as ``max(ground_truth) - min(ground_truth)``.
    """
    gt = np.asarray(ground_truth)
    mse = np.mean((np.asarray(reconstruction) - gt)**2)
    if mse == 0.:
        return float('inf')
    if data_range is None:
        data_range = np.max(gt) - np.min(gt)
    return 20*np.log10(data_range) - 10*np.log10(mse)

def SSIM(
        reconstruction: ArrayLike,
        ground_truth: ArrayLike,
        data_range: Optional[float] = None) -> np.number:
    """
    Return the structural similarity (SSIM) [1]_ of a reconstruction given the ground truth.

    If ``data_range is None``, it is computed as ``max(ground_truth) - min(ground_truth)``.

    .. [1] Z. Wang, A.C. Bovik, H.R. Sheikh, E.P. Simoncelli, 2004, "Image quality assessment: from
           error visibility to structural similarity". IEEE Transactions on Image Processing.
           https://doi.org/10.1109/TIP.2003.819861
    """
    gt = np.asarray(ground_truth)
    if data_range is None:
        data_range = np.max(gt) - np.min(gt)
    return structural_similarity(reconstruction, gt, data_range=data_range)

def normalize(x: Union[Tensor, np.ndarray], inplace: bool = False) -> Union[Tensor, np.ndarray]:
    """
    Normalize the input as ``(x - x.min()) / (x.max() - x.min())``, optionally inplace.
    """
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
        use_log_re_variant: bool = False,
        ignore_numerical_warning: bool = False,
        ) -> Tuple[Tensor, Tensor]:
    """
    Solve a linear system approximately with the conjugate gradients method (CG).

    This function wraps :func:`bayes_dip.utils.linear_cg_gpytorch.linear_cg`, which itself is a
    clone of :func:`gpytorch.utils.linear_cg.linear_cg` also returning the residual.

    Parameters
    ----------
    closure : callable
        Matmul closure, see first argument to :func:`linear_cg`.
    v : Tensor
        Right hand side, see second argument to :func:`linear_cg`.
    precon_closure : callable, optional
        Left-preconditioning closure, see ``preconditioner`` argument to :func:`linear_cg`.
        The default is ``None``.
    max_niter : int, optional
        Maximum number of iterations, see ``max_iter`` argument to :func:`linear_cg`.
        The default is ``10``.
    rtol : float, optional
        Tolerance at which to stop early (before ``max_niter``), see ``tolerance`` argument to
        :func:`linear_cg`. The default is ``1e-6``.
    use_log_re_variant : bool, optional
        Whether to use the low precision arithmetic variant by Maddox et al.,
        :meth:`linear_log_cg_re`. The default is ``False``.
    ignore_numerical_warning : bool, optional
        Not implemented yet. Should control whether numerical warnings are ignored.
        The default is ``False``.

    Returns
    -------
    solve : Tensor
        Approximate solution.
    residual_norm : Tensor
        Residual norm of the solution.
    """

    if ignore_numerical_warning:
        raise NotImplementedError

    cg_func = linear_cg if not use_log_re_variant else linear_log_cg_re
    max_niter = min(max_niter, v.shape[0])

    # pylint: disable=unbalanced-tuple-unpacking
    solve, residual_norm = cg_func(closure, v, tolerance=rtol,
                eps=1e-10, stop_updating_after=1e-10, max_iter=max_niter,
                max_tridiag_iter=max_niter-1, preconditioner=precon_closure,
            )

    return solve, residual_norm

def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Adapted from python 3.10's bisect module.
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


def assert_positive_diag(mat : Tensor) -> None:
    """
    Assert that the diagonal of a matrix tensor has strictly positive values only.

    Parameters
    ----------
    mat : Tensor
        Square 2D tensor.
    """
    assert mat.diag().min() > 0


def make_choleskable(
        mat: Tensor, step: float = 1e-6, max_nsteps: int = 1000, verbose: bool = True) -> Tensor:
    """
    Make a matrix tensor Cholesky decomposable by adding small values to the diagonal if needed.

    Parameters
    ----------
    mat : Tensor
        Square 2D tensor. Modified in-place.
    step : float, optional
        Step to add to the diagonal in each try.
        The default is ``1e-6``.
    max_nsteps : int, optional
        Maximum number of steps. This limits the value to be added to ``max_nsteps * step``.
        The default is ``1000``.
    verbose : bool, optional
        If ``True``, print the value that was added to the diagonal if anything was added.
        The default is ``True``.

    Returns
    -------
    chol : Tensor
        The Cholesky factor of the successfully decomposed matrix, i.e.
        ``chol = torch.linalg.cholesky(mat)`` after ``mat`` being in-place modified by this
        function.
    """
    succeed = False
    cnt = 0
    while not succeed:
        try:
            chol = torch.linalg.cholesky(mat)
            succeed = True
        except RuntimeError:
            assert cnt < max_nsteps
            mat[np.diag_indices(mat.shape[0])] += step
            cnt += 1
    if verbose and cnt != 0:
        print(f'amount added to make choleskable: {cnt*step}')
    return chol


class eval_mode:
    """Context manager calling ``nn_model.eval()`` when entering and resetting on exit."""
    def __init__(self, nn_model: nn.Module):
        self.nn_model = nn_model
        self.training = None  # will be set in __enter__

    def __enter__(self):
        self.training = self.nn_model.training
        self.nn_model.eval()

    def __exit__(self, *exc):
        self.nn_model.train(self.training)


class CustomAutogradFunction(torch.autograd.Function):
    """
    Custom autograd function defined by callables ``forward_fun`` and ``backward_fun``.
    """
    # pylint: disable=abstract-method

    @staticmethod
    def forward(
            ctx: Any,
            x: Any,
            forward_fun: Callable,
            backward_fun: Callable
            # pylint: disable=arguments-differ
            ) -> Any:
        ctx.backward_fun = backward_fun
        y = forward_fun(x.detach()).detach()
        return y

    @staticmethod
    def backward(
            ctx: Any,
            y: Any
            # pylint: disable=arguments-differ
            ) -> Any:
        x = ctx.backward_fun(y.detach()).detach()
        return x, None, None


class CustomAutogradModule(nn.Module):
    """
    Custom autograd module defined by callables ``forward_fun`` and ``backward_fun``.
    """
    def __init__(self, forward_fun: Callable, backward_fun: Callable):
        """
        Parameters
        ----------
        forward_fun : callable
            Forward function.
        backward_fun : callable
            Backward function (computing the vector-Jacobian product).
        """
        super().__init__()
        self.forward_fun = forward_fun
        self.backward_fun = backward_fun

    def forward(self, x: Any) -> Any:
        """Apply ``forward_fun``, saving ``backward_fun`` for the backward pass."""
        return CustomAutogradFunction.apply(x, self.forward_fun, self.backward_fun)
