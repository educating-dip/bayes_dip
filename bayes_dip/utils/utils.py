import os
import numpy as np
from skimage.metrics import structural_similarity
import torch
from torch import nn
from functools import reduce
from typing import List
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

def list_norm_layer_params(model):

    """ compute list of names of all GroupNorm (or BatchNorm2d) layers in the model """
    norm_layer_params = []
    for (name, module) in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module,
                (torch.nn.GroupNorm, torch.nn.BatchNorm2d,
                torch.nn.InstanceNorm2d)):
            norm_layer_params.append(name + '.weight')
            norm_layer_params.append(name + '.bias')
    return norm_layer_params

def get_params_from_nn_module(model, exclude_norm_layers=True, include_bias=False):

    norm_layer_params = []
    if exclude_norm_layers:
        norm_layer_params = list_norm_layer_params(model)

    params = []
    for (name, param) in model.named_parameters():
        if name not in norm_layer_params:
            if name.endswith('.weight') or (name.endswith('.bias') and include_bias):
                params.append(param)

    return params

def get_modules_by_names(
        model: nn.Module, 
        layer_names: List[str]
        ) -> List[nn.Module]:
    layers = [
        reduce(getattr, layer_name.split(sep='.'), model)
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
