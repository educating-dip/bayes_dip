import os
import numpy as np
from skimage.metrics import structural_similarity
import torch
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