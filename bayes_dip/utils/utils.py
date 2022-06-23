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


def list_norm_layers(model):

    """ compute list of names of all GroupNorm (or BatchNorm2d) layers in the model """
    norm_layers = []
    for (name, module) in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module,
                (torch.nn.GroupNorm, torch.nn.BatchNorm2d,
                torch.nn.InstanceNorm2d)):
            norm_layers.append(name + '.weight')
            norm_layers.append(name + '.bias')
    return norm_layers


def count_parameters(model, norm_layers, include_biases):
    len_w = 0
    for name, param in model.named_parameters():
        name = name.replace('module.', '')
        if 'weight' in name and name not in norm_layers:
            len_w += param.data.numel()
        if include_biases and 'bias' in name and name not in norm_layers:
            len_w += param.data.numel()
    return len_w


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