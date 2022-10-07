"""
Provides utilities for the Monte-Carlo Dropout baseline.
"""
from typing import Optional
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn.modules.dropout import _DropoutNd
from tqdm import tqdm
from bayes_dip.dip.network import UNet
from sklearn.neighbors import KernelDensity

class mc_dropout2d(_DropoutNd):
    """Dropout 2D layer"""
    def forward(self, inp):
        """Apply dropout."""
        return F.dropout2d(inp, self.p, True, self.inplace)

class conv2d_dropout(nn.Module):
    """Wrapper for a given layer that appends a subsequent dropout layer."""
    def __init__(self, sub_module, p):
        super().__init__()
        self.layer = sub_module
        self.dropout = mc_dropout2d(p=p)
    def forward(self, x):
        """Apply layer followed by dropout."""
        x = self.layer(x)
        return self.dropout(x)

def bayesianize_unet_architecture(nn_model: UNet, p: float = 0.05) -> None:
    """
    Replace all 3x3 :class:`nn.Conv2d` layers that are contained in an
    :class:`nn.Sequential` module with wrapping :class:`conv2d_dropout` layers.

    Parameters
    ----------
    nn_model : :class:`bayes_dip.dip.network.UNet`
        U-Net model (or other supported model).
    p : float, optional
        Dropout rate. The default is ``0.05``.
    """
    for _, module in nn_model.named_modules():
        if isinstance(module, nn.Sequential):
            for name_sub_module, sub_module in module.named_children():
                if isinstance(sub_module, nn.Conv2d):
                    if sub_module.kernel_size == (3, 3):
                        setattr(module, name_sub_module, conv2d_dropout(sub_module, p))

def sample_from_bayesianized_model(nn_model, filtbackproj, mc_samples, device=None):
    """
    Sample from MCDO model.

    Parameters
    ----------
    nn_model : :class:`bayes_dip.dip.network.UNet`
        Model returned from :func:`bayesianize_unet_architecture`.
    filtbackproj : Tensor
        Filtered back-projection.
    mc_samples : int
        Number of Monte-Carlo samples to draw.
    device : str or torch.device, optional
        Device. If ``None`` (the default), ``filtbackproj.device`` is used.
    """
    sampled_recons = []
    if device is None:
        device = filtbackproj.device
    for _ in tqdm(range(mc_samples), desc='sampling'):
        sampled_recons.append(nn_model.forward(filtbackproj).detach().to(device))
    return torch.cat(sampled_recons, dim=0)

def approx_kernel_density(
        ground_truth: Tensor,
        samples: Tensor,
        bw: float = 0.1,
        noise_x_correction_term: Optional[float] = None
    ):

    assert ground_truth.shape[1:] == samples.shape[1:]

    if noise_x_correction_term is not None:
        samples = samples + torch.randn_like(samples) * noise_x_correction_term **.5
    kde = KernelDensity(
            kernel='gaussian',
            bandwidth=bw
        ).fit(
                samples.view(samples.shape[0], -1).cpu().numpy()
            )
    return kde.score_samples(ground_truth.flatten().cpu().numpy()[None, :])


