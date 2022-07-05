"""
Provides utilities for the Monte-Carlo Dropout baseline.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.dropout import _DropoutNd
from tqdm import tqdm
from .network import UNet

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

def bayesianize_unet_architecture(model: UNet, p: float = 0.05) -> None:
    """
    Replace all 3x3 :class:`nn.Conv2d` layers that are contained in an
    :class:`nn.Sequential` module with wrapping :class:`conv2d_dropout` layers.

    Parameters
    ----------
    model : :class:`bayes_dip.dip.network.UNet`
        U-Net model (or other supported model).
    p : float, optional
        Dropout rate. The default is ``0.05``.
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            for name_sub_module, sub_module in module.named_children():
                if isinstance(sub_module, nn.Conv2d):
                    if sub_module.kernel_size == (3, 3):
                        setattr(module, name_sub_module, conv2d_dropout(sub_module, p))

def sample_from_bayesianized_model(model, filtbackproj, mc_samples, device=None):
    """
    Sample from MCDO model.

    Parameters
    ----------
    model : :class:`bayes_dip.dip.network.UNet`
        Model returned from :func:`bayesianize_unet_architecture`.
    filtbackproj : Tensor
        Filtered back-projection.
    mc_samples : int
        Number of Monte-Carlo samples to draw.
    device : str or torch.device, optional
        Device. If `None` (the default), ``filtbackproj.device`` is used.
    """
    sampled_recons = []
    if device is None:
        device = filtbackproj.device
    for _ in tqdm(range(mc_samples), desc='sampling'):
        sampled_recons.append(model.forward(filtbackproj).detach().to(device))
    return torch.cat(sampled_recons, dim=0)
