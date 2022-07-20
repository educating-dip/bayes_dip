import torch
from torch import Tensor 

def tv_loss(x: Tensor):
    """
    Isotropic TV loss.
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    # note that this differs from Baguer et al., who used
    # torch.sum(dh[..., :-1, :] + dw[..., :, :-1]) instead
    return torch.sum(dh) + torch.sum(dw)

def batch_tv_grad(x: Tensor):

    assert x.shape[-1] == x.shape[-2]
    batch_size = x.shape[0]
    sign_diff_x = torch.sign(torch.diff(-x, n=1, dim=-1))
    pad = torch.zeros((batch_size, 1, x.shape[-2], 1), device = x.device)
    diff_x_pad = torch.cat([pad, sign_diff_x, pad], dim=-1)
    grad_tv_x = torch.diff(diff_x_pad, n=1, dim=-1)
    sign_diff_y = torch.sign(torch.diff(-x, n=1, dim=-2))
    pad = torch.zeros((batch_size, 1, 1, x.shape[-1]), device = x.device)
    diff_y_pad = torch.cat([pad, sign_diff_y, pad], dim=-2)
    grad_tv_y = torch.diff(diff_y_pad, n=1, dim=-2)
    
    return grad_tv_x + grad_tv_y