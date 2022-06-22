import torch

def tv_loss(x):
    """
    Isotropic TV loss.
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh) + torch.sum(dw)  # note that this differs from Baguer et al., who used torch.sum(dh[..., :-1, :] + dw[..., :, :-1])
