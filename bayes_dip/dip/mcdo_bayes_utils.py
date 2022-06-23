import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.dropout import _DropoutNd
from tqdm import tqdm

class mc_dropout2d(_DropoutNd):
    def forward(self, inp):
        return F.dropout2d(inp, self.p, True, self.inplace)

class conv2d_dropout(nn.Module):
    def __init__(self, sub_module, p):
        super().__init__()
        self.layer = sub_module
        self.dropout = mc_dropout2d(p=p)
    def forward(self, x): 
        x = self.layer(x)
        return self.dropout(x)

def bayesianize_architecture(model, p=0.05):
    for _, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            for name_sub_module, sub_module in module.named_children(): 
                if isinstance(sub_module, nn.Conv2d):
                    if sub_module.kernel_size == (3, 3):
                        setattr(module, name_sub_module, conv2d_dropout(sub_module, p))

def sample_from_bayesianized_model(model, filtbackproj, mc_samples, device=None):
    sampled_recons = []
    if device is None: 
        device = filtbackproj.device
    for _  in tqdm(range(mc_samples), desc='sampling'):
        sampled_recons.append(model.forward(filtbackproj)[0].detach().to(device))
    return torch.cat(sampled_recons, dim=0)