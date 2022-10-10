import torch
import torch.nn as nn
import numpy as np
from bayes_dip.utils import get_params_from_nn_module
from bayes_dip.dip import UNet
from bayes_dip.probabilistic_models import NeuralBasisExpansion, LowRankNeuralBasisExpansion

class DummyNetwork(nn.Module):
    def __init__(self, device) -> None:
        super(DummyNetwork, self).__init__()

        self.layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ).to(device)

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10)
        ).to(device)

    def forward(self, x):

        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x.squeeze()

device = torch.device('cuda')
# nn_model = DummyNetwork(device)

nn_model = UNet(
    in_ch=1,
    out_ch=1,
    channels=[32, 32, 32],
    skip_channels=[0, 0, 1],
    use_sigmoid=True,
    use_norm=True,
    sigmoid_saturation_thresh=9
    ).to(device)

exclude_bias = True
# n_params = 2140 if exclude_bias else 2158
# n_params = np.sum([param.data.numel() for param in nn_model.parameters()])

nn_input = torch.randn((1, 1, 28, 28), device=device)
ordered_nn_params = get_params_from_nn_module(nn_model, exclude_bias=exclude_bias)
neural_basis_expansion = NeuralBasisExpansion(
    nn_model=nn_model,
    nn_input=nn_input,
    ordered_nn_params=ordered_nn_params,
    nn_out_shape=nn_input.shape,  # optional
)

print('jac shape', neural_basis_expansion.jac_shape)

# v_out = torch.randn((3, 10), device=device)
v_out = torch.randn((3, 1, 1, 28, 28), device=device)
v_params = torch.randn((3, neural_basis_expansion.num_params), device=device)
v_out.requires_grad_(True)
out = neural_basis_expansion.vjp(v_out)
print(out.shape)
print(out.requires_grad)
out.sum().backward()
assert torch.allclose(v_out.grad, neural_basis_expansion.jvp(torch.ones_like(out)))

v_params.requires_grad_(True)
# _, out = neural_basis_expansion.jvp_with_out(v_params)
out = neural_basis_expansion.jvp(v_params)
print(out.shape)
print(out.requires_grad)
out.sum().backward()
assert torch.allclose(v_params.grad, neural_basis_expansion.vjp(torch.ones_like(out)), atol=1e-5)

low_rank_neural_basis_expansion = LowRankNeuralBasisExpansion(
    neural_basis_expansion=neural_basis_expansion,
    batch_size=1,
    low_rank_rank_dim=5,
    oversampling_param=5,
    device=device,
    use_cpu=True)

print(low_rank_neural_basis_expansion.vjp(v_out).shape)
print(low_rank_neural_basis_expansion.jvp(v_params).shape)
