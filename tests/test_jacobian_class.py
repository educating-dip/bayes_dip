import torch
import torch.nn as nn

from bayes_dip.probabilistic_models import NeuralBasisExpansion, ApproxNeuralBasisExpansion

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

nn = DummyNetwork(device)
include_biases = True
n_params = 2140 if not include_biases else 2158
input = torch.randn((1, 1, 28, 28), device=device)
neural_basis_expansion = NeuralBasisExpansion(
    model=nn,
    nn_input=input,
    include_biases=include_biases
)
v_out = torch.randn((3, 10), device=device)
v_params = torch.randn((3, n_params), device=device)

out = neural_basis_expansion.vjp(v_out)
print(out.shape)

_, out = neural_basis_expansion.jvp(v_params)
print(out.shape)

approx_neural_basis_expansion = ApproxNeuralBasisExpansion(
    model=nn,
    nn_input=input,
    include_biases=include_biases, 
    vec_batch_size=1,
    oversampling_param=5, 
    low_rank_rank_dim=5,
    device=device,
    use_cpu=True)

print(approx_neural_basis_expansion.vjp_approx(v_out).shape)
print(approx_neural_basis_expansion.jvp_approx(v_params).shape)
