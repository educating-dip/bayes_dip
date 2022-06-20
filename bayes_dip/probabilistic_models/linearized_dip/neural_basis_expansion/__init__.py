"""
Provides neural basis expansion class that allows for exact and approximate vjp and jvp closure.
"""

from .neural_basis_expansion import NeuralBasisExpansion
from .approx_neural_basis import ApproxNeuralBasisExpansion 
from .functorch_utils import (
        flatten_grad_functorch, 
        unflatten_nn_functorch
    )