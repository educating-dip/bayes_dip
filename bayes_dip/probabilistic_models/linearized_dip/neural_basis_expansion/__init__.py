"""
Provides neural basis expansion class that allows for exact and approximate vjp and jvp closure.
"""

from .base_neural_basis_expansion import BaseNeuralBasisExpansion
from .neural_basis_expansion import (
        NeuralBasisExpansion, MatmulNeuralBasisExpansion, GpriorNeuralBasisExpansion)
from .low_rank_neural_basis_expansion import LowRankNeuralBasisExpansion
from .functorch_utils import (
        flatten_grad_functorch,
        unflatten_nn_functorch
)
