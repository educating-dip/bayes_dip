"""
Provides neural basis expansion class that allows for exact and approximate vjp and jvp closure.
"""

from .base_neural_basis_expansion import BaseNeuralBasisExpansion, BaseMatmulNeuralBasisExpansion
from .neural_basis_expansion import NeuralBasisExpansion, MatmulNeuralBasisExpansion
from .gprior_neural_basis_expansion import (
        GpriorNeuralBasisExpansion, MatmulGpriorNeuralBasisExpansion)
from .low_rank_neural_basis_expansion import LowRankNeuralBasisExpansion
from .functorch_utils import (
        flatten_grad_functorch,
        unflatten_nn_functorch
)
from .getter import get_neural_basis_expansion, get_matmul_neural_basis_expansion
