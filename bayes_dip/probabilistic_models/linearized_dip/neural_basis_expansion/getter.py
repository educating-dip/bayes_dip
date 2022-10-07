"""
Provides getter (creator) functions for neural basis expansions.
"""

from typing import Sequence, Tuple, Optional
from torch import nn, Tensor
from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from .base_neural_basis_expansion import BaseNeuralBasisExpansion, BaseMatmulNeuralBasisExpansion
from .neural_basis_expansion import NeuralBasisExpansion, MatmulNeuralBasisExpansion
from .gprior_neural_basis_expansion import (
        GpriorNeuralBasisExpansion, MatmulGpriorNeuralBasisExpansion)

def get_neural_basis_expansion(
        nn_model: nn.Module,
        nn_input: Tensor,
        ordered_nn_params: Sequence[nn.Parameter],
        nn_out_shape: Optional[Tuple[int, int, int, int]] = None,
        use_gprior: bool = False,
        trafo: Optional[BaseRayTrafo] = None,
        scale_kwargs: Optional[dict] = None,
        ) -> BaseNeuralBasisExpansion:
    """
    Return a :class:`.NeuralBasisExpansion` or :class:`.GpriorNeuralBasisExpansion`.

    Parameters
    ----------
    nn_model : :class:`nn.Module`
        Network.
    nn_input : Tensor
        Network input.
    ordered_nn_params : sequence of nn.Parameter
        Sequence of parameters that should be included in this expansion.
    nn_out_shape : 4-tuple of int, optional
        Shape of the network output. If not specified, it is determined by a forward call
        (performed in eval mode).
    use_gprior : bool, optional
        Whether to wrap the neural basis expansion in a :class:`GpriorNeuralBasisExpansion`.
        The default is ``False``.
    trafo : :class:`.BaseRayTrafo`, optional
        Ray transform; required iff ``use_gprior``.
    scale_kwargs : dict, optional
        ``scale_kwargs`` passed to :meth:`.GpriorNeuralBasisExpansion.__init__`; required iff
        ``use_gprior``.

    Returns
    -------
    neural_basis_expansion : :class:`.BaseNeuralBasisExpansion`
        Neural basis expansion, instance of :class:`.NeuralBasisExpansion` or
        :class:`.GpriorNeuralBasisExpansion`, depending on ``use_gprior``.
    """
    neural_basis_expansion = NeuralBasisExpansion(
            nn_model=nn_model,
            nn_input=nn_input,
            ordered_nn_params=ordered_nn_params,
            nn_out_shape=nn_out_shape,
    )
    if use_gprior:
        neural_basis_expansion = GpriorNeuralBasisExpansion(
                neural_basis_expansion=neural_basis_expansion,
                trafo=trafo,
                scale_kwargs=scale_kwargs,
                device=nn_input.device,
        )
    return neural_basis_expansion

def get_matmul_neural_basis_expansion(
        nn_model: nn.Module,
        nn_input: Tensor,
        ordered_nn_params: Sequence[nn.Parameter],
        nn_out_shape: Optional[Tuple[int, int, int, int]] = None,
        use_gprior: bool = False,
        trafo: Optional[BaseRayTrafo] = None,
        scale_kwargs: Optional[dict] = None,
        ) -> BaseMatmulNeuralBasisExpansion:
    """
    Return a :class:`.MatmulNeuralBasisExpansion` or :class:`.MatmulGpriorNeuralBasisExpansion`.

    Parameters
    ----------
    nn_model : :class:`nn.Module`
        Network.
    nn_input : Tensor
        Network input.
    ordered_nn_params : sequence of nn.Parameter
        Sequence of parameters that should be included in this expansion.
    nn_out_shape : 4-tuple of int, optional
        Shape of the network output. If not specified, it is determined by a forward call
        (performed in eval mode).
    use_gprior : bool, optional
        Whether to wrap the neural basis expansion in a :class:`MatmulGpriorNeuralBasisExpansion`.
        The default is ``False``.
    trafo : :class:`.BaseRayTrafo`, optional
        Ray transform; required iff ``use_gprior``.
    scale_kwargs : dict, optional
        ``scale_kwargs`` passed to :meth:`.MatmulGpriorNeuralBasisExpansion.__init__`; required iff
        ``use_gprior``.

    Returns
    -------
    neural_basis_expansion : :class:`.BaseMatmulNeuralBasisExpansion`
        Neural basis expansion, instance of :class:`.MatmulNeuralBasisExpansion` or
        :class:`.MatmulGpriorNeuralBasisExpansion`, depending on ``use_gprior``.
    """
    matmul_neural_basis_expansion = MatmulNeuralBasisExpansion(
            nn_model=nn_model,
            nn_input=nn_input,
            ordered_nn_params=ordered_nn_params,
            nn_out_shape=nn_out_shape,
    )
    if use_gprior:
        matmul_neural_basis_expansion = MatmulGpriorNeuralBasisExpansion(
                neural_basis_expansion=matmul_neural_basis_expansion,
                trafo=trafo,
                scale_kwargs=scale_kwargs,
        )
    return matmul_neural_basis_expansion
