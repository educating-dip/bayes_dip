from typing import Tuple
from torch import Tensor
from ..base_image_cov import BaseImageCov
from ..linear_sandwich_cov import LinearSandwichCov
from .neural_basis_expansion import NeuralBasisExpansion
from .parameter_cov import ParameterCov

class ImageCov(BaseImageCov, LinearSandwichCov):

    def __init__(self,
        parameter_cov: ParameterCov,
        neural_basis_expansion: NeuralBasisExpansion,
        ) -> None:

        super().__init__(inner_cov=parameter_cov)

        self.parameter_cov = parameter_cov
        self.neural_basis_expansion = neural_basis_expansion

    forward = LinearSandwichCov.forward  # lin_op @ parameter_cov @ lin_op_transposed

    def lin_op(self, v: Tensor) -> Tensor:
        return self.neural_basis_expansion.jvp(v).squeeze(dim=1)

    def lin_op_transposed(self, v: Tensor) -> Tensor:
        return self.neural_basis_expansion.vjp(v.unsqueeze(dim=1))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.neural_basis_expansion.jac_shape[0],) * 2
