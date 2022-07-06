from typing import Tuple
from torch import Tensor, nn
from ..base_image_cov import BaseImageCov
from .neural_basis_expansion import NeuralBasisExpansion
from .parameter_cov import ParameterCov

class ImageCov(BaseImageCov):

    def __init__(self,
        parameter_cov: ParameterCov,
        neural_basis_expansion: NeuralBasisExpansion,
        ) -> None:

        super().__init__()

        self.parameter_cov = parameter_cov
        self.neural_basis_expansion = neural_basis_expansion

    def forward(self,
                v: Tensor,
                **kwargs
            ) -> Tensor:

        v = self.neural_basis_expansion.vjp(v.unsqueeze(dim=1))
        v = self.parameter_cov(v, **kwargs)
        _, v = self.neural_basis_expansion.jvp(v)

        return v.squeeze(dim=1)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.neural_basis_expansion.jac_shape[0],) * 2
