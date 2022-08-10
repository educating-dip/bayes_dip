"""Provides :class:`ImageCov`"""
from typing import Tuple, Union, Optional
from torch import Tensor
from ..base_image_cov import BaseImageCov
from ..linear_sandwich_cov import LinearSandwichCov
from .neural_basis_expansion import BaseNeuralBasisExpansion
from .parameter_cov import ParameterCov

class ImageCov(BaseImageCov, LinearSandwichCov):
    """
    Covariance in image space.
    """

    def __init__(self,
        parameter_cov: ParameterCov,
        neural_basis_expansion: BaseNeuralBasisExpansion,
        ) -> None:
        """
        Parameters
        ----------
        parameter_cov : :class:`bayes_dip.probabilistic_models.ParameterCov`
            Parameter space covariance module.
        neural_basis_expansion : :class:`bayes_dip.probabilistic_models.BaseNeuralBasisExpansion`
            Object for Jacobian vector products (:meth:`jvp`) and vector Jacobian products
            (:meth:`vjp`).
        """

        super().__init__(inner_cov=parameter_cov)

        self.neural_basis_expansion = neural_basis_expansion

    forward = LinearSandwichCov.forward  # lin_op @ parameter_cov @ lin_op_transposed

    def lin_op(self, v: Tensor) -> Tensor:
        """
        Parameters
        ----------
        v : Tensor
            Input. Shape: ``(batch_size, self.neural_basis_expansion.num_params)``

        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, 1, *self.neural_basis_expansion.nn_out_shape[2:])``
        """

        return self.neural_basis_expansion.jvp(v).squeeze(dim=1)

    def lin_op_transposed(self, v: Tensor) -> Tensor:
        """
        Parameters
        ----------
        v : Tensor
            Input. Shape: ``(*self.neural_basis_expansion.nn_out_shape)``

        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, self.neural_basis_expansion.num_params)``
        """

        return self.neural_basis_expansion.vjp(v.unsqueeze(dim=1))

    def sample(self,
        num_samples: int = 10,
        return_weight_samples: bool = False,
        mean: Optional[Tensor] = None,
        ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        """
        Return num_samples draws from Gaussian prior over images

        Parameters
        ----------
        num_samples : int
        return_weight_samples : bool
            Whether to return parameters samples from Gaussian prior over nn weights.
            The default is `False`.

        Returns
        -------
        Tensor or tuple of Tensor
            ``samples`` or a tuple ``(samples, weight_samples)``, with
            ``samples.shape == (batch_size, 1, *self.neural_basis_expansion.nn_out_shape[2:])``
            and ``weight_samples.shape == (batch_size, self.neural_basis_expansion.num_params)``.
            ``weight_samples`` always has mean zero.
        """

        # params ~ N(0, parameter_cov)
        weight_samples = self.inner_cov.sample(num_samples=num_samples)
        # image = J_{params} @ params
        samples = self.lin_op(weight_samples)

        if mean is not None:
            samples = samples + mean
        return samples if not return_weight_samples else (samples, weight_samples)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.neural_basis_expansion.jac_shape[0],) * 2
