"""Provides :class:`ImageCov`"""
from typing import Tuple, Union, Optional
import torch
from torch import Tensor
from torch import nn
from ..base_image_cov import BaseImageCov
from ..linear_sandwich_cov import LinearSandwichCov
from .neural_basis_expansion import BaseNeuralBasisExpansion, BaseMatmulNeuralBasisExpansion
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

    def lin_op(self, v: Tensor, **kwargs) -> Tensor:
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

        return self.neural_basis_expansion.jvp(v, **kwargs).squeeze(dim=1)

    def lin_op_transposed(self, v: Tensor, **kwargs) -> Tensor:
        """
        Parameters
        ----------
        v : Tensor
            Input. Shape: ``(batch_size, 1, *self.neural_basis_expansion.nn_out_shape[2:])``

        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, self.neural_basis_expansion.num_params)``
        """

        return self.neural_basis_expansion.vjp(v.unsqueeze(dim=1), **kwargs)

    def sample(self,
        num_samples: int = 10,
        return_weight_samples: bool = False,
        mean: Optional[Tensor] = None,
        sample_only_from_prior: nn.Module = None,
        ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        """
        Return num_samples draws from Gaussian prior over images

        Parameters
        ----------
        num_samples : int
        return_weight_samples : bool
            Whether to return parameters samples from Gaussian prior over nn weights.
            The default is ``False``.

        Returns
        -------
        Tensor or tuple of Tensor
            ``samples`` or a tuple ``(samples, weight_samples)``, with
            ``samples.shape == (batch_size, 1, *self.neural_basis_expansion.nn_out_shape[2:])``
            and ``weight_samples.shape == (batch_size, self.neural_basis_expansion.num_params)``.
            ``weight_samples`` always has mean zero.
        """

        # params ~ N(0, parameter_cov)
        weight_samples = self.inner_cov.sample(
                num_samples=num_samples, sample_only_from_prior=sample_only_from_prior)

        if sample_only_from_prior is None:
            samples = self.lin_op(weight_samples)
        else:
            weight_sub_slice = self.inner_cov.params_slices_per_prior[sample_only_from_prior]
            # image = J_{params} @ params
            if isinstance(self.neural_basis_expansion, BaseMatmulNeuralBasisExpansion):
                samples = self.lin_op(
                        weight_samples,
                        sub_slice=weight_sub_slice)
            else:
                weight_samples_full = torch.zeros(
                        num_samples, self.inner_cov.shape[0], device=weight_samples.device)
                weight_samples_full[:, weight_sub_slice] = weight_samples
                samples = self.lin_op(weight_samples_full)

        if mean is not None:
            samples = samples + mean
        return samples if not return_weight_samples else (samples, weight_samples)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.neural_basis_expansion.jac_shape[0],) * 2
