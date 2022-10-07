"""
Provides :class:`LinearSandwichCov`.
"""

from abc import ABC, abstractmethod
from torch import Tensor, nn

class LinearSandwichCov(nn.Module, ABC):
    """
    Wrapper for a covariance module that applies a linear operation before and the transposed
    operation after it.
    """

    def __init__(self,
        inner_cov: nn.Module,
        ) -> None:
        """
        Parameters
        ----------
        inner_cov : nn.Module
            Inner covariance that is sandwiched by :meth:`lin_op` and :meth:`lin_op_transposed`.
        """

        super().__init__()

        self.inner_cov = inner_cov

    def forward(self,
                v: Tensor,
                **kwargs
            ) -> Tensor:
        """
        Evaluate ``(lin_op @ inner_cov @ lin_op.T @ v.flatten()).view(*v.shape)``, where ``lin_op``
        and ``inner_cov`` are matrix representations of :meth:`lin_op` and :attr:`inner_cov`,
        respectively.

        Parameters
        ----------
        v : Tensor
            Inputs. Shape: ``(batch_size, *)``.
        kwargs : dict, optional
            Keyword arguments forwarded to :attr:`inner_cov`.

        Returns
        -------
        Tensor
            Products. Shape: same as ``v``.
        """

        v = self.lin_op_transposed(v)
        v = self.inner_cov(v, **kwargs)
        v = self.lin_op(v)

        return v

    @abstractmethod
    def lin_op(self, v: Tensor) -> Tensor:
        """
        Linear operation mapping from the space of :attr:`inner_cov` to this covariance's space.

        :meth:`lin_op_transposed` should implement the transposed of this linear operation.

        Parameters
        ----------
        v : Tensor
            Inputs. The shape is the working (i.e. input and output) shape of :attr:`inner_cov`,
            ``(batch_size, ?)``.

        Returns
        -------
        out : Tensor
            Products. Shape: ``(batch_size, *)``.
        """
        raise NotImplementedError

    @abstractmethod
    def lin_op_transposed(self, v: Tensor) -> Tensor:
        """
        Linear operation mapping from this covariance's space to the space of :attr:`inner_cov`.

        This method should implement the transposed of the linear operation :meth:`lin_op`.

        Parameters
        ----------
        v : Tensor
            Inputs. Shape: ``(batch_size, *)``.

        Returns
        -------
        out : Tensor
            Products. The shape is the working (i.e. input and output) shape of :attr:`inner_cov`,
            ``(batch_size, ?)``.
        """
        raise NotImplementedError
