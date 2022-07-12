from typing import Tuple
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

        super().__init__()

        self.inner_cov = inner_cov

    def forward(self,
                v: Tensor,
                **kwargs
            ) -> Tensor:

        v = self.lin_op_transposed(v)
        v = self.inner_cov(v, **kwargs)
        v = self.lin_op(v)

        return v

    @abstractmethod
    def lin_op(self, v: Tensor) -> Tensor:
        """
        Linear operation mapping the output of inner_cov to the other covariance space.
        :meth:`lin_op_transposed` should implement the transposed of this linear operation.
        """
        raise NotImplementedError

    @abstractmethod
    def lin_op_transposed(self, v: Tensor) -> Tensor:
        """
        Linear operation mapping the output of inner_cov to the other covariance space.
        This method should implement the transposed of the linear operation :meth:`lin_op`.
        """
        raise NotImplementedError
