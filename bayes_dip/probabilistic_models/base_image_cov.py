from abc import ABC, abstractmethod
from torch import Tensor, nn

class BaseImageCov(nn.Module, ABC):

    @abstractmethod
    def forward(self,
            v: Tensor
            ) -> Tensor:

        raise NotImplementedError
