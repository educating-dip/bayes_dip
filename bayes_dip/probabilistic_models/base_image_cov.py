from typing import Tuple
from abc import ABC, abstractmethod
from torch import Tensor, nn

class BaseImageCov(nn.Module, ABC):

    @abstractmethod
    def forward(self,
            v: Tensor
            ) -> Tensor:

        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self,
            ) -> Tuple[int, int]:

        raise NotImplementedError
