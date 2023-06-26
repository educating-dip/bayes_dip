"""
Provides :class:`PaddedRayTrafo`.
"""

from typing import Tuple, Union
import torch
from torch import Tensor
import numpy as np
from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo


class PaddedRayTrafo(BaseRayTrafo):
    """
    Ray transform wrapper that pads the input and crops the back-projection.

    This can be utilized to force the image content to be within the
    field-of-view of the wrapped ray transform.
    """
    def __init__(self, im_shape: Union[Tuple[int, int], Tuple[int, int, int]], ray_trafo: BaseRayTrafo):
        """
        im_shape : 2-tuple or 3-tuple of int
            Image shape.
            Must be smaller than or equal to the image shape of `ray_trafo`, and
            the difference must be even in each dimension.
        ray_trafo : :class:`BaseRayTrafo`
            Ray transform defined on a larger image shape but with a possibly
            limited field-of-view.
        """

        assert all(s <= s_pad for s, s_pad in zip(im_shape, ray_trafo.im_shape))
        assert all((s_pad - s) % 2 == 0 for s, s_pad in zip(im_shape, ray_trafo.im_shape))

        super().__init__(im_shape=im_shape, obs_shape=ray_trafo.obs_shape)
        self.ray_trafo = ray_trafo

        halfpad = tuple((s_pad - s) // 2 for s, s_pad in zip(self.im_shape, self.ray_trafo.im_shape))
        self._reversed_padding_repeated_twice = tuple(x for x in reversed(halfpad) for _ in range(2))
        self._unpadding_index = (..., *(slice(p, p+s) for p, s in zip(halfpad, self.im_shape)))

    @property
    def angles(self) -> np.ndarray:
        return self.ray_trafo.angles

    def trafo(self, x: Tensor) -> Tensor:
        x = torch.nn.functional.pad(x, self._reversed_padding_repeated_twice)
        return self.ray_trafo(x)

    def trafo_adjoint(self, observation: Tensor) -> Tensor:
        x = self.ray_trafo.trafo_adjoint(observation)
        x = x[self._unpadding_index]
        return x

    def fbp(self, observation: Tensor) -> Tensor:
        x = self.ray_trafo.fbp(observation)
        x = x[self._unpadding_index]
        return x

    trafo_flat = BaseRayTrafo._trafo_flat_via_trafo
    trafo_adjoint_flat = BaseRayTrafo._trafo_adjoint_flat_via_trafo_adjoint
