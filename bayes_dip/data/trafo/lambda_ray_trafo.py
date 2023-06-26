"""
Provides :class:`LambdaRayTrafo`.
"""

from __future__ import annotations  # postponed evaluation, to make ArrayLike look good in docs
from typing import Union, Optional, Callable, Tuple, Any
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any
from torch import Tensor
from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo


class LambdaRayTrafo(BaseRayTrafo):
    """
    Ray transform implemented by callables.

    Adjoint computations are accurate in this implementation (which is not
    always the case when using back-projection for the adjoint).
    """

    def __init__(self,
            im_shape: Union[Tuple[int, int], Tuple[int, int, int]],
            obs_shape: Union[Tuple[int, int], Tuple[int, int, int]],
            trafo_fun: Callable[[Tensor], Tensor],
            trafo_adjoint_fun: Callable[[Tensor], Tensor],
            fbp_fun: Optional[Callable[[Tensor], Tensor]] = None,
            angles: Optional[ArrayLike] = None):
        """
        Parameters
        ----------
        im_shape, obs_shape
            See :meth:`BaseRayTrafo.__init__`.
        trafo_fun : callable, optional
            Function applying the forward ray transform, used for providing
            :meth:`trafo`.
        trafo_adjoint_fun : callable, optional
            Function applying the adjoint ray transform (back-projection), used
            for providing :meth:`trafo_adjoint`.
        fbp_fun : callable, optional
            Function applying a filtered back-projection, used for providing
            :meth:`fbp`.
        angles : array-like, optional
            Angles of the ray transform, only used for providing the
            :attr:`angles` property; not used for any computations.
        """
        super().__init__(im_shape=im_shape, obs_shape=obs_shape)

        self.trafo_fun = trafo_fun
        self.trafo_adjoint_fun = trafo_adjoint_fun
        self.fbp_fun = fbp_fun
        self._angles = angles

    @property
    def angles(self) -> ArrayLike:
        """array-like : The angles (in radian)."""
        if self._angles is not None:
            return self._angles
        raise ValueError('`angles` was not set for `LambdaRayTrafo`')

    def trafo(self, x: Tensor) -> Tensor:
        return self.trafo_fun(x)

    def trafo_adjoint(self, observation: Tensor) -> Tensor:
        return self.trafo_adjoint_fun(observation)

    def fbp(self, observation: Tensor) -> Tensor:
        return self.fbp_fun(observation)

    trafo_flat = BaseRayTrafo._trafo_flat_via_trafo
    trafo_adjoint_flat = BaseRayTrafo._trafo_adjoint_flat_via_trafo_adjoint
