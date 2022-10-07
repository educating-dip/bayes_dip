"""
Provides :class:`ParallelBeam2DRayTrafo`, as well as getters
for its matrix representation and a :class:`MatmulRayTrafo` implementation.
"""

from itertools import product
from typing import Tuple
import numpy as np
from odl.contrib.torch import OperatorModule
import odl
from tqdm import tqdm
from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from bayes_dip.data.trafo.matmul_ray_trafo import MatmulRayTrafo


def get_odl_ray_trafo_parallel_beam_2d(
        im_shape: Tuple[int, int],
        num_angles: int,
        first_angle_zero: bool = True,
        impl: str = 'astra_cuda') -> odl.tomo.RayTransform:
    """
    Return an ODL 2D parallel beam ray transform.

    Parameters
    ----------
    im_shape : 2-tuple of int
        Image shape, ``(im_0, im_1)``.
    num_angles : int
        Number of angles (to distribute from ``0`` to ``pi``).
    first_angle_zero : bool, optional
        Whether to shift all angles such that the first angle becomes ``0.``.
        If ``False``, the default angles from ODL are used, where the first angle
        is at half an angle step.
        The default is ``True``.
    impl : str, optional
        Backend for :class:`odl.tomo.RayTransform`.
        The default is ``'astra_cuda'``.
    """

    space = odl.uniform_discr(
            [-im_shape[0] / 2, -im_shape[1] / 2],
            [im_shape[0] / 2, im_shape[1] / 2],
            im_shape,
            dtype='float32')

    default_odl_geometry = odl.tomo.parallel_beam_geometry(
            space, num_angles=num_angles)

    if first_angle_zero:
        default_first_angle = (
                default_odl_geometry.motion_grid.coord_vectors[0][0])
        angle_partition = odl.uniform_partition_fromgrid(
                odl.discr.grid.RectGrid(
                        default_odl_geometry.motion_grid.coord_vectors[0]
                        - default_first_angle))
        geometry = odl.tomo.Parallel2dGeometry(
                apart=angle_partition,
                dpart=default_odl_geometry.det_partition)
    else:
        geometry = default_odl_geometry

    odl_ray_trafo = odl.tomo.RayTransform(
                space, geometry, impl=impl)

    return odl_ray_trafo


class ParallelBeam2DRayTrafo(BaseRayTrafo):
    """
    Ray transform implemented via ODL.

    Adjoint computations use the back-projection (might be slightly inaccurate).
    """

    def __init__(self,
            im_shape: Tuple[int, int],
            num_angles: int,
            first_angle_zero: bool = True,
            angular_sub_sampling: int = 1,
            impl: str = 'astra_cuda'):
        """
        Parameters
        ----------
        im_shape : 2-tuple of int
            Image shape, ``(im_0, im_1)``.
        num_angles : int
            Number of angles (to distribute from ``0`` to ``pi``).
        first_angle_zero : bool, optional
            Whether to shift all angles such that the first angle becomes ``0.``.
            If ``False``, the default angles from ODL are used, where the first angle
            is at half an angle step.
            The default is ``True``.
        angular_sub_sampling : int, optional
            Sub-sampling factor for the angles.
            The default is ``1`` (no sub-sampling).
        impl : str, optional
            Backend for :class:`odl.tomo.RayTransform`.
            The default is ``'astra_cuda'``.
        """
        odl_ray_trafo_full = get_odl_ray_trafo_parallel_beam_2d(
                im_shape, num_angles, first_angle_zero=first_angle_zero,
                impl=impl)
        odl_ray_trafo = odl.tomo.RayTransform(
                odl_ray_trafo_full.domain,
                odl_ray_trafo_full.geometry[::angular_sub_sampling], impl=impl)
        odl_fbp = odl.tomo.fbp_op(odl_ray_trafo)

        obs_shape = odl_ray_trafo.range.shape

        super().__init__(im_shape=im_shape, obs_shape=obs_shape)

        self.odl_ray_trafo = odl_ray_trafo
        self._angles = odl_ray_trafo.geometry.angles

        self.ray_trafo_module = OperatorModule(odl_ray_trafo)
        self.ray_trafo_module_adj = OperatorModule(odl_ray_trafo.adjoint)
        self.fbp_module = OperatorModule(odl_fbp)

    @property
    def angles(self) -> np.ndarray:
        """:class:`np.ndarray` : The angles (in radian)."""
        return self._angles

    def trafo(self, x):
        return self.ray_trafo_module(x)

    def trafo_adjoint(self, observation):
        return self.ray_trafo_module_adj(observation)

    trafo_flat = BaseRayTrafo._trafo_flat_via_trafo
    trafo_adjoint_flat = BaseRayTrafo._trafo_adjoint_flat_via_trafo_adjoint

    def fbp(self, observation):
        return self.fbp_module(observation)

def get_odl_ray_trafo_parallel_beam_2d_matrix(
        im_shape: Tuple[int, int],
        num_angles: int,
        first_angle_zero: bool = True,
        angular_sub_sampling: int = 1,
        impl: str = 'astra_cuda',
        flatten: bool = True) -> np.ndarray:
    """
    Return the matrix representation of an ODL 2D parallel beam ray transform.

    See documentation of :class:`ParallelBeam2DRayTrafo` for
    documentation of the parameters not documented here.

    Parameters
    ----------
    flatten : bool, optional
        If ``True``, the observation dimensions and image dimensions are flattened,
        the resulting shape is ``(np.prod(obs_shape), np.prod(im_shape))``);
        if ``False``, the shape is ``obs_shape + im_shape``.
        The default is ``True``.
    """

    odl_ray_trafo_full = get_odl_ray_trafo_parallel_beam_2d(
                im_shape, num_angles, first_angle_zero=first_angle_zero,
                impl=impl)
    odl_ray_trafo = odl.tomo.RayTransform(
            odl_ray_trafo_full.domain,
            odl_ray_trafo_full.geometry[::angular_sub_sampling], impl=impl)
    obs_shape = odl_ray_trafo.range.shape

    matrix = np.zeros(obs_shape + im_shape, dtype=np.float32)
    x = np.zeros(im_shape, dtype=np.float32)
    for i0, i1 in tqdm(product(range(im_shape[0]), range(im_shape[1])),
            total=im_shape[0] * im_shape[1],
            desc='generating ray transform matrix'):
        x[i0, i1] = 1.
        matrix[:, :, i0, i1] = odl_ray_trafo_full(x)
        x[i0, i1] = 0.

    # matrix = odl.operator.oputils.matrix_representation(
    #         odl_ray_trafo_full)

    if angular_sub_sampling != 1:
        matrix = matrix[::angular_sub_sampling]

    if flatten:
        matrix = matrix.reshape(-1, im_shape[0] * im_shape[1])

    return matrix


def get_parallel_beam_2d_matmul_ray_trafo(
        im_shape: Tuple[int, int],
        num_angles: int,
        first_angle_zero: bool = True,
        angular_sub_sampling: int = 1,
        impl: str = 'astra_cuda') -> MatmulRayTrafo:
    """
    Return a :class:`bayes_dip.data.MatmulRayTrafo` with the matrix
    representation of an ODL 2D parallel beam ray transform.

    See documentation of :class:`ParallelBeam2DRayTrafo` for
    documentation of the parameters.
    """

    odl_ray_trafo_full = get_odl_ray_trafo_parallel_beam_2d(
            im_shape, num_angles, first_angle_zero=first_angle_zero,
            impl=impl)
    odl_ray_trafo = odl.tomo.RayTransform(
            odl_ray_trafo_full.domain,
            odl_ray_trafo_full.geometry[::angular_sub_sampling], impl=impl)
    odl_fbp = odl.tomo.fbp_op(odl_ray_trafo)

    obs_shape = odl_ray_trafo.range.shape
    angles = odl_ray_trafo.geometry.angles

    fbp_module = OperatorModule(odl_fbp)

    matrix = get_odl_ray_trafo_parallel_beam_2d_matrix(
            im_shape, num_angles, first_angle_zero=first_angle_zero,
            angular_sub_sampling=angular_sub_sampling, impl=impl, flatten=True)

    ray_trafo = MatmulRayTrafo(im_shape, obs_shape, matrix, fbp_fun=fbp_module, angles=angles)

    return ray_trafo
