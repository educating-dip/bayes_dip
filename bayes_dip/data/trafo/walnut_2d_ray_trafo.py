"""
Provides the (pseudo) 2D ray transform for the walnut data.
"""
from typing import Optional
from functools import partial
import torch
import numpy as np
from bayes_dip.data.trafo.matmul_ray_trafo import MatmulRayTrafo
from bayes_dip.data.walnut_utils import (
        get_single_slice_ray_trafo, get_single_slice_ray_trafo_matrix)


def _walnut_2d_fdk(observation, walnut_ray_trafo):
    # only trivial batch and channel dims supported
    assert observation.shape[0] == 1 and observation.shape[1] == 1
    # observation.shape: (1, 1, 1, obs_numel)
    observation_np = observation.detach().cpu().numpy().squeeze((0, 1, 2))

    fdk = walnut_ray_trafo.apply_fdk(observation_np, squeeze=True)
    fdk = torch.from_numpy(fdk)[None, None].to(observation.device)
    return fdk


def get_walnut_2d_ray_trafo(
        data_path: str, matrix_path: Optional[str] = None,
        walnut_id: int = 1, orbit_id: int = 2,
        angular_sub_sampling: int = 1,
        proj_col_sub_sampling: int = 1) -> MatmulRayTrafo:
    """
    Return a :class:`bayes_dip.data.MatmulRayTrafo` with the matrix
    representation of the walnut 2D ray transform.

    A single slice configuration must be defined in
    ``bayes_dip.data.walnut_utils.SINGLE_SLICE_CONFIGS`` for the requested
    ``walnut_id, orbit_id``.

    Parameters
    ----------
    data_path : str
        Walnut dataset path (containing ``'Walnut1'`` as a subfolder).
    matrix_path : str, optional
        Walnut ray transform matrix path (folder containing the ``'.mat'`` file).
        If ``None`` (the default), the value of ``data_path`` is used.
    walnut_id : int, optional
        Walnut ID, an integer from 1 to 42.
        The default is ``1``.
    orbit_id : int, optional
        Orbit (source position) ID, options are ``1``, ``2`` or ``3``.
        The default is ``2``.
    angular_sub_sampling : int, optional
        Sub-sampling factor for the angles.
        The default is ``1`` (no sub-sampling).
    proj_col_sub_sampling : int, optional
        Sub-sampling factor for the projection columns.
        The default is ``1`` (no sub-sampling).
    """

    matrix_path = data_path if matrix_path is None else matrix_path

    walnut_kwargs = dict(
            walnut_id=walnut_id, orbit_id=orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling)

    matrix = get_single_slice_ray_trafo_matrix(
            path=matrix_path, **walnut_kwargs)

    walnut_ray_trafo = get_single_slice_ray_trafo(
            data_path=data_path, **walnut_kwargs)

    im_shape = walnut_ray_trafo.vol_shape[1:]
    obs_shape = (1, np.sum(walnut_ray_trafo.proj_mask))

    fbp_fun = partial(_walnut_2d_fdk, walnut_ray_trafo=walnut_ray_trafo)

    ray_trafo = MatmulRayTrafo(im_shape, obs_shape, matrix,
            fbp_fun=fbp_fun, angles=None)

    # expose index information via attribute
    ray_trafo.inds_in_flat_projs_per_angle = (
            walnut_ray_trafo.get_inds_in_flat_projs_per_angle())

    return ray_trafo
