"""
Provides the 3D ray transform for the walnut data.
"""
from functools import partial
import numpy as np
import torch
from .lambda_ray_trafo import LambdaRayTrafo
from .utils.torch_linked_ray_trafo_module import TorchLinkedRayTrafoModule
from bayes_dip.data.walnut_utils import WalnutRayTrafo, astra_fdk_cuda


def _walnut_3d_fdk(observation, im_shape, vol_geom, proj_geom):
    # only trivial batch and channel dims supported
    assert observation.shape[0] == 1 and observation.shape[1] == 1
    # observation.shape: (1, 1, rows, angles, cols)
    observation_np = observation.detach().cpu().numpy().squeeze((0, 1))

    fdk_np = np.zeros(im_shape, dtype=np.float32)

    astra_fdk_cuda(
            projs=observation_np, vol_geom=vol_geom, proj_geom=proj_geom,
            vol_x_out=fdk_np)

    fdk = torch.from_numpy(fdk_np)[None, None].to(observation.device)
    return fdk


def get_walnut_3d_ray_trafo(
        data_path: str,
        walnut_id: int = 1,
        orbit_id: int = 2,
        angular_sub_sampling: int = 1,
        proj_row_sub_sampling: int = 1,
        proj_col_sub_sampling: int = 1,
        vol_down_sampling: int = 1) -> LambdaRayTrafo:
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
    walnut_id : int, optional
        Walnut ID, an integer from 1 to 42.
        The default is ``1``.
    orbit_id : int, optional
        Orbit (source position) ID, options are ``1``, ``2`` or ``3``.
        The default is ``2``.
    angular_sub_sampling : int, optional
        Sub-sampling factor for the angles.
        The default is ``1`` (no sub-sampling).
    proj_row_sub_sampling : int, optional
        Sub-sampling factor for the projection rows.
        The default is ``1`` (no sub-sampling).
    proj_col_sub_sampling : int, optional
        Sub-sampling factor for the projection columns.
        The default is ``1`` (no sub-sampling).
    vol_down_sampling : int, optional
        Down-sampling factor. The same factor is applied to each image dimension.
        The default is ``1``.
    """

    walnut_ray_trafo = WalnutRayTrafo(
            data_path=data_path,
            walnut_id=walnut_id,
            orbit_id=orbit_id,
            vol_down_sampling=vol_down_sampling,
            angular_sub_sampling=angular_sub_sampling,
            proj_row_sub_sampling=proj_row_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling)

    vol_geom = walnut_ray_trafo.vol_geom
    proj_geom = walnut_ray_trafo.proj_geom
    im_shape = walnut_ray_trafo.vol_shape
    obs_shape = walnut_ray_trafo.proj_shape

    torch_trafo = TorchLinkedRayTrafoModule(
            vol_geom, proj_geom, adjoint=False)
    torch_trafo_adjoint = TorchLinkedRayTrafoModule(
            vol_geom, proj_geom, adjoint=True)

    fbp_fun = partial(_walnut_3d_fdk,
            im_shape=im_shape,
            vol_geom=vol_geom,
            proj_geom=proj_geom)

    ray_trafo = LambdaRayTrafo(
            im_shape=im_shape,
            obs_shape=obs_shape,
            trafo_fun=torch_trafo,
            trafo_adjoint_fun=torch_trafo_adjoint,
            fbp_fun=fbp_fun,
            angles=None,
            )

    return ray_trafo
