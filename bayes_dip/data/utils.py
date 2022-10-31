"""
Provides data utilities.
"""

from .trafo import (
        get_walnut_2d_ray_trafo, get_parallel_beam_2d_matmul_ray_trafo, BaseRayTrafo)

def get_ray_trafo(name : str, kwargs : dict) -> BaseRayTrafo:
    """
    Return the ray transform by setting name and keyword arguments.

    Parameters
    ----------
    name : str
        Name of the setting, one of: ``'mnist', 'kmnist', 'rectangles', 'walnut'``.
    kwargs : dict
        Keyword arguments, specific to the setting. Passed as a dictionary.

        For the settings ``'mnist', 'kmnist', 'rectangles'``, the arguments are:

            ``'im_shape'`` : tuple of int
                Image shape.
            ``'num_angles'`` : int
                Number of projection angles.
            ``'angular_sub_sampling'`` : int
                Sub-sampling factor for the projection angles.
                To disable sub-sampling, set this to ``1``.

        For the setting ``'walnut'``, the arguments are those of
        :func:`bayes_dip.data.trafo.walnut_2d_ray_trafo.get_walnut_2d_ray_trafo`,
        but with all arguments being required.

    Returns
    -------
    ray_trafo : BaseRayTrafo
        Ray transform.
    """
    if name == 'mnist':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                impl=kwargs.get('impl', 'astra_cuda'))
    elif name == 'kmnist':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                impl=kwargs.get('impl', 'astra_cuda'))
    elif name == 'rectangles':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                impl=kwargs.get('impl', 'astra_cuda'))
    elif name == 'walnut':
        ray_trafo = get_walnut_2d_ray_trafo(
                data_path=kwargs['data_path'],
                matrix_path=kwargs['matrix_path'],
                walnut_id=kwargs['walnut_id'],
                orbit_id=kwargs['orbit_id'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                proj_col_sub_sampling=kwargs['proj_col_sub_sampling'])
    else:
        raise ValueError

    return ray_trafo
