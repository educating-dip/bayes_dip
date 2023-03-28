"""
Provides data utilities.
"""

from .trafo import (
        get_walnut_2d_ray_trafo, get_parallel_beam_2d_matmul_ray_trafo,
        get_walnut_3d_ray_trafo,
        get_parallel_beam_2d_matmul_ray_trafos_bayesian_exp_design,
        BaseRayTrafo)

def get_ray_trafo(name : str, kwargs : dict, return_full : bool = False) -> BaseRayTrafo:
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
    return_full : bool 
        If `True` return pair of full ray_trafo (i.e. `angular_sub_sampling = 1`)
        and ray_trafo :class:`bayes_dip.data.BaseRayTrafo`. 

    Returns
    -------
    ray_trafo : BaseRayTrafo
        Ray transform.
    """
    if name == 'mnist':
        func = get_parallel_beam_2d_matmul_ray_trafo if not return_full else get_parallel_beam_2d_matmul_ray_trafos_bayesian_exp_design
        ray_trafo = func(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                impl=kwargs.get('impl', 'astra_cuda'))
    elif name == 'kmnist':
        func = get_parallel_beam_2d_matmul_ray_trafo if not return_full else get_parallel_beam_2d_matmul_ray_trafos_bayesian_exp_design
        ray_trafo = func(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                impl=kwargs.get('impl', 'astra_cuda'))
    elif name == 'rectangles':
        func = get_parallel_beam_2d_matmul_ray_trafo if not return_full else get_parallel_beam_2d_matmul_ray_trafos_bayesian_exp_design
        ray_trafo = func(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                impl=kwargs.get('impl', 'astra_cuda'))        
    elif name == 'walnut':
        assert not return_full
        ray_trafo = get_walnut_2d_ray_trafo(
                data_path=kwargs['data_path'],
                matrix_path=kwargs['matrix_path'],
                walnut_id=kwargs['walnut_id'],
                orbit_id=kwargs['orbit_id'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                proj_col_sub_sampling=kwargs['proj_col_sub_sampling'])
    elif name == 'walnut_3d':
        assert not return_full
        ray_trafo = get_walnut_3d_ray_trafo(
                data_path=kwargs['data_path'],
                walnut_id=kwargs['walnut_id'],
                orbit_id=kwargs['orbit_id'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                proj_row_sub_sampling=kwargs['proj_row_sub_sampling'],
                proj_col_sub_sampling=kwargs['proj_col_sub_sampling'],
                vol_down_sampling=kwargs['vol_down_sampling'])
    else:
        raise ValueError

    if return_full:

        subsampling_indices = range(
                0,
                len(ray_trafo[0].angles),
                kwargs['angular_sub_sampling']
            )
        ray_trafo = (*ray_trafo, subsampling_indices)

    return ray_trafo
