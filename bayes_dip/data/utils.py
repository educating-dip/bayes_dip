"""
Provides data utilities.
"""

from .trafo import (
        get_walnut_2d_ray_trafo, get_parallel_beam_2d_matmul_ray_trafo,
        get_walnut_3d_ray_trafo,
        get_param_fan_beam_2d_ray_trafo,
        get_rect_padded_param_fan_beam_2d_ray_trafo,
        get_parallel_beam_2d_matmul_ray_trafos_bayesian_exp_design,
        BaseRayTrafo)

def _get_simple_ray_trafo(kwargs: dict, return_full: bool = False) -> BaseRayTrafo:
    geometry = kwargs.get('geometry', 'parallel')
    if geometry == 'parallel':
        impl = kwargs.get('impl', 'astra_cuda_via_matrix')
        if kwargs.get('impl', 'astra_cuda_via_matrix'):
            func = get_parallel_beam_2d_matmul_ray_trafo if not return_full else get_parallel_beam_2d_matmul_ray_trafos_bayesian_exp_design
            ray_trafo = func(
                    im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                    angular_sub_sampling=kwargs['angular_sub_sampling'],
                    impl='astra_cuda')
        else:
            raise ValueError(f'Unknown implementation "{impl}" for geometry "{geometry}"')
    else:
        raise ValueError(f'Unknown simple ray trafo geometry "{geometry}"')
    return ray_trafo

def get_ray_trafo(name : str, kwargs : dict, return_full : bool = False) -> BaseRayTrafo:
    """
    Return the ray transform by trafo name and keyword arguments.

    Parameters
    ----------
    name : str
        Name of the trafo, one of: ``'simple', 'param_fan_beam', 'walnut', 'walnut_3d'``.
    kwargs : dict
        Keyword arguments, specific to the setting. Passed as a dictionary.

        For ``name == 'simple'``, the arguments are:

            ``'im_shape'`` : tuple of int
                Image shape.
            ``'num_angles'`` : int
                Number of projection angles.
            ``'angular_sub_sampling'`` : int
                Sub-sampling factor for the projection angles.
                To disable sub-sampling, set this to ``1``.
            ``'geometry'`` : str, optional
                Geometry type. Currently only ``'parallel'`` (the default) is
                supported.
            ``'geometry_specs'`` : dict, optional
                Additional geometry specifications. May become relevant for
                other geometries than ``'parallel'`` in the future.
            ``'impl'`` : str, optional
                Implementation. Currently only ``'astra_cuda_via_matrix'``
                (the default) is supported.

        For ``name == 'param_fan_beam'``, the arguments are:

            ``'im_shape'`` : tuple of int
                Image shape.
            ``'num_angles'`` : int
                Number of projection angles.
            ``'rect_padded'`` : bool
                Whether to use a :class:`PaddingRayTransform`-wrapped trafo
                in order to cover the whole (rectangular) image.
            ``'num_det_pixels'`` : int
                Number of detector pixels.
            ``'src_radius'`` : float
                Distance from source to origin.
            ``'angular_sub_sampling'`` : int
                Sub-sampling factor for the projection angles.
                To disable sub-sampling, set this to ``1``.

        For ``name in ('walnut', 'walnut_3d')``, the arguments are those of
        :func:`bayes_dip.data.trafo.walnut_2d_ray_trafo.get_walnut_2d_ray_trafo` and
        :func:`bayes_dip.data.trafo.walnut_3d_ray_trafo.get_walnut_3d_ray_trafo`,
        respectively, but with all arguments being required.
    return_full : bool 
        If `True` return pair of full ray_trafo (i.e. `angular_sub_sampling = 1`)
        and ray_trafo :class:`bayes_dip.data.BaseRayTrafo`.
        Only supported with ``name in ('simple', 'param_fan_beam')``.

    Returns
    -------
    ray_trafo : BaseRayTrafo
        Ray transform.
    """
    if name == 'simple':
        ray_trafo = _get_simple_ray_trafo(kwargs, return_full=return_full)
    elif name == 'param_fan_beam':
        func = get_rect_padded_param_fan_beam_2d_ray_trafo if kwargs['rect_padded'] else get_param_fan_beam_2d_ray_trafo
        ray_trafo = func(
                im_shape=kwargs['im_shape'],
                num_angles=kwargs['num_angles'],
                num_det_pixels=kwargs['num_det_pixels'],
                src_radius=kwargs['src_radius'],
                angular_sub_sampling=kwargs['angular_sub_sampling'])
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
