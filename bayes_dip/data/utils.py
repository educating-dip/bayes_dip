from .trafo import (
        get_walnut_2d_ray_trafo, get_parallel_beam_2d_matmul_ray_trafo)

def get_ray_trafo(name, kwargs):
    if name == 'mnist':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'])
    elif name == 'kmnist':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'])
    elif name == 'rectangles':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'])
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
