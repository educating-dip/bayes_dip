import os
from torch.utils.data import Dataset, TensorDataset
from bayes_dip.data import get_ray_trafo, SimulatedDataset
from bayes_dip.data import (
        RectanglesDataset, get_mnist_testset, get_kmnist_testset,
        get_walnut_2d_observation, get_walnut_2d_ground_truth)
from .utils import get_original_cwd

def get_standard_ray_trafo(cfg):
    kwargs = {}
    kwargs['angular_sub_sampling'] = cfg.trafo.angular_sub_sampling
    if cfg.dataset.name in ('mnist', 'kmnist', 'rectangles'):
        kwargs['im_shape'] = (cfg.dataset.im_size, cfg.dataset.im_size)
        kwargs['num_angles'] = cfg.trafo.num_angles
    elif cfg.dataset.name in ('walnut', 'walnut_120_angles'):
        kwargs['data_path'] = os.path.join(get_original_cwd(), cfg.dataset.data_path)
        kwargs['matrix_path'] = os.path.join(get_original_cwd(), cfg.dataset.data_path)
        kwargs['walnut_id'] = cfg.dataset.walnut_id
        kwargs['orbit_id'] = cfg.trafo.orbit_id
        kwargs['proj_col_sub_sampling'] = cfg.trafo.proj_col_sub_sampling
    else:
        raise ValueError
    return get_ray_trafo(cfg.dataset.name, kwargs=kwargs)

def get_standard_dataset(cfg, ray_trafo, use_fixed_seeds_starting_from=1, device=None) -> Dataset:
    """
    Returns a dataset of tuples ``noisy_observation, x, filtbackproj``, where
        * `noisy_observation` has shape ``(1,) + obs_shape``
        * `x` is the ground truth (label) and has shape ``(1,) + im_shape``
        * ``filtbackproj = FBP(noisy_observation)`` has shape ``(1,) + im_shape``

    Parameters
    ----------
    use_fixed_seeds_starting_from : int, optional
        Fixed seed for noise generation, only used in simulated datasets.
    device : str or torch.device, optional
        If specified, data will be moved to the device. `ray_trafo`
        (including `ray_trafo.fbp`) must support tensors on the device.
    """
    if cfg.dataset.name == 'mnist':

        image_dataset = get_mnist_testset()
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=cfg.dataset.noise_stddev,
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)

    elif cfg.dataset.name == 'kmnist':

        image_dataset = get_kmnist_testset()
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=cfg.dataset.noise_stddev,
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)

    elif cfg.dataset.name == 'rectangles':

        image_dataset = RectanglesDataset(
                (cfg.dataset.im_size, cfg.dataset.im_size),
                num_rects=cfg.dataset.num_rects,
                num_angle_modes=cfg.dataset.num_angle_modes,
                angle_modes_sigma=cfg.dataset.angle_modes_sigma)
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=cfg.dataset.noise_stddev,
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)

    elif cfg.dataset.name == 'walnut':

        noisy_observation = get_walnut_2d_observation(
                data_path=os.path.join(get_original_cwd(), cfg.dataset.data_path),
                walnut_id=cfg.dataset.walnut_id, orbit_id=cfg.trafo.orbit_id,
                angular_sub_sampling=cfg.trafo.angular_sub_sampling,
                proj_col_sub_sampling=cfg.trafo.proj_col_sub_sampling,
                scaling_factor=cfg.dataset.scaling_factor).to(device=device)
        ground_truth = get_walnut_2d_ground_truth(
                data_path=os.path.join(get_original_cwd(), cfg.dataset.data_path),
                walnut_id=cfg.dataset.walnut_id, orbit_id=cfg.trafo.orbit_id,
                scaling_factor=cfg.dataset.scaling_factor).to(device=device)
        filtbackproj = ray_trafo.fbp(
                noisy_observation[None].to(device=device))[0].to(device=device)
        dataset = TensorDataset(  # include batch dims
                noisy_observation[None], ground_truth[None], filtbackproj[None])

    else:
        raise ValueError

    return dataset
