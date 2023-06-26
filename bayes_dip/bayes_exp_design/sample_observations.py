from typing import Optional, Any 
import torch
from tqdm import tqdm
from math import ceil

from torch import Tensor
from bayes_dip.probabilistic_models import ObservationCov
from bayes_dip.data import MatmulRayTrafo

def sample_observations_shifted_bayes_exp_design(
    observation_cov: ObservationCov,
    ray_trafo_obj: MatmulRayTrafo, 
    ray_trafo_comp_obj: MatmulRayTrafo,
    cov_obs_mat_chol: Tensor,
    mc_samples: int,
    batch_size: int,
    device: Optional[Any] = None,
    ):

    num_batches = ceil(mc_samples / batch_size)
    mc_samples = num_batches * batch_size
    
    comp_observation_samples = []
    images_samples = []
    for _ in tqdm(
        range(num_batches), desc='sample_from_posterior', miniters=num_batches//100):
    
        x_samples = observation_cov.image_cov.sample(
            num_samples=batch_size,
            return_weight_samples=False
            )
        samples = ray_trafo_obj(x_samples)
        samples_comp = ray_trafo_comp_obj(x_samples)

        noise_term = (observation_cov.log_noise_variance.exp()**.5) * torch.randn_like(samples)
        samples = (noise_term - samples).reshape(batch_size, -1)
        samples = torch.linalg.solve_triangular(cov_obs_mat_chol.T, torch.linalg.solve_triangular(
            cov_obs_mat_chol, samples.T, upper=False), upper=True).T

        delta_x = ray_trafo_obj.trafo_adjoint(samples.view(batch_size, 1, *ray_trafo_obj.obs_shape))
        delta_x = observation_cov.image_cov(delta_x)
        delta_y = ray_trafo_comp_obj(delta_x)

        comp_observation_samples.append((samples_comp + delta_y).detach().to(device))
        images_samples.append(x_samples + delta_x.view(*x_samples.shape))

    samples = torch.cat(comp_observation_samples, axis=0)
    images_samples = torch.cat(images_samples, axis=0)

    return samples, images_samples

    