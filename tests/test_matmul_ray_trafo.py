"""
Tests for :module:`bayes_dip.data.trafo.parallel_beam_2d_ray_trafo`.
"""

import pytest
import torch
import numpy as np
from bayes_dip.data import MatmulRayTrafo

@pytest.fixture(scope='session')
def matmul_ray_trafo_and_im_obs_data():
    im_shape = (128, 128)
    obs_shape = (30, 183)
    matrix = torch.rand(np.prod(obs_shape), np.prod(im_shape))
    ray_trafo = MatmulRayTrafo(
            im_shape=im_shape, obs_shape=obs_shape, matrix=matrix)
    torch.random.manual_seed(1)
    x = torch.rand(1, 1, *im_shape)
    y = torch.rand(1, 1, *obs_shape)
    return ray_trafo, x, y

def test_adjoint_scalar_product_property(matmul_ray_trafo_and_im_obs_data):
    """
    Test ``<x,ray_trafo.trafo_adjoint(y)> == <y,ray_trafo.trafo(x)>``.
    """
    ray_trafo, x, y = matmul_ray_trafo_and_im_obs_data
    y2 = ray_trafo.trafo(x)
    x2 = ray_trafo.trafo_adjoint(y)
    sp_x = (x * x2).sum().item()
    sp_y = (y * y2).sum().item()
    assert sp_y / sp_x == pytest.approx(1., 1e-7)
