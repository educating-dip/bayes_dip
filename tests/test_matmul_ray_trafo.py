"""
Tests for :module:`bayes_dip.data.trafo.parallel_beam_2d_ray_trafo`.
"""

import unittest
import torch
import numpy as np
from bayes_dip.data import MatmulRayTrafo


class TestMatmulRayTrafo(unittest.TestCase):
    """
    Tests for :class:`MatmulRayTrafo`.
    """
    def test_adjoint_scalar_product_property(self):
        """
        Test ``<x,ray_trafo.trafo_adjoint(y)> == <y,ray_trafo.trafo(x)>``.
        """
        im_shape = (128, 128)
        obs_shape = (30, 183)
        matrix = torch.rand(np.prod(obs_shape), np.prod(im_shape))
        ray_trafo = MatmulRayTrafo(
                im_shape=im_shape, obs_shape=obs_shape, matrix=matrix)
        torch.random.manual_seed(1)
        x = torch.rand(1, 1, *im_shape)
        y = torch.rand(1, 1, *obs_shape)
        y2 = ray_trafo.trafo(x)
        x2 = ray_trafo.trafo_adjoint(y)
        sp_x = (x * x2).sum().item()
        sp_y = (y * y2).sum().item()
        self.assertAlmostEqual(sp_y / sp_x, 1., places=6)


if __name__ == '__main__':
    unittest.main()
