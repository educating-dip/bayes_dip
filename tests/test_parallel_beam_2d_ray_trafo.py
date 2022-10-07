"""
Tests for :module:`bayes_dip.data.trafo.parallel_beam_2d_ray_trafo`.
"""

import unittest
from skimage.metrics import peak_signal_noise_ratio
import odl
import numpy as np
try:
    import astra
except ImportError:
    ASTRA_CUDA_AVAILABLE = False
else:
    ASTRA_CUDA_AVAILABLE = astra.use_cuda()
import matplotlib.pyplot as plt
from bayes_dip.data import (
        get_odl_ray_trafo_parallel_beam_2d, ParallelBeam2DRayTrafo,
        get_odl_ray_trafo_parallel_beam_2d_matrix,
        get_parallel_beam_2d_matmul_ray_trafo)
from bayes_dip.utils.test_utils import get_random_ellipses_images


class TestParallelBeam2DRayTrafo(unittest.TestCase):
    """
    Tests for :class:`ParallelBeam2DRayTrafo`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_on_failure = True
        self.impl = 'astra_cuda' if ASTRA_CUDA_AVAILABLE else 'skimage'

    def assert_psnr_greater(self, x, reco, min_psnr):
        """
        Assert PSNR is greater ``min_psnr`` and show plot on failure if
        ``self.show_on_failure``.
        """
        x = x.cpu().numpy().squeeze()
        reco = reco.cpu().numpy().squeeze()
        psnr = peak_signal_noise_ratio(x, reco, data_range=x.max() - x.min())
        try:
            self.assertGreater(psnr, min_psnr)
        except AssertionError:
            if self.show_on_failure:
                fig, ax = plt.subplots(1, 3)
                vmax = max(np.max(x), np.max(reco))
                ax[0].imshow(x, vmin=0., vmax=vmax, cmap='gray')
                ax[0].set_title('$x$')
                im = ax[1].imshow(reco, vmin=0., vmax=vmax, cmap='gray')
                ax[1].set_title('$reco$')
                ax[1].set_xlabel(f'PSNR: {psnr:.2f} dB')
                fig.colorbar(im, ax=ax[0:2])
                im_diff = ax[2].imshow(reco - x, cmap='PiYG')
                ax[2].set_title('$reco-x$')
                fig.colorbar(im_diff, ax=ax[2])
                plt.show()
            raise

    def test_fbp(self):
        """
        Test if ``ray_trafo.fbp(ray_trafo(x))`` approx. matches ``x``.
        """
        ray_trafo = ParallelBeam2DRayTrafo(
                im_shape=(128, 128), num_angles=20, impl=self.impl)
        for x in get_random_ellipses_images(3):
            y = ray_trafo(x)
            fbp = ray_trafo.fbp(y)
            self.assert_psnr_greater(x, fbp, 20.)

    def test_angles(self):
        """
        Test some properties of ``ray_trafo.angles``.
        """
        num_angles = 20

        ray_trafo = ParallelBeam2DRayTrafo(
                im_shape=(128, 128), num_angles=num_angles, impl=self.impl)
        self.assertTrue(np.allclose(
                ray_trafo.angles,
                np.linspace(0., np.pi, num_angles, endpoint=False)))

        ray_trafo_default_odl_angles = ParallelBeam2DRayTrafo(
                im_shape=(128, 128), num_angles=num_angles,
                first_angle_zero=False, impl=self.impl)
        self.assertTrue(np.allclose(
                ray_trafo_default_odl_angles.angles,
                np.linspace(
                        0.5 * (np.pi / num_angles),
                        np.pi - 0.5 * (np.pi / num_angles),
                        num_angles)))


class Test_get_parallel_beam_2d_matmul_ray_trafo(unittest.TestCase):
    """
    Test the :class:`MatmulRayTrafo` resembling the ODL ray transform.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ray_trafo_kwargs = dict(
                im_shape=(128, 128), num_angles=20,
                impl='astra_cuda' if ASTRA_CUDA_AVAILABLE else 'skimage')
        self.ray_trafo_via_odl = ParallelBeam2DRayTrafo(**self.ray_trafo_kwargs)
        self.ray_trafo_via_matmul = get_parallel_beam_2d_matmul_ray_trafo(
                **self.ray_trafo_kwargs)

    def test_trafo_same_as_via_odl(self):
        """
        Test if forward projection matches the ODL version.
        """
        for x in get_random_ellipses_images(3):
            y_via_odl = self.ray_trafo_via_odl.trafo(x)
            y_via_matmul = self.ray_trafo_via_matmul.trafo(x)
            self.assertTrue(np.allclose(y_via_odl, y_via_matmul))

    def test_trafo_adjoint_same_as_via_odl(self):
        """
        Test if forward projection matches the ODL version (bound is not very
        tight because ``matrix.T`` is used instead of back-projection).
        """
        for x in get_random_ellipses_images(3):
            y = self.ray_trafo_via_matmul(x)
            x_via_odl = self.ray_trafo_via_odl.trafo_adjoint(y)
            x_via_matmul = self.ray_trafo_via_matmul.trafo_adjoint(y)
            self.assertLess((x_via_odl - x_via_matmul).abs().mean(), 0.02 * x_via_odl.mean())

    def test_fbp_same_as_via_odl(self):
        """
        Test if filtered back-projection matches the ODL version
        (bound is tight because ODL is used for the FBP).
        """
        for x in get_random_ellipses_images(3):
            y = self.ray_trafo_via_matmul(x)
            x_via_odl = self.ray_trafo_via_odl.fbp(y)
            x_via_matmul = self.ray_trafo_via_matmul.fbp(y)
            self.assertTrue(np.allclose(x_via_odl, x_via_matmul))


class Test_get_odl_ray_trafo_parallel_beam_2d_matrix(unittest.TestCase):
    """
    Test :func:`get_odl_ray_trafo_parallel_beam_2d_matrix`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        impl = 'astra_cuda' if ASTRA_CUDA_AVAILABLE else 'skimage'
        angular_sub_sampling = 1
        self.ray_trafo_kwargs = dict(im_shape=(128, 128), num_angles=20)
        odl_ray_trafo_full = get_odl_ray_trafo_parallel_beam_2d(
                **self.ray_trafo_kwargs, impl=impl)
        self.odl_ray_trafo = odl.tomo.RayTransform(
                odl_ray_trafo_full.domain,
                odl_ray_trafo_full.geometry[::angular_sub_sampling], impl=impl)
        self.matrix = get_odl_ray_trafo_parallel_beam_2d_matrix(
                **self.ray_trafo_kwargs)

    def test_same_as_matrix_representation(self):
        """
        Test if matrix representation matches that computed via ODL.
        """
        odl_matrix_representation = odl.operator.oputils.matrix_representation(
                self.odl_ray_trafo).reshape(self.matrix.shape)
        self.assertTrue(np.allclose(self.matrix, odl_matrix_representation))

if __name__ == '__main__':
    unittest.main()
