"""
Tests for :module:`bayes_dip.data.trafo.parallel_beam_2d_ray_trafo`.
"""

import pytest
from skimage.metrics import peak_signal_noise_ratio
import odl
import numpy as np
try:
    import astra
except ImportError:
    ASTRA_AVAILABLE = False
    ASTRA_CUDA_AVAILABLE = False
else:
    ASTRA_AVAILABLE = True
    ASTRA_CUDA_AVAILABLE = astra.use_cuda()
from bayes_dip.data import (
        get_odl_ray_trafo_parallel_beam_2d, ParallelBeam2DRayTrafo,
        get_odl_ray_trafo_parallel_beam_2d_matrix,
        get_parallel_beam_2d_matmul_ray_trafo)
from bayes_dip.utils.test_utils import get_random_ellipses_images

def assert_psnr_greater(x, reco, min_psnr, show_on_failure=True):
    """
    Assert PSNR is greater ``min_psnr`` and show plot on failure if
    ``self.show_on_failure``.
    """
    x = x.cpu().numpy().squeeze()
    reco = reco.cpu().numpy().squeeze()
    psnr = peak_signal_noise_ratio(x, reco, data_range=x.max() - x.min())
    try:
        assert psnr > min_psnr
    except AssertionError:
        if show_on_failure:
            import matplotlib.pyplot as plt
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

@pytest.fixture(scope='session')
def ray_trafo_kwargs():
    kwargs = dict(
            im_shape=(32, 32), num_angles=10,
            impl=(('astra_cuda' if ASTRA_CUDA_AVAILABLE else 'astra_cpu')
                    if ASTRA_AVAILABLE else 'skimage')
            )
    return kwargs

@pytest.fixture(scope='session')
def ray_trafo_via_odl(ray_trafo_kwargs):
    return ParallelBeam2DRayTrafo(**ray_trafo_kwargs)

@pytest.fixture(scope='session')
def ray_trafo_via_matmul(ray_trafo_kwargs):
    return get_parallel_beam_2d_matmul_ray_trafo(**ray_trafo_kwargs)

## tests for ParallelBeam2DRayTrafo

def test_parallel_beam_2d_fbp(ray_trafo_kwargs, ray_trafo_via_odl):
    """
    Test if ``ray_trafo.fbp(ray_trafo(x))`` approx. matches ``x`` for
    :class:`ParallelBeam2DRayTrafo`.
    """
    for x in get_random_ellipses_images(3, im_shape=ray_trafo_kwargs['im_shape']):
        y = ray_trafo_via_odl(x)
        fbp = ray_trafo_via_odl.fbp(y)
        assert_psnr_greater(x, fbp, 19.)

def test_parallel_beam_2d_angles(ray_trafo_kwargs, ray_trafo_via_odl):
    """
    Test some properties of ``ray_trafo.angles`` for :class:`ParallelBeam2DRayTrafo`.
    """
    num_angles = len(ray_trafo_via_odl.angles)

    assert np.allclose(
            ray_trafo_via_odl.angles,
            np.linspace(0., np.pi, num_angles, endpoint=False))

    ray_trafo_default_odl_angles = ParallelBeam2DRayTrafo(
            **ray_trafo_kwargs, first_angle_zero=False)
    assert np.allclose(
            ray_trafo_default_odl_angles.angles,
            np.linspace(
                    0.5 * (np.pi / num_angles),
                    np.pi - 0.5 * (np.pi / num_angles),
                    num_angles))


## tests for get_parallel_beam_2d_matmul_ray_trafo

def test_trafo_same_as_via_odl(ray_trafo_kwargs, ray_trafo_via_odl, ray_trafo_via_matmul):
    """
    Test if forward projection via matmul matches the ODL version.
    """
    for x in get_random_ellipses_images(3, im_shape=ray_trafo_kwargs['im_shape']):
        y_via_odl = ray_trafo_via_odl.trafo(x)
        y_via_matmul = ray_trafo_via_matmul.trafo(x)
        assert np.allclose(y_via_odl, y_via_matmul)

def test_trafo_adjoint_same_as_via_odl(ray_trafo_kwargs, ray_trafo_via_odl, ray_trafo_via_matmul):
    """
    Test if forward projection via matmul matches the ODL version (bound is not very
    tight because ``matrix.T`` is used instead of back-projection).
    """
    for x in get_random_ellipses_images(3, im_shape=ray_trafo_kwargs['im_shape']):
        y = ray_trafo_via_matmul(x)
        x_via_odl = ray_trafo_via_odl.trafo_adjoint(y)
        x_via_matmul = ray_trafo_via_matmul.trafo_adjoint(y)
        assert (x_via_odl - x_via_matmul).abs().mean() < 0.02 * x_via_odl.mean()

def test_fbp_same_as_via_odl(ray_trafo_kwargs, ray_trafo_via_odl, ray_trafo_via_matmul):
    """
    Test if filtered back-projection matches the ODL version
    (bound is tight because ODL is used for the FBP).
    """
    for x in get_random_ellipses_images(3, im_shape=ray_trafo_kwargs['im_shape']):
        y = ray_trafo_via_matmul(x)
        x_via_odl = ray_trafo_via_odl.fbp(y)
        x_via_matmul = ray_trafo_via_matmul.fbp(y)
        assert np.allclose(x_via_odl, x_via_matmul)


# tests for get_odl_ray_trafo_parallel_beam_2d_matrix

def test_get_odl_ray_trafo_parallel_beam_2d_matrix(ray_trafo_kwargs):
    angular_sub_sampling = 1
    odl_ray_trafo_full = get_odl_ray_trafo_parallel_beam_2d(
            **ray_trafo_kwargs)
    odl_ray_trafo = odl.tomo.RayTransform(
            odl_ray_trafo_full.domain,
            odl_ray_trafo_full.geometry[::angular_sub_sampling], impl=ray_trafo_kwargs['impl'])
    matrix = get_odl_ray_trafo_parallel_beam_2d_matrix(
            **ray_trafo_kwargs)

    # Test if matrix representation matches that computed via ODL.
    odl_matrix_representation = odl.operator.oputils.matrix_representation(
            odl_ray_trafo).reshape(matrix.shape)
    assert np.allclose(matrix, odl_matrix_representation)
