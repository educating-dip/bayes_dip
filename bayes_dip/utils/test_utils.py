"""
Utilities for testing.
"""

from typing import Generator, Tuple
import torch
import numpy as np
from odl import uniform_discr
from odl.phantom import ellipsoid_phantom


def get_random_ellipses_images(
        num_images: int = 3,
        im_shape: Tuple[int, int] = (128, 128)
        ) -> Generator[np.ndarray, None, None]:
    """
    num_images : int, optional
        Number of images.
        The default is ``3``.
    im_shape : 2-tuple of int, optional
        Image shape.
        The default is ``(128, 128)``.
    """

    space = uniform_discr([-1., -1.], [1., 1.], im_shape, dtype=np.float32)
    max_n_ellipse = 70
    ellipsoids = np.empty((max_n_ellipse, 6))
    rng = np.random.default_rng(1)
    for _ in range(num_images):
        v = (rng.uniform(-0.4, 1.0, (max_n_ellipse,)))
        a1 = .2 * rng.exponential(1., (max_n_ellipse,))
        a2 = .2 * rng.exponential(1., (max_n_ellipse,))
        x = rng.uniform(-0.9, 0.9, (max_n_ellipse,))
        y = rng.uniform(-0.9, 0.9, (max_n_ellipse,))
        rot = rng.uniform(0., 2 * np.pi, (max_n_ellipse,))
        n_ellipse = min(rng.poisson(40), max_n_ellipse)
        v[n_ellipse:] = 0.
        ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
        image = ellipsoid_phantom(space, ellipsoids)
        image[np.array(image) != 0.] -= np.min(image)
        image /= np.max(image)
        yield torch.from_numpy(np.asarray(image))[None, None]
