"""
Generate rectangles images, observations, and filtered back-projections using
:class:`bayes_dip.data.SimulatedDataset`.
"""

import matplotlib.pyplot as plt
from bayes_dip.data import (
        RectanglesDataset, SimulatedDataset,
        ParallelBeam2DRayTrafo, get_parallel_beam_2d_matmul_ray_trafo)

# images
im_shape = (128, 128)
image_dataset = RectanglesDataset(shape=im_shape)

# for i in range(3):
#     x = image_dataset[i]
# # for i, x in zip(range(3), image_dataset):
#     plt.imshow(x.squeeze(0), cmap='gray')
#     plt.show()

USE_MATMUL_RAY_TRAFO = False  # True

# simulated data
if USE_MATMUL_RAY_TRAFO:
    ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(im_shape, num_angles=20)
else:
    ray_trafo = ParallelBeam2DRayTrafo(im_shape, num_angles=20)

dataset = SimulatedDataset(image_dataset, ray_trafo, white_noise_rel_stddev=0.05)

for i in range(3):
    observation, x, filtbackproj = dataset[i]
# for i, (observation, x, filtbackproj) in zip(range(3), dataset):
    plt.subplot(1, 3, 1)
    plt.imshow(observation.T.squeeze(0), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(x.squeeze(0), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(filtbackproj.squeeze(0), cmap='gray')
    plt.show()
