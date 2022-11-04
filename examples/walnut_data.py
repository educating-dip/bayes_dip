"""
Use the walnut data.
"""
import numpy as np
import matplotlib.pyplot as plt
from bayes_dip.data import get_walnut_2d_ground_truth, get_walnut_2d_ray_trafo
from bayes_dip.data.datasets.walnut import get_walnut_2d_observation

DATA_PATH = '../experiments/walnuts/'

angular_sub_sampling = 20  # 1200 -> 60
proj_col_sub_sampling = 6  # 768 -> 128

walnut_kwargs = dict(
        angular_sub_sampling=angular_sub_sampling,
        proj_col_sub_sampling=proj_col_sub_sampling)

ray_trafo = get_walnut_2d_ray_trafo(
        data_path=DATA_PATH, **walnut_kwargs)

observation = get_walnut_2d_observation(
        data_path=DATA_PATH, **walnut_kwargs)

x = get_walnut_2d_ground_truth(data_path=DATA_PATH)

filtbackproj = ray_trafo.fbp(observation[None])[0]

observation_reshaped = observation.squeeze(0)[
        0, np.asarray(ray_trafo.inds_in_flat_projs_per_angle)][None]

plt.subplot(1, 3, 1)
plt.imshow(observation_reshaped.squeeze(0).T, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(x.squeeze(0), cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(filtbackproj.squeeze(0), cmap='gray')
plt.show()
