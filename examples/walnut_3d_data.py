"""
Use the walnut 3d data.
"""
import matplotlib.pyplot as plt
from bayes_dip.data import (
        get_walnut_3d_ground_truth, get_walnut_3d_ray_trafo, get_walnut_3d_observation)

DATA_PATH = '../experiments/walnuts/'

angular_sub_sampling = 60  # 1200 -> 20
proj_row_sub_sampling = 3  # 972 -> 324
proj_col_sub_sampling = 3  # 768 -> 256
vol_down_sampling = 3  # 501 -> 167

walnut_kwargs = dict(
        angular_sub_sampling=angular_sub_sampling,
        proj_row_sub_sampling=proj_row_sub_sampling,
        proj_col_sub_sampling=proj_col_sub_sampling)

ray_trafo = get_walnut_3d_ray_trafo(
        data_path=DATA_PATH, vol_down_sampling=vol_down_sampling, **walnut_kwargs)

observation = get_walnut_3d_observation(
        data_path=DATA_PATH, **walnut_kwargs)

x = get_walnut_3d_ground_truth(
        data_path=DATA_PATH, vol_down_sampling=vol_down_sampling)

filtbackproj = ray_trafo.fbp(observation[None])[0]

mid_proj_row = observation.shape[1] // 2
mid_vol_row = x.shape[1] // 2

plt.subplot(1, 3, 1)
plt.imshow(observation.squeeze(0)[mid_proj_row, :, :].T, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(x.squeeze(0)[mid_vol_row, :, :], cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(filtbackproj.squeeze(0)[mid_vol_row, :, :], cmap='gray')
plt.show()
