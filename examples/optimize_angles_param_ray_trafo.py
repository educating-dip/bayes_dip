"""
Optimize angles to maximize TV of 1D projection in fan beam geometry.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from bayes_dip.data import (
        RectanglesDataset,
        get_mnist_testset, get_kmnist_testset,
        ParamFanBeam2DRayTrafo)

DATA = 'rectangles'
# DATA = 'mnist'
# DATA = 'kmnist'

# images
if DATA == 'rectangles':
    im_shape = (128, 128)
    image_dataset = RectanglesDataset(shape=im_shape)
elif DATA == 'mnist':
    im_shape = (28, 28)
    image_dataset = get_mnist_testset()
elif DATA == 'kmnist':
    im_shape = (28, 28)
    image_dataset = get_kmnist_testset()
else:
    raise ValueError

# for i in range(3):
#     x = image_dataset[i]
# # for i, x in zip(range(3), image_dataset):
#     plt.imshow(x.squeeze(0), cmap='gray')
#     plt.show()

# simulated data
obs_shape = (2, 183)
angles = torch.from_numpy(np.linspace(0., 2.*np.pi, num=obs_shape[0], endpoint=False, dtype=np.float32)).requires_grad_()
scale = torch.tensor(1.)
d_source = torch.tensor(1000.)
ray_trafo_kwargs = {'im_shape': im_shape, 'obs_shape': obs_shape, 'scale': scale, 'd_source': d_source, 'filter_type': 'hann'}

ray_trafo = ParamFanBeam2DRayTrafo(angles=angles, **ray_trafo_kwargs)


x = image_dataset[1]  # use second image as it has preferential direction


def objective(y):
    return (y[..., 1:] - y[..., :-1]).abs().sum(dim=-1)


x = x[None].cuda()
ray_trafo.cuda()

# just for plotting, not for the optimization
grid = np.linspace(0., 2.*np.pi, num=1000)
objective_on_grid = np.asarray([objective(ParamFanBeam2DRayTrafo(angles=torch.tensor([float(angle)]), **ray_trafo_kwargs).cuda()(x)).item() for angle in grid])

steps = 1000
optimizer = torch.optim.Adam([ray_trafo._angles], lr=1e-2, betas=[0.999, 0.999])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.)
# scheduler = None
reg_noise = 0.001

with tqdm(range(steps)) as pbar:
    for i in pbar:
        optimizer.zero_grad()
        y = ray_trafo(x)
        mean_obj = objective(y).mean()
        loss = -mean_obj
        loss.backward()
        if reg_noise:
            ray_trafo._angles.grad += torch.randn_like(ray_trafo._angles.grad) * reg_noise
        optimizer.step()
        postfix = {'mean obj': f'{mean_obj.item():.3f}'}
        if scheduler is not None:
            postfix.update({'lr': f'{scheduler.get_lr()[0]:.2e}'})
            scheduler.step()
        pbar.set_postfix(postfix)

final_angles = np.mod(ray_trafo.angles, 2.*np.pi)
with torch.no_grad():
    final_y = ray_trafo(x)
    final_objectives = objective(final_y).squeeze(0).squeeze(0).cpu().numpy()
    print(f'Final objective: {final_objectives}')

x_np = x.squeeze(0).squeeze(0).detach().cpu().numpy()
final_y_np = final_y.squeeze(0).squeeze(0).detach().cpu().numpy()

plt.subplot(1, 3, 1)
plt.imshow(x_np, cmap='gray')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.plot(grid, objective_on_grid)
plt.scatter(final_angles, final_objectives)
plt.subplot(1, 3, 3)
plt.plot(final_y_np[0])
plt.show()
