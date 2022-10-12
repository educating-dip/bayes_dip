import os
import yaml
import argparse
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from omegaconf import OmegaConf
from bayes_dip.data.datasets.walnut import get_walnut_2d_inner_part_defined_by_patch_size, VOL_SZ
from bayes_dip.utils.evaluation_utils import get_abs_diff, get_ground_truth, get_stddev
from bayes_dip.utils.plot_utils import configure_matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_walnut_sample_based_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--do_not_use_predcp', action='store_true', default=False, help='use the run without PredCP (i.e., use MLL instead of TV-MAP)')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--include_outer_part', action='store_true', default=False, help='include the outer part of the walnut image (that only contains background)')
parser.add_argument('--define_inner_part_by_patch_size', type=int, default=1, help='patch size defining the effective inner part (due to not necessarily aligned patches)')
parser.add_argument('--do_not_subtract_image_noise_correction', action='store_true', default=False, help='do not subtract the image noise correction term from the covariance diagonals')
args = parser.parse_args()

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

dic = {
    'images': {
        'metrics': {
            'pos': [424, 436],
            'kwargs': {'fontsize': 5, 'color': '#ffffff'},
        },
        'log_lik': {
            'pos': [424, 436],
            'kwargs': {'fontsize': 5, 'color': '#ffffff'},
        },
        'insets': [
                {
                 'rect': [190, 345, 133, 70],
                 'axes_rect': [0.8, 0.62, 0.2, 0.38],
                 'frame_path': [[0., 1.], [0., 0.], [1., 0.]],
                 'clip_path_closing': [[1., 1.]],
                },
                {
                 'rect': [220, 200, 55, 65],
                 'axes_rect': [0., 0., 0.39, 0.33],
                 'frame_path': [[1., 0.], [1., 0.45], [0.3, 1.], [0., 1.]],
                 'clip_path_closing': [[0., 0.]],
                },
        ],
    },
    'hist': {
            'num_bins': 25,
            'num_k_retained': 5,
            'opacity': [0.3, 0.3, 0.3],
            'zorder': [10, 5, 0],
            'color': ['#e63946', '#ee9b00', '#606c38'],
            'linewidth': 0.75,
            },
    'qq': {
            'zorder': [10, 5, 0],
            'color': ['blue', 'red'],
    }
}

def create_image_plot(fig, ax, image, title='', vmin=None, vmax=None, cmap='gray', interpolation='none', insets=False, insets_mark_in_orig=False, colorbar=False):
    im = ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
    ax.set_title(title)
    if insets:
        for inset_spec in dic['images']['insets']:
            add_inset(fig, ax, image, **inset_spec, vmin=vmin, vmax=vmax, cmap=cmap, mark_in_orig=insets_mark_in_orig)
    if colorbar:
        cb = add_colorbar(fig, ax, im)
        if colorbar == 'invisible':
            cb.ax.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def add_colorbar(fig, ax, im):
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="4%", pad="2%")
    cb = fig.colorbar(im, cax=cax)
    cax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    return cb

def add_inset(fig, ax, image, axes_rect, rect, cmap='gray', vmin=None, vmax=None, interpolation='none', frame_color='#aa0000', frame_path=None, clip_path_closing=None, mark_in_orig=False, origin='upper'):
    ip = InsetPosition(ax, axes_rect)
    axins = matplotlib.axes.Axes(fig, [0., 0., 1., 1.])
    axins.set_axes_locator(ip)
    fig.add_axes(axins)
    slice0 = slice(rect[0], rect[0]+rect[2])
    slice1 = slice(rect[1], rect[1]+rect[3])
    inset_image = image[slice0, slice1]
    inset_image_handle = axins.imshow(
            inset_image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.patch.set_visible(False)
    for spine in axins.spines.values():
        spine.set_visible(False)
    if frame_path is None:
        frame_path = [[0., 0.], [1., 0.], [0., 1.], [1., 1]]
    if frame_path:
        frame_path_closed = frame_path + (clip_path_closing if clip_path_closing is not None else [])
        if mark_in_orig:
            scalex, scaley = rect[3], rect[2]
            offsetx, offsety = rect[1], (image.shape[0]-(rect[0]+rect[2]) if origin == 'upper' else rect[0])
            y_trans = matplotlib.transforms.Affine2D().scale(1., -1.).translate(0., image.shape[0]-1) if origin == 'upper' else matplotlib.transforms.IdentityTransform()
            trans_data = matplotlib.transforms.Affine2D().scale(scalex, scaley).translate(offsetx, offsety) + y_trans + ax.transData
            x, y = [*zip(*(frame_path_closed + [frame_path_closed[0]]))]
            ax.plot(x, y, transform=trans_data, color=frame_color, linestyle='dashed', linewidth=1.)
        axins.plot(
                *np.array(frame_path).T,
                transform=axins.transAxes,
                color=frame_color,
                solid_capstyle='butt')
        inset_image_handle.set_clip_path(matplotlib.path.Path(frame_path_closed),
                transform=axins.transAxes)
        inset_image_handle.set_clip_on(True)
    return axins

configure_matplotlib()

dip_mll_optim_run = OmegaConf.load(
        os.path.join(runs['include_predcp_False'], '.hydra', 'config.yaml')
        ).inference.load_path

kwargs = {
        'sample_idx': 0,
        'experiment_paths': experiment_paths,
        }
stddev_kwargs = {
        'patch_idx_list': None if args.include_outer_part else 'walnut_inner',
        'subtract_image_noise_correction': not args.do_not_subtract_image_noise_correction,
        }

ground_truth = get_ground_truth(dip_mll_optim_run, **kwargs)
abs_diff = get_abs_diff(dip_mll_optim_run, **kwargs)
stddev = get_stddev(runs['include_predcp_False'], **kwargs, **stddev_kwargs)
stddev_predcp = get_stddev(runs['include_predcp_True'], **kwargs, **stddev_kwargs)

mask = torch.logical_not(torch.isnan(stddev))
print(f'Using {mask.sum()} pixels.')

slice_0, slice_1 = (
    (slice(0, ground_truth.shape[0]), slice(0, ground_truth.shape[1])) if args.include_outer_part
    else get_walnut_2d_inner_part_defined_by_patch_size(
            args.define_inner_part_by_patch_size))
assert mask.sum() == (slice_0.stop - slice_0.start) * (slice_1.stop - slice_1.start)

color_abs_error = '#e63946'
color_map = '#5a6c17'

# TODO also subtract eps?

fig, ax = plt.subplots(figsize=(3, 1.5), gridspec_kw={'left': 0., 'right': 0.5})

# nan parts black
stddev[torch.logical_not(mask)] = 0.
stddev_predcp[torch.logical_not(mask)] = 0.

stddev_for_plot = stddev if args.do_not_use_predcp else stddev_predcp

create_image_plot(fig, ax, ground_truth[slice_0, slice_1], vmin=0.)
rect = [120-slice_0.start, 230-slice_0.start, 52, 104]
ax_abs_error = add_inset(fig, ax, abs_diff[slice_0, slice_1], [1.01, 0.505, 0.99, 0.495], rect, vmin=None, vmax=None, interpolation='none', frame_color='#aa0000', frame_path=[[0., 1.], [0., 0.], [1., 0.], [1., 1.], [0., 1.]], clip_path_closing=[], mark_in_orig=True)
ax_std = add_inset(fig, ax, stddev_for_plot[slice_0, slice_1], [1.01, 0., 0.99, 0.495], rect, vmin=None, vmax=None, interpolation='none', frame_color='#aa0000', frame_path=[[0., 1.], [0., 0.], [1., 0.], [1., 1.], [0., 1.]], clip_path_closing=[], mark_in_orig=False)
ax_abs_error_twin = ax_abs_error.twinx()
ax_std_twin = ax_std.twinx()
ax_abs_error_twin.set_xticks([])
ax_abs_error_twin.set_yticks([])
ax_std_twin.set_xticks([])
ax_std_twin.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
for spine in ax_abs_error_twin.spines.values():
    spine.set_visible(False)
for spine in ax_std_twin.spines.values():
    spine.set_visible(False)
ax.set_ylabel('original image', labelpad=2)
ax_abs_error_twin.set_ylabel('error', rotation=-90, labelpad=9)
ax_std_twin.set_ylabel('std-dev', rotation=-90, labelpad=9)

fig.savefig(f'walnut_mini_include_predcp_{not args.do_not_use_predcp}.pdf', bbox_inches='tight', pad_inches=0.)
fig.savefig(f'walnut_mini_include_predcp_{not args.do_not_use_predcp}.png', bbox_inches='tight', pad_inches=0., dpi=600)
