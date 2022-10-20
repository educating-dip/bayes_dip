import os
import yaml
import argparse
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from bayes_dip.data.datasets.walnut import get_walnut_2d_inner_part_defined_by_patch_size
from bayes_dip.utils.evaluation_utils import get_abs_diff, get_ground_truth, get_stddev, translate_path
from bayes_dip.utils.plot_utils import configure_matplotlib, plot_image, add_inset

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_walnut_sample_based_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--do_not_use_predcp', action='store_true', default=False, help='use the run without PredCP (i.e., use MLL instead of TV-MAP)')
parser.add_argument('--include_outer_part', action='store_true', default=False, help='include the outer part of the walnut image (that only contains background)')
parser.add_argument('--define_inner_part_by_patch_size', type=int, default=1, help='patch size defining the effective inner part (due to not necessarily aligned patches)')
parser.add_argument('--do_not_subtract_image_noise_correction', action='store_true', default=False, help='do not subtract the image noise correction term (if any) from the covariance diagonals')
parser.add_argument('--save_data_to', type=str, default='', help='path to cache the plot data, such that they can be loaded with --load_data_from')
parser.add_argument('--load_data_from', type=str, default='', help='load data cached from a previous run with --save_data_to')
args = parser.parse_args()

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)


def collect_walnut_mini_figure_data(args, runs):
    data = {}


    dip_mll_optim_run = OmegaConf.load(os.path.join(
            translate_path(runs['include_predcp_False'], experiment_paths=experiment_paths),
            '.hydra', 'config.yaml')
            ).inference.load_path

    kwargs = {
        'sample_idx': 0,
        'experiment_paths': experiment_paths,
        }
    stddev_kwargs = {
        'patch_idx_list': None if args.include_outer_part else 'walnut_inner',
        'subtract_image_noise_correction_if_any': not args.do_not_subtract_image_noise_correction,
        }

    data['ground_truth'] = get_ground_truth(dip_mll_optim_run, **kwargs)

    print('collecting bayes_dip data')
    data['abs_diff'] = get_abs_diff(dip_mll_optim_run, **kwargs)
    data['stddev'] = get_stddev(
            runs[f'include_predcp_{not args.do_not_use_predcp}'], **kwargs, **stddev_kwargs)


    data['mask'] = torch.logical_not(torch.isnan(data['stddev']))
    print(f'Using {data["mask"].sum()} pixels.')

    slice_0, slice_1 = (
            (slice(0, data['ground_truth'].shape[0]), slice(0, data['ground_truth'].shape[1]))
            if args.include_outer_part else
            get_walnut_2d_inner_part_defined_by_patch_size(args.define_inner_part_by_patch_size))
    assert data['mask'].sum() == (slice_0.stop - slice_0.start) * (slice_1.stop - slice_1.start)
    data['slice_0'], data['slice_1'] = slice_0, slice_1


    return data


if args.load_data_from:
    print(f'loading data from {args.load_data_from}')
    data = torch.load(args.load_data_from)
else:
    data = collect_walnut_mini_figure_data(args, runs)

if args.save_data_to:
    print(f'saving data to {args.save_data_to}')
    torch.save(data, args.save_data_to)


configure_matplotlib()


fig, ax = plt.subplots(figsize=(4.5, 2.25), gridspec_kw={'left': 0., 'right': 0.5})


ground_truth = data['ground_truth']
abs_diff = data['abs_diff']

# nan parts black
stddev = data['stddev'].clone()
stddev[torch.logical_not(data['mask'])] = 0.

slice_0, slice_1 = data['slice_0'], data['slice_1']

plot_image(fig, ax, ground_truth[slice_0, slice_1], vmin=0.)
rect = [240-slice_0.start, 230-slice_0.start, 52, 104]
ax_abs_error = add_inset(fig, ax, abs_diff[slice_0, slice_1], [1.01, 0.505, 0.99, 0.495], rect, vmin=None, vmax=None, interpolation='none', frame_color='#aa0000', frame_path=[[0., 1.], [0., 0.], [1., 0.], [1., 1.], [0., 1.]], clip_path_closing=[], mark_in_orig=True)
ax_std = add_inset(fig, ax, stddev[slice_0, slice_1], [1.01, 0., 0.99, 0.495], rect, vmin=None, vmax=None, interpolation='none', frame_color='#aa0000', frame_path=[[0., 1.], [0., 0.], [1., 0.], [1., 1.], [0., 1.]], clip_path_closing=[], mark_in_orig=False)
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
