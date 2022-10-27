import os
import argparse
from math import ceil
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from bayes_dip.utils.plot_utils import DEFAULT_COLORS, configure_matplotlib
from bayes_dip.utils.evaluation_utils import extract_tensorboard_scalars, find_single_log_file, translate_path

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file_exact', type=str, default='runs_kmnist_exact_dip_mll_optim.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--runs_file_approx', type=str, default='runs_kmnist_dip_mll_optim.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--angles', type=int, default=20)
parser.add_argument('--noise', type=float, default=0.05)
parser.add_argument('--num_images', type=int, default=10)
parser.add_argument('--do_not_use_predcp', action='store_true', default=False, help='use the run without PredCP (i.e., use MLL instead of TV-MAP)')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--save_data_to', type=str, default='', help='path to cache the plot data, such that they can be loaded with --load_data_from')
parser.add_argument('--load_data_from', type=str, default='', help='load data cached from a previous run with --save_data_to')
parser.add_argument('--tag_list', type=str, nargs='*', default=[], help='tensorboard tags to plot (e.g. "GPprior_lengthscale_0 GPprior_variance_0 NormalPrior_variance_2 observation_noise_variance"); if empty, plots all hyperparameters of GPprior and NormalPrior priors and the noise variance')
parser.add_argument('--suffix', type=str, default='', help='suffix for the figure filenames (e.g. summarizing the --tag_list)')
parser.add_argument('--rows', type=int, default=1, help='number of subplot rows')
parser.add_argument('--skip_sub_plots', type=int, nargs='*', default=[], help='subplot indices to skip (will be empty subplots)')
parser.add_argument('--legend_pos', type=int, default=-1, help='subplot index to place the legend in')
parser.add_argument('--legend_loc', type=str, default='', help='legend loc argument')
parser.add_argument('--legend_bbox_to_anchor', type=float, nargs='*', default=[], help='legend bbox_to_anchor argument')
parser.add_argument('--hspace', type=float, default=0.275, help='matplotlib\'s "hspace" gridspec_kw')
parser.add_argument('--wspace', type=float, default=0.4, help='matplotlib\'s "wspace" gridspec_kw')
parser.add_argument('--image2highlight', type=int, default=4)

args = parser.parse_args()

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file_exact, 'r') as f:
    runs_exact = yaml.safe_load(f)

with open(args.runs_file_approx, 'r') as f:
    runs_approx = yaml.safe_load(f)

runs_exact = runs_exact[args.noise][args.angles]
runs_approx = runs_approx[args.noise][args.angles]

def collect_kmnist_hyperparams_figure_data(args, runs_exact, runs_approx):
    data = {'scalars_exact': [], 'scalars_approx': []}

    include_predcp = not args.do_not_use_predcp

    for i in range(args.num_images):
        log_file_exact = find_single_log_file(os.path.join(
                translate_path(runs_exact[f'include_predcp_{include_predcp}'], experiment_paths=experiment_paths),
                f'mrglik_optim_{i}'))
        data['scalars_exact'].append(extract_tensorboard_scalars(log_file=log_file_exact))

        log_file_approx = find_single_log_file(os.path.join(
                translate_path(runs_approx[f'include_predcp_{include_predcp}'], experiment_paths=experiment_paths),
                f'mrglik_optim_{i}'))
        data['scalars_approx'].append(extract_tensorboard_scalars(log_file=log_file_approx))

    return data

if args.load_data_from:
    print(f'loading data from {args.load_data_from}')
    data = torch.load(args.load_data_from)
else:
    data = collect_kmnist_hyperparams_figure_data(args, runs_exact, runs_approx)

if args.save_data_to:
    print(f'saving data to {args.save_data_to}')
    torch.save(data, args.save_data_to)

configure_matplotlib()

tag_list = args.tag_list or [
        k[:-len('_scalars')] for k in data['scalars_exact'].keys() if k.endswith('_scalars') and (
                k[:-len('_scalars')].rstrip('0123456789') in (
                        'GPprior_variance_', 'GPprior_lengthscale_', 'NormalPrior_variance_') or
                k[:-len('_scalars')] == 'observation_noise_variance')]

num_rows, num_cols = args.rows, ceil(len(tag_list) / args.rows)

fig, axs = plt.subplots(num_rows, num_cols,
        figsize=(2.25 * num_cols, 2. * num_rows),
        gridspec_kw={'hspace': args.hspace, 'wspace': args.wspace})
axs = np.atleast_1d(axs).flatten()

def get_hyperparam_tex_from_tensor_board_tag(tag):
    title = None
    if tag == 'observation_noise_variance':
        title = '\sigma_y^2'
    elif tag.startswith('GPprior_') or tag.startswith('NormalPrior_'):
        tag_base, idx = tag.rsplit('_', 1)
        idx = int(idx) + 1
        if tag_base == 'GPprior_variance':
            title = f'\sigma_{{{idx}}}^2'
        elif tag_base == 'GPprior_lengthscale':
            title = f'\ell_{{{idx}}}'
        elif tag_base == 'NormalPrior_variance':
            title = f'\sigma_{{1\\times 1,{idx}}}^2'
    return title

for ax, tag in zip(
        axs[[i for i in range(axs.size) if i not in args.skip_sub_plots]], tag_list):
    for i in range(args.num_images):
        ax.plot(
            data['scalars_exact'][i][f'{tag}_steps'], data['scalars_exact'][i][f'{tag}_scalars'],
            label='MLL exact' if args.do_not_use_predcp else 'TV-MAP exact',
            color=DEFAULT_COLORS['bayes_dip' if args.do_not_use_predcp else 'bayes_dip_predcp'],
            alpha=0.05 if i != args.image2highlight else .9,
            linewidth=1 if i != args.image2highlight else 2,
            linestyle='solid' if i != args.image2highlight else 'dashed')
        ax.plot(
            data['scalars_approx'][i][f'{tag}_steps'], data['scalars_approx'][i][f'{tag}_scalars'],
            label='MLL approx.' if args.do_not_use_predcp else 'TV-MAP approx.',
            color='#F155FF' if args.do_not_use_predcp else '#6C3D17',
            alpha=0.05 if i != args.image2highlight else .9,
            linewidth=1 if i != args.image2highlight else 2,
            linestyle='solid' if i != args.image2highlight else 'dotted')
    ax.set_yscale('log')
    ax.set_title(f'${get_hyperparam_tex_from_tensor_board_tag(tag)}$')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='xx-small')
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for i, ax in enumerate(axs):
    if i // num_cols < num_rows - 1 and (i + num_cols) not in args.skip_sub_plots:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel('iterations', fontsize='small')
    if i in args.skip_sub_plots:
        if i != args.legend_pos:
            ax.remove()
        else:
            ax.set_axis_off()
handles, labels = next(
        ax.get_legend_handles_labels() for i, ax in enumerate(axs)
        if i not in args.skip_sub_plots)
legend_kwargs = {}
if args.legend_bbox_to_anchor:
    legend_kwargs['bbox_to_anchor'] = args.legend_bbox_to_anchor
if args.legend_loc:
    legend_kwargs['loc'] = args.legend_loc
elif args.legend_pos in args.skip_sub_plots:
    legend_kwargs['loc'] = 'center'
axs[args.legend_pos].legend(
        [handles[args.image2highlight*2], handles[args.image2highlight*2 + 1]],
        [labels[args.image2highlight*2], labels[args.image2highlight*2 + 1]],
        **legend_kwargs
    )

include_predcp = not args.do_not_use_predcp
suffix = '_' + args.suffix if args.suffix and not args.suffix.startswith('_') else args.suffix
fig.savefig(f'kmnist_hyperparams_approx_vs_exact_include_predcp_{include_predcp}{suffix}.pdf', bbox_inches='tight', pad_inches=0.)
fig.savefig(f'kmnist_hyperparams_approx_vs_exact_include_predcp_{include_predcp}{suffix}.png', bbox_inches='tight', pad_inches=0., dpi=600)
