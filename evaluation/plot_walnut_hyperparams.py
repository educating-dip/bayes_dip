import os
import argparse
from math import ceil
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from bayes_dip.utils.plot_utils import DEFAULT_COLORS, configure_matplotlib
from bayes_dip.utils.evaluation_utils import extract_tensorboard_scalars, find_single_log_file, translate_path

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_walnut_sample_based_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--save_data_to', type=str, default='', help='path to cache the plot data, such that they can be loaded with --load_data_from')
parser.add_argument('--load_data_from', type=str, default='', help='load data cached from a previous run with --save_data_to')
parser.add_argument('--tag_list', type=str, nargs='*', default=[], help='tensorboard tags to plot (e.g. "GPprior_lengthscale_0 GPprior_variance_0 NormalPrior_variance_2 observation_noise_variance"); if empty, plots all hyperparameters of GPprior and NormalPrior priors and the noise variance')
parser.add_argument('--rows', type=int, default=1, help='number of subplot rows')
parser.add_argument('--legend_pos', type=int, default=-1, help='subplot index to place the legend in')
args = parser.parse_args()

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)


def collect_walnut_hyperparams_figure_data(runs):
    data = {}

    log_file = find_single_log_file(os.path.join(
            translate_path(runs['include_predcp_False'], experiment_paths=experiment_paths),
            'mrglik_optim_0'))
    data['scalars'] = extract_tensorboard_scalars(log_file=log_file)

    log_file_predcp = find_single_log_file(os.path.join(
            translate_path(runs['include_predcp_True'], experiment_paths=experiment_paths),
            'mrglik_optim_0'))
    data['scalars_predcp'] = extract_tensorboard_scalars(log_file=log_file_predcp)

    return data


if args.load_data_from:
    print(f'loading data from {args.load_data_from}')
    data = torch.load(args.load_data_from)
else:
    data = collect_walnut_hyperparams_figure_data(runs)

if args.save_data_to:
    print(f'saving data to {args.save_data_to}')
    torch.save(data, args.save_data_to)


configure_matplotlib()


tag_list = args.tag_list or [
        k[:-len('_scalars')] for k in data['scalars'].keys() if k.endswith('_scalars') and (
                k[:-len('_scalars')].rstrip('0123456789') in (
                        'GPprior_variance_', 'GPprior_lengthscale_', 'NormalPrior_variance_') or
                k[:-len('_scalars')] == 'observation_noise_variance')]


num_rows, num_cols = args.rows, ceil(len(tag_list) / args.rows)


fig, axs = plt.subplots(num_rows, num_cols,
        figsize=(2.25 * num_cols, 2. * num_rows),
        gridspec_kw={'hspace': 0.275, 'wspace': 0.275})
axs = np.atleast_2d(axs)

def get_hyperparam_tex_from_tensor_board_tag(tag):
    title = None
    if tag == 'observation_noise_variance':
        title = '\sigma_y^2'
    elif tag.startswith('GPprior_') or tag.startswith('NormalPrior_'):
        tag_base, idx = tag.rsplit('_', 1)
        idx = int(idx) + 1
        if tag_base == 'GPprior_variance':
            title = f'\sigma_{idx}^2'
        elif tag_base == 'GPprior_lengthscale':
            title = f'\ell_{idx}^2'
        elif tag_base == 'NormalPrior_variance':
            title = f'\sigma_{{1\\times 1,{idx}}}^2'
    return title


for ax, tag in zip(axs.flat, tag_list):
    ax.plot(data['scalars'][f'{tag}_steps'], data['scalars'][f'{tag}_scalars'],
        label='MLL', color=DEFAULT_COLORS['bayes_dip'], alpha=0.9)
    ax.plot(data['scalars_predcp'][f'{tag}_steps'], data['scalars_predcp'][f'{tag}_scalars'],
            label='TV-MAP', color=DEFAULT_COLORS['bayes_dip_predcp'], alpha=0.9)
    ax.set_yscale('log')
    ax.set_title(f'${get_hyperparam_tex_from_tensor_board_tag(tag)}$')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.grid(alpha=0.2)

for ax in axs[:-1, :].flatten():
    ax.tick_params(labelbottom=False)
for ax in axs[-1, :]:
    ax.set_xlabel('iterations', fontsize='small')

axs.flatten()[args.legend_pos].legend()


fig.savefig(f'walnut_hyperparams.pdf', bbox_inches='tight', pad_inches=0.)
fig.savefig(f'walnut_hyperparams.png', bbox_inches='tight', pad_inches=0., dpi=600)
