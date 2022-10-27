import os
import yaml
import argparse
from math import ceil
import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from bayes_dip.utils.evaluation_utils import (
        get_density_data, compute_log_prob_for_patch_size_from_cov, get_ground_truth, get_recon,
        translate_path)
from bayes_dip.utils.plot_utils import configure_matplotlib

def float_or_none(v):
    return None if v == 'None' else float(v)

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file_exact', type=str, default='runs_kmnist_exact_density.yaml', help='path of yaml file containing hydra output directory names, this script will re-evaluate these exact density runs for different patch sizes')
parser.add_argument('--runs_folder_sample_based', type=str, default='runs_kmnist_sample_based_density_varying_num_samples', help='folder containing yaml files (named "patch_size_??_num_samples_??.yaml") containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--noise', type=float, default=0.05)
parser.add_argument('--angles', type=int, default=20)
parser.add_argument('--num_images', type=int, default=10)
parser.add_argument('--num_subplots', type=int, default=1)
parser.add_argument('--ylim_min', type=float_or_none, nargs='*', default=[])
parser.add_argument('--legend_pos', type=int, default=-1, help='subplot index to place the symbol legend in')
parser.add_argument('--save_data_to', type=str, default='', help='path to cache the plot data, such that they can be loaded with --load_data_from')
parser.add_argument('--load_data_from', type=str, default='', help='load data cached from a previous run with --save_data_to')
args = parser.parse_args()

INCLUDE_PREDCP = False

PATCH_SIZES = [1, 2, 4, 7, 14, 28]
NUM_SAMPLES_LIST = [2**i for i in range(5, 15)]

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file_exact, 'r') as f:
    runs_exact = yaml.safe_load(f)

runs_sample_based = {}
for patch_size in PATCH_SIZES:
    runs_sample_based.setdefault(patch_size, {})
    for num_samples in NUM_SAMPLES_LIST:
        with open(os.path.join(args.runs_folder_sample_based, f'patch_size_{patch_size}_num_samples_{num_samples}.yaml'), 'r') as f:
            runs_sample_based[patch_size][num_samples] = yaml.safe_load(f)


def collect_kmnist_sample_based_vs_exact_figure_data(args, runs_exact, runs_sample_based):
    data = {'log_prob_stats': {}}

    noise, angles = args.noise, args.angles

    data['log_prob_stats'][f'include_predcp_{INCLUDE_PREDCP}'] = {}
    for patch_size in PATCH_SIZES:
        data['log_prob_stats'][f'include_predcp_{INCLUDE_PREDCP}'][patch_size] = {}

        run_path_exact = runs_exact[noise][angles][f'include_predcp_{INCLUDE_PREDCP}']
        dip_mll_optim_run = OmegaConf.load(os.path.join(
                translate_path(run_path_exact, experiment_paths=experiment_paths),
                '.hydra', 'config.yaml')).inference.load_path
        log_probs_exact = []
        for sample_idx in range(args.num_images):
            density_data, is_exact = get_density_data(
                    run_path_exact, sample_idx=sample_idx, experiment_paths=experiment_paths)
            assert is_exact
            recon = get_recon(
                    dip_mll_optim_run, sample_idx=sample_idx, experiment_paths=experiment_paths)
            ground_truth = get_ground_truth(
                    dip_mll_optim_run, sample_idx=sample_idx, experiment_paths=experiment_paths)
            log_prob = compute_log_prob_for_patch_size_from_cov(
                    cov=density_data['cov'], patch_size=patch_size, mean=recon,
                    ground_truth=ground_truth)
            log_probs_exact.append(log_prob)
        data['log_prob_stats'][f'include_predcp_{INCLUDE_PREDCP}'][patch_size].update({
            'mean_exact': np.mean(log_probs_exact).item(),
            'stderr_exact': (np.std(log_probs_exact) / np.sqrt(len(log_probs_exact))).item(),
        })

        for num_samples in NUM_SAMPLES_LIST:
            log_probs_sample_based = []
            run_path_sample_based = runs_sample_based[patch_size][num_samples][noise][angles][f'include_predcp_{INCLUDE_PREDCP}']
            for sample_idx in range(args.num_images):
                density_data, is_exact = get_density_data(
                        run_path_sample_based,
                        sample_idx=sample_idx, experiment_paths=experiment_paths)
                assert not is_exact
                cfg = OmegaConf.load(os.path.join(
                        translate_path(run_path_sample_based, experiment_paths=experiment_paths),
                        '.hydra', 'config.yaml'))
                assert cfg.inference.patch_size == patch_size
                assert cfg.inference.patch_idx_list is None
                log_probs_sample_based.append(density_data['log_prob'])
            data['log_prob_stats'][f'include_predcp_{INCLUDE_PREDCP}'][patch_size].update({
                f'mean_sample_based_num_samples_{num_samples}': np.mean(log_probs_sample_based).item(),
                f'stderr_sample_based_num_samples_{num_samples}': (np.std(log_probs_sample_based) / np.sqrt(len(log_probs_sample_based))).item(),
            })

    return data


if args.load_data_from:
    print(f'loading data from {args.load_data_from}')
    data = torch.load(args.load_data_from)
else:
    data = collect_kmnist_sample_based_vs_exact_figure_data(args, runs_exact, runs_sample_based)

if args.save_data_to:
    print(f'saving data to {args.save_data_to}')
    torch.save(data, args.save_data_to)


configure_matplotlib()

fig, axs = plt.subplots(1, args.num_subplots, figsize=(4 * args.num_subplots, 3))
axs = np.atleast_1d(axs)

max_num_patch_sizes_per_ax = ceil(len(PATCH_SIZES) / args.num_subplots)
ax_dict = {
        patch_size: axs[i//max_num_patch_sizes_per_ax] for i, patch_size in enumerate(PATCH_SIZES)}

color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

xlim = min(NUM_SAMPLES_LIST), max(NUM_SAMPLES_LIST)

handles_exact = {ax: {} for ax in axs.flat}
handles_sample_based = {ax: {} for ax in axs.flat}
for patch_size, color in zip(PATCH_SIZES, color_list):
    ax = ax_dict[patch_size]
    stats = data['log_prob_stats'][f'include_predcp_{INCLUDE_PREDCP}'][patch_size]
    # exact
    handles_exact[ax][patch_size] = ax.hlines(
            stats[f'mean_exact'],
            xmin=xlim[0], xmax=xlim[1],
            color=color, linestyle='solid')
    ax.fill_between(
            [xlim[0], xlim[1]],
            stats[f'mean_exact'] - stats[f'stderr_exact'],
            stats[f'mean_exact'] + stats[f'stderr_exact'],
            color=color, alpha=0.1)
    # sample based
    handles_sample_based[ax][patch_size], = ax.plot(NUM_SAMPLES_LIST,
            [stats[f'mean_sample_based_num_samples_{num_samples}']
                for num_samples in NUM_SAMPLES_LIST],
            color=color, linestyle='dashed')
    ax.fill_between(
            NUM_SAMPLES_LIST,
            [stats[f'mean_sample_based_num_samples_{num_samples}'] - stats[
                    f'stderr_sample_based_num_samples_{num_samples}']
                for num_samples in NUM_SAMPLES_LIST],
            [stats[f'mean_sample_based_num_samples_{num_samples}'] + stats[
                    f'stderr_sample_based_num_samples_{num_samples}']
                for num_samples in NUM_SAMPLES_LIST],
            color=color, alpha=0.1)
for i, ax in enumerate(axs.flat):
    if i == 0:
        ax.set_ylabel('test log-likelihood')
    ax.set_xlabel('number of samples')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_ticks(NUM_SAMPLES_LIST)
    ax.set_xlim(xlim)
    ax.set_ylim((args.ylim_min[i] if len(args.ylim_min) > i else None), None)
    ax.grid(alpha=0.3)
for ax in axs.flat:
    lgd = ax.legend(
            list(handles_sample_based[ax].values())[::-1],
            [f'patch size ${patch_size}\\times {patch_size}$'
             for patch_size in list(handles_sample_based[ax].keys())][::-1],
            loc='lower right',
            )
    ax.add_artist(lgd)
lgd_kind = axs[args.legend_pos].legend(
        [list(handles_exact[axs[args.legend_pos]].values())[0],
         list(handles_sample_based[axs[args.legend_pos]].values())[0]],
        ['exact', 'sampling'],
        loc='center right')
for h in lgd_kind.legendHandles:
    h.set_color('black')

fig.savefig(f'kmnist_sample_based_vs_exact.pdf', bbox_inches='tight', pad_inches=0.)
fig.savefig(f'kmnist_sample_based_vs_exact.png', bbox_inches='tight', pad_inches=0., dpi=600)
