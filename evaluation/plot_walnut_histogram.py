import os
import yaml
import argparse
from omegaconf import OmegaConf
import torch
from bayes_dip.utils.evaluation_utils import get_abs_diff, get_stddev
from bayes_dip.utils.plot_utils import configure_matplotlib, plot_hist, DEFAULT_COLORS

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_walnut_sample_based_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--include_outer_part', action='store_true', default=False, help='include the outer part of the walnut image (that only contains background)')
parser.add_argument('--do_not_subtract_image_noise_correction', action='store_true', default=False, help='do not subtract the image noise correction term (if any) from the covariance diagonals')
parser.add_argument('--do_not_use_log_yscale', action='store_true', default=False, help='do not use logarithmic scale for y axis')
args = parser.parse_args()

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

def _get_xlim(data):
    return (0, max((d.max() for d in data)) * 1.1)

def _get_ylim(n_list, ylim_min_fct=0.5):
    ylim_min = ylim_min_fct * min(n[n > 0].min() for n in n_list)
    ylim_max = max(n.max() for n in n_list)
    return (ylim_min, ylim_max)

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
        'subtract_image_noise_correction_if_any': not args.do_not_subtract_image_noise_correction,
        }

abs_diff = get_abs_diff(dip_mll_optim_run, **kwargs)
stddev = get_stddev(runs['include_predcp_False'], **kwargs, **stddev_kwargs)
stddev_predcp = get_stddev(runs['include_predcp_True'], **kwargs, **stddev_kwargs)

mask = torch.logical_not(torch.isnan(stddev))
print(f'Using {mask.sum()} pixels.')

data = [d[mask].flatten().numpy() for d in [abs_diff, stddev, stddev_predcp]]
label_list = ['$|x-x^*|$', 'std-dev (MLL)', 'std-dev (TV-MAP)']
color_list = [DEFAULT_COLORS[k] for k in ['abs_diff', 'bayes_dip', 'bayes_dip_predcp']]

yscale = 'linear' if args.do_not_use_log_yscale else 'log'
ax, n_list, _ = plot_hist(
        data=data, label_list=label_list, yscale=yscale, remove_ticks=False, color_list=color_list)
ax.set_xlim(_get_xlim(data))
ax.set_ylim(_get_ylim(n_list))
ax.get_figure().savefig(f'walnut_hist_{yscale}_yscale.pdf')
