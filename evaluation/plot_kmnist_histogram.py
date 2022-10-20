import os
import yaml
import argparse
import torch
from omegaconf import OmegaConf
from bayes_dip.utils.evaluation_utils import get_abs_diff, get_stddev, translate_path
from bayes_dip.utils.plot_utils import configure_matplotlib, plot_hist

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_kmnist_exact_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--noise_list', type=float, nargs='+', default=[0.05, 0.1])
parser.add_argument('--angles_list', type=int, nargs='+', default=[5, 10, 20, 30])
parser.add_argument('--sample_idx', type=int, default=0)
parser.add_argument('--do_not_subtract_image_noise_correction', action='store_true', default=False, help='do not subtract the image noise correction term (if any) from the covariance diagonals')
parser.add_argument('--do_not_use_log_yscale', action='store_true', default=False, help='do not use logarithmic scale for y axis')
parser.add_argument('--save_data_to', type=str, default='', help='path to cache the plot data, such that they can be loaded with --load_data_from')
parser.add_argument('--load_data_from', type=str, default='', help='load data cached from a previous run with --save_data_to')
args = parser.parse_args()

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)


def _get_xlim(data):
    return (0, max((d.max() for d in data)))

def _get_ylim(n_list, ylim_min_fct=0.5):
    ylim_min = ylim_min_fct * min(n[n > 0].min() for n in n_list)
    ylim_max = max(n.max() for n in n_list)
    return (ylim_min, ylim_max)


def collect_kmnist_histogram_figure_data(args, runs):
    data = {}

    for noise in args.noise_list:
        data.setdefault(noise, {})
        for angles in args.angles_list:
            data[noise][angles] = {}

            dip_mll_optim_run = OmegaConf.load(os.path.join(
                    translate_path(runs[noise][angles]['include_predcp_False'],
                            experiment_paths=experiment_paths), '.hydra', 'config.yaml')
                    ).inference.load_path

            kwargs = {'sample_idx': args.sample_idx, 'experiment_paths': experiment_paths}

            data[noise][angles]['abs_diff'] = get_abs_diff(dip_mll_optim_run, **kwargs)
            data[noise][angles]['stddev'] = get_stddev(
                    runs[noise][angles]['include_predcp_False'],
                    subtract_image_noise_correction_if_any=(
                            not args.do_not_subtract_image_noise_correction),
                    **kwargs)
            data[noise][angles]['stddev_predcp'] = get_stddev(
                    runs[noise][angles]['include_predcp_True'],
                    subtract_image_noise_correction_if_any=(
                            not args.do_not_subtract_image_noise_correction),
                    **kwargs)

    return data


if args.load_data_from:
    print(f'loading data from {args.load_data_from}')
    data = torch.load(args.load_data_from)
else:
    data = collect_kmnist_histogram_figure_data(args, runs)

if args.save_data_to:
    print(f'saving data to {args.save_data_to}')
    torch.save(data, args.save_data_to)


configure_matplotlib()

yscale = 'linear' if args.do_not_use_log_yscale else 'log'

for noise in args.noise_list:
    for angles in args.angles_list:
        abs_diff = data[noise][angles]['abs_diff']
        stddev = data[noise][angles]['stddev']
        stddev_predcp = data[noise][angles]['stddev_predcp']

        hist_data = [d.flatten().numpy() for d in [abs_diff, stddev, stddev_predcp]]
        label_list = ['$|\\hat x-x|$', 'std-dev (MLL)', 'std-dev (TV-MAP)']

        ax, n_list, _ = plot_hist(data=hist_data, label_list=label_list, yscale=yscale, remove_ticks=False)
        ax.set_xlim(_get_xlim(hist_data))
        ax.set_ylim(_get_ylim(n_list))
        ax.get_figure().savefig(f'kmnist_hist_noise_{noise}_angles_{angles}_sample_{args.sample_idx}_{yscale}_yscale.pdf')
