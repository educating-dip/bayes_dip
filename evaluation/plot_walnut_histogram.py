import os
import yaml
import argparse
from omegaconf import OmegaConf
from bayes_dip.utils.evaluation_utils import get_abs_diff, get_stddev
from bayes_dip.utils.plot_utils import configure_matplotlib, plot_hist

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_kmnist_exact_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--do_not_subtract_image_noise_correction', action='store_true', default=False, help='do not subtract the image noise correction term from the covariance diagonals')
args = parser.parse_args()

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

def _get_xlim(data):
    return (0, max((d.max() for d in data)))

def _get_ylim(n_list, ylim_min_fct=0.5):
    ylim_min = ylim_min_fct * min(n[n > 0].min() for n in n_list)
    ylim_max = max(n.max() for n in n_list)
    return (ylim_min, ylim_max)

configure_matplotlib()

dip_mll_optim_run = OmegaConf.load(
        os.path.join(runs[f'include_predcp_False'], '.hydra', 'config.yaml')
        ).inference.load_path

kwargs = {'sample_idx': 0, 'outputs_path': args.experiments_outputs_path}

abs_diff = get_abs_diff(dip_mll_optim_run, **kwargs)
stddev = get_stddev(runs[f'include_predcp_False'],
        subtract_image_noise_correction=not args.do_not_subtract_image_noise_correction, **kwargs)
stddev_predcp = get_stddev(runs[f'include_predcp_True'],
        subtract_image_noise_correction=not args.do_not_subtract_image_noise_correction, **kwargs)

data = [d.flatten().numpy() for d in [abs_diff, stddev, stddev_predcp]]
label_list = ['$|x-x^*|$', 'std-dev (MLL)', 'std-dev (TV-MAP)']

ax, n_list, _ = plot_hist(data=data, label_list=label_list, remove_ticks=False)
ax.set_xlim(_get_xlim(data))
ax.set_ylim(_get_ylim(n_list))
ax.get_figure().savefig(f'walnut_hist_log_yscale.pdf')
ax.set_yscale('linear')
ax.get_figure().savefig(f'walnut_hist.pdf')
