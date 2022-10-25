import yaml
import argparse
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from bayes_dip.utils.evaluation_utils import get_image_cov
from bayes_dip.utils.plot_utils import configure_matplotlib, plot_hist, DEFAULT_COLORS
from bayes_dip.data.datasets import get_kmnist_testset
from bayes_dip.utils import normalize

def float_or_none(v):
    return None if v == 'None' else float(v)

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_kmnist_exact_dip_mll_optim.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--hmc_priors_data_file', type=str, help='file path of HMC experiment results, e.g. "prior_HMC/prior_samples.pickle"')
parser.add_argument('--num_samples', type=int, default=50)
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--save_data_to', type=str, default='', help='path to cache the plot data, such that they can be loaded with --load_data_from')
parser.add_argument('--load_data_from', type=str, default='', help='load data cached from a previous run with --save_data_to')
args = parser.parse_args()

NOISE = 0.05
ANGLES = 30
NUM_PRIOR_SAMPLES = 500

experiment_paths = {
    'outputs_path': args.experiments_outputs_path,
    'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

def _get_ylim(n_list, ylim_min_fct=0.5):
    ylim_min = ylim_min_fct * min(n[n > 0].min() for n in n_list)
    ylim_max = max(n.max() for n in n_list)
    return (ylim_min, ylim_max)

def plot_sample(ax, sample, title='', **kwargs):
    ax.imshow(sample.squeeze(0), cmap='gray', **kwargs)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def n_tv_entries(side):
    return 2 * (side - 1) * side

def tv(x):
    h_tv = np.abs(np.diff(x, axis=-1, n=1)).sum()
    v_tv = np.abs(np.diff(x, axis=-2, n=1)).sum()
    return h_tv + v_tv

def collect_kmnist_tv_hists_and_samples_from_dists_figure_data(args):
    data = {}

    with open(args.hmc_priors_data_file, 'rb') as file:
        prior_sample_dict = pickle.load(file, encoding='unicode_escape')
    data['samples_from_tv'] = prior_sample_dict['TV_samples']['x']
    data['samples_from_hybrid'] = prior_sample_dict['Hybrid_samples']['x']
    data['samples_from_gauss'] = prior_sample_dict['Gaussian_samples']['x']
    # convert to NCHW tensors and restrict to first NUM_PRIOR_SAMPLES samples
    for key in ('samples_from_tv', 'samples_from_hybrid', 'samples_from_gauss'):
        assert data[key].shape[0] >= NUM_PRIOR_SAMPLES
        data[key] = torch.from_numpy(
                data[key][:NUM_PRIOR_SAMPLES].reshape(NUM_PRIOR_SAMPLES, 1, 28, 28))
    kmnist_testset = get_kmnist_testset()
    data['samples_from_kmnist'] = torch.stack([normalize(kmnist_testset[i]) for i in range(NUM_PRIOR_SAMPLES)])

    for name in ('tv', 'hybrid', 'gauss', 'kmnist'):
        data[f'tv_losses_from_{name}'] = np.array([
                tv(sample).item() / n_tv_entries(sample.shape[-1]) for sample in data[f'samples_from_{name}']])
    return data

def collect_kmnist_samples_from_lin_dip(runs, sample_idx):

    data = {}
    torch.manual_seed(2)
    image_cov = get_image_cov(
            runs[NOISE][ANGLES]['include_predcp_False'],
            sample_idx=sample_idx,
            use_matmul_neural_basis_expansion=True, experiment_paths=experiment_paths)
    image_cov_mat = image_cov.inner_cov(image_cov.neural_basis_expansion._matrix) @ image_cov.neural_basis_expansion._matrix.T
    image_cov_mat[np.diag_indices(image_cov_mat.shape[0])] += 1e-6
    mll_vmax = 2 * image_cov_mat.diag().mean().pow(.5)
    data['samples_from_bayes_dip'] = image_cov.sample(
            num_samples=NUM_PRIOR_SAMPLES).detach().cpu()
    data['mll_vmax'] = mll_vmax.item()

    torch.manual_seed(2)
    image_cov_predcp = get_image_cov(
            runs[NOISE][ANGLES]['include_predcp_True'],
            sample_idx=sample_idx,
            use_matmul_neural_basis_expansion=True, experiment_paths=experiment_paths)
    image_cov_predcp_mat = image_cov_predcp.inner_cov(image_cov_predcp.neural_basis_expansion._matrix) @ image_cov_predcp.neural_basis_expansion._matrix.T
    image_cov_predcp_mat[np.diag_indices(image_cov_predcp_mat.shape[0])] += 1e-6
    map_vmax = 2 * image_cov_predcp_mat.diag().mean().pow(.5)
    data['samples_from_bayes_dip_predcp'] = image_cov_predcp.sample(
            num_samples=NUM_PRIOR_SAMPLES).detach().cpu()
    data['map_vmax'] = map_vmax.item()

    return data

def collect_kmnist_tv_hists_from_lin_dip_samples(args, runs): 

    data = {'bayes_dip':[], 'mll_vmax':[], 'bayes_dip_predcp':[], 'map_vmax':[],
        'tv_losses_from_bayes_dip':[], 'tv_losses_from_bayes_dip_predcp':[], 'tv_losses_from_kmnist':[]}
    for sample_idx in range(args.num_samples):
        data['bayes_dip'].append(
                collect_kmnist_samples_from_lin_dip(runs, sample_idx)['samples_from_bayes_dip'].numpy()
            )
        data['mll_vmax'].append(collect_kmnist_samples_from_lin_dip(runs, sample_idx)['mll_vmax'])
        data['bayes_dip_predcp'].append(collect_kmnist_samples_from_lin_dip(runs, sample_idx)['samples_from_bayes_dip_predcp'].numpy())
        data['map_vmax'].append(collect_kmnist_samples_from_lin_dip(runs, sample_idx)['map_vmax'])
    
    kmnist_testset = get_kmnist_testset()
    data['kmnist'] = torch.stack([normalize(kmnist_testset[i]) for i in range(NUM_PRIOR_SAMPLES)])

    for name in ('bayes_dip', 'bayes_dip_predcp', 'kmnist'):
        for samples in data[name]:
            for sample in samples:
                data[f'tv_losses_from_{name}'].append(tv(np.clip(sample, a_max=1, a_min=0)) / n_tv_entries(sample.shape[-1]))
        data[f'tv_losses_from_{name}'] = np.asarray(data[f'tv_losses_from_{name}'])
    return data 

if args.load_data_from:
    print(f'loading data from {args.load_data_from}')
    data = torch.load(args.load_data_from)
    data_lin_dip = data['data_lin_dip']
    data = data['data']
else:
    data = collect_kmnist_tv_hists_and_samples_from_dists_figure_data(args)
    data_lin_dip = collect_kmnist_tv_hists_from_lin_dip_samples(args, runs)

if args.save_data_to:
    print(f'saving data to {args.save_data_to}')
    torch.save(
        {'data': data, 'data_lin_dip': data_lin_dip},  args.save_data_to)

configure_matplotlib()

fig, axs = plt.subplots(1, 9, figsize=(16, 2), gridspec_kw={
    'width_ratios': [1., 0.25, 1., 0.05, 1., 1., 1., 1., 1.],  # includes spacer columns
    'wspace': 0.005})

title_dict = {
    'tv': 'TV',
    'hybrid': 'TV-PredCP',
    'gauss': 'Fact.~Gauss.',
    'kmnist': 'KMNIST',
    'bayes_dip': 'lin.-DIP (MLL)',
    'bayes_dip_predcp': 'lin.-DIP (TV-MAP)',
    }
color_dict = {
    'tv': '#2D4263',
    'hybrid': '#D9534F',
    'gauss': '#3F0071',
    'kmnist': '#E2703A',
    'bayes_dip': DEFAULT_COLORS['bayes_dip'],
    'bayes_dip_predcp': DEFAULT_COLORS['bayes_dip_predcp'],
    }
alpha = 0.45

hist_kwargs = {
    'bins': 10,
    'zorder': [10, 8, 6, 4, 2, 0],
    'linewidth': 1.5
}

hist_0_names = ('tv', 'hybrid', 'gauss', 'kmnist')
_, n_list, _ = plot_hist(
        [data[f'tv_losses_from_{name}'] for name in hist_0_names],
        label_list=[title_dict[name] for name in hist_0_names],
        color_list=[color_dict[name] for name in hist_0_names],
        alpha_list=[alpha] * len(hist_0_names),
        yscale='linear',
        hist_kwargs=hist_kwargs,
        legend_kwargs='off',
        ax=axs[0],
        )
axs[0].set_ylim(_get_ylim(n_list))
axs[0].set_xlabel('average TV')
axs[1].remove()
hist_1_names = ('bayes_dip', 'bayes_dip_predcp', 'kmnist')
_, n_list, _ = plot_hist(
        [data_lin_dip[f'tv_losses_from_{name}'] for name in hist_1_names],
        label_list=[title_dict[name] for name in hist_1_names],
        color_list=[color_dict[name] for name in hist_1_names],
        alpha_list=[alpha] * len(hist_1_names),
        yscale='linear',
        hist_kwargs=hist_kwargs,
        legend_kwargs='off',
        ax=axs[2],
        )
axs[2].set_ylim(_get_ylim(n_list))
axs[2].set_ylabel('')
axs[2].set_xlabel('average TV')
axs[3].remove()
hist_0_handles, hist_0_labels = axs[0].get_legend_handles_labels()
hist_1_handles, hist_1_labels = axs[2].get_legend_handles_labels()
handles = hist_0_handles[:-1] + hist_1_handles
labels = hist_0_labels[:-1] + hist_1_labels
fig.legend(handles, labels, ncol=len(handles), loc='lower right', bbox_to_anchor=(0.88, -0.1))

plot_sample(axs[4], data['samples_from_tv'][0], title='TV')
plot_sample(axs[5], data['samples_from_hybrid'][0], title='TV-PredCP')
plot_sample(axs[6], data['samples_from_gauss'][0], title='Fact.~Gauss.')
plot_sample(axs[7], data_lin_dip['bayes_dip'][4][7], title='lin.-DIP (MLL)', **{'vmax': data_lin_dip['mll_vmax'][4], 'vmin': 1e-3})
plot_sample(axs[8], data_lin_dip['bayes_dip_predcp'][4][7], title='lin.-DIP (TV-MAP)', **{'vmax': data_lin_dip['map_vmax'][4], 'vmin': 1e-3})

fig.savefig(f'kmnist_tv_hists_and_samples_from_dists.pdf', bbox_inches='tight', pad_inches=0.)
fig.savefig(f'kmnist_tv_hists_and_samples_from_dists.png', bbox_inches='tight', pad_inches=0., dpi=600)
