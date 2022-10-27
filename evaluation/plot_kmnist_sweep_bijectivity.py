import os
import numpy as np
from math import ceil
import argparse
import yaml
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from bayes_dip.utils import tv_loss
from bayes_dip.probabilistic_models import (
        get_default_unet_gaussian_prior_dicts, ParameterCov, MatmulNeuralBasisExpansion, GPprior)
from bayes_dip.utils.evaluation_utils import get_nn_model, translate_path
from bayes_dip.utils.plot_utils import DEFAULT_COLORS, configure_matplotlib


parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_kmnist_dip.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--sample_idx', type=int, default=0)
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--sweep_grid_points', type=int, default=100)
parser.add_argument('--do_not_fix_marginal_1', action='store_true', default=False, help='do not fix marginal variances to 1')
parser.add_argument('--do_not_print_titles', action='store_true', default=False, help='do not print titles')
parser.add_argument('--angles', type=int, default=5)
parser.add_argument('--noise', type=float, default=0.05)
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--save_data_to', type=str, default='', help='path to cache the plot data, such that they can be loaded with --load_data_from')
parser.add_argument('--load_data_from', type=str, default='', help='load data cached from a previous run with --save_data_to')
parser.add_argument('--prior_key_list', type=str, nargs='*', default=[], help='priors to plot (e.g. "inc down_0 down_1 up_0 up_1")')
parser.add_argument('--suffix', type=str, default='', help='suffix for the figure filenames (e.g. summarizing the --prior_key_list)')
parser.add_argument('--rows', type=int, default=1, help='number of subplot rows')
parser.add_argument('--hspace', type=float, default=0.2, help='matplotlib\'s "hspace" gridspec_kw')
parser.add_argument('--wspace', type=float, default=0.2, help='matplotlib\'s "wspace" gridspec_kw')
parser.add_argument('--figsize_add', type=float, nargs=2, default=[0., 0.], help='matplotlib\'s "figsize" will computed as the sum of `(2 * cols, 2 * rows)` and the values passed with this argument')

args = parser.parse_args()
configure_matplotlib()

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)


def collect_kmnist_sweep_bijectivity_figure_data(args, runs):

    run_path = runs[args.noise][args.angles]
    cfg = OmegaConf.load(os.path.join(
            translate_path(run_path, experiment_paths=experiment_paths),
            '.hydra', 'config.yaml')
            )
    assert not cfg.priors.use_gprior

    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    nn_model, filtbackproj = get_nn_model(
            run_path, sample_idx=args.sample_idx, experiment_paths=experiment_paths, device=device)
    recon = nn_model(filtbackproj).detach()

    prior_assignment_dict, hyperparams_init_dict = get_default_unet_gaussian_prior_dicts(nn_model)

    num_samples = args.num_samples
    expected_tv_dict = {}
    for key in prior_assignment_dict.keys():
        expected_tv_per_layer = []
        prior_assignment_dict_red = {key: prior_assignment_dict[key]}
        parameter_cov = ParameterCov(
                nn_model,
                prior_assignment_dict_red,
                hyperparams_init_dict,
                device=device
        )
        if isinstance(parameter_cov.priors[key], GPprior):
            print(f'sweeping for prior {key}')
            matmul_neural_basis_expansion = MatmulNeuralBasisExpansion(
                nn_model=nn_model,
                nn_input=filtbackproj.to(device),
                ordered_nn_params=parameter_cov.ordered_nn_params,
                nn_out_shape=filtbackproj.shape,
            )
            for i in tqdm(np.logspace(-2, 2, args.sweep_grid_points)):
                parameter_cov.priors[key].log_lengthscale.data[:] = np.log(i)
                sigma_xx = (
                        parameter_cov(matmul_neural_basis_expansion.matrix)
                        @ matmul_neural_basis_expansion.matrix.T)
                sigma_xx = sigma_xx.detach()
                sigma_xx[np.diag_indices(sigma_xx.shape[0])] += 1e-6
                if not args.do_not_fix_marginal_1:
                    corr = sigma_xx.diag().sqrt().pow(-1).diag()  # fix marginal variance to be one
                    sigma_xx = corr @ sigma_xx @ corr
                cnt = 0
                succeed = False
                while not succeed:
                    try:
                        dist = \
                            torch.distributions.multivariate_normal.MultivariateNormal(
                                    loc=recon.flatten().to(device),
                                    scale_tril=torch.linalg.cholesky(sigma_xx))
                        succeed = True
                    except:
                        sigma_xx[np.diag_indices(sigma_xx.shape[0])] += 1e-4
                        cnt += 1
                    assert cnt < 100
                samples = dist.sample((num_samples, )).detach().cpu()
                expected_tv_per_layer.append(
                        tv_loss(samples.reshape(
                                num_samples, 1, *filtbackproj.shape[2:])).numpy() / num_samples)
            expected_tv_dict[key] = np.asarray(expected_tv_per_layer)
    return expected_tv_dict

if args.load_data_from:
    print(f'loading data from {args.load_data_from}')
    data = torch.load(args.load_data_from)
else:
    data = collect_kmnist_sweep_bijectivity_figure_data(args, runs)

if args.save_data_to:
    print(f'saving data to {args.save_data_to}')
    torch.save(data, args.save_data_to)

num_rows, num_cols = args.rows, ceil(len(args.prior_key_list) / args.rows)

fig, axs = plt.subplots(
        1, 5, figsize=(2 * num_cols + args.figsize_add[0], 2 * num_rows + args.figsize_add[1]),
        gridspec_kw={
        'wspace': args.wspace, 'hspace': args.hspace})

labels = {
    'inc': '$(\\ell_1, \sigma_1^2)$',
    'down_0': '$(\\ell_2, \sigma_2^2)$',
    'down_1': '$(\\ell_3, \sigma_3^2)$',
    'up_0': '$(\\ell_4, \sigma_4^2)$',
    'up_1': '$(\\ell_5, \sigma_5^2)$'
    }

for i, (ax, key) in enumerate(zip(axs, args.prior_key_list)):
    ax.plot(np.logspace(-2, 2, args.sweep_grid_points), data[key],
            color=DEFAULT_COLORS['bayes_dip'], linewidth=2)
    ax.set_xscale('log')
    ax.grid(0.3)
    if i == 0:
        ax.set_ylabel('$\kappa$', fontsize=12)
    if not args.do_not_print_titles:
        ax.set_title(labels[key])
    ax.set_xlabel('$\ell$')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

print('saving figures')
suffix = '_' + args.suffix if args.suffix and not args.suffix.startswith('_') else args.suffix
fig.savefig(f'kmnist_bijectivity_fix_{not args.do_not_fix_marginal_1}_{args.sample_idx}_{args.noise}_{args.angles}{suffix}.pdf', bbox_inches='tight', pad_inches=0.)
fig.savefig(f'kmnist_bijectivity_fix_{not args.do_not_fix_marginal_1}_{args.sample_idx}_{args.noise}_{args.angles}{suffix}.png', bbox_inches='tight', pad_inches=0., dpi=600)
