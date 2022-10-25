import os
import numpy as np 
import argparse
import yaml
from omegaconf import OmegaConf
import torch
import matplotlib.pyplot as plt 
from bayes_dip.utils import tv_loss
from bayes_dip.utils.experiment_utils import (
        get_standard_ray_trafo)
from bayes_dip.utils import PSNR, SSIM
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import (
        get_default_unet_gaussian_prior_dicts, get_default_unet_gprior_dicts)
from bayes_dip.probabilistic_models import (
        get_matmul_neural_basis_expansion, ParameterCov)
from bayes_dip.utils.evaluation_utils import (
        translate_path)
from bayes_dip.utils.plot_utils import (
        DEFAULT_COLORS, configure_matplotlib)


parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_kmnist_exact_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--sample_idx', type=int, default=0)
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--sweep_grid_points', type=int, default=100)
parser.add_argument('--do_not_fix_marginal_1', action='store_true', default=False, help='do not fix marginal variances to 1')
parser.add_argument('--do_not_print_titles', action='store_true', default=False, help='do not fix marginal variances to 1')
parser.add_argument('--angles', type=int, default=5)
parser.add_argument('--noise', type=float, default=0.05)
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--save_data_to', type=str, default='', help='path to cache the plot data, such that they can be loaded with --load_data_from')
parser.add_argument('--load_data_from', type=str, default='', help='load data cached from a previous run with --save_data_to')

args = parser.parse_args()
configure_matplotlib()

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

run_path = translate_path(runs[args.noise][args.angles], experiment_paths=experiment_paths)
dip_run_cfg = OmegaConf.load(os.path.join(
        translate_path(run_path, experiment_paths=experiment_paths),
        '.hydra', 'config.yaml')
        )
def collect_bijectivity_data(): 

    dtype = torch.get_default_dtype()
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    ray_trafo = get_standard_ray_trafo(dip_run_cfg)
    ray_trafo.to(dtype=dtype, device=device)

    ground_truth = torch.load(os.path.join(run_path, f'sample_{args.sample_idx}.pt'),
                map_location='cpu')['ground_truth'].detach()
    filtbackproj = torch.load(os.path.join(run_path, f'sample_{args.sample_idx}.pt'),
                map_location='cpu')['filtbackproj'].detach()
    recon = torch.load(os.path.join(run_path, f'sample_{args.sample_idx}.pt'),
                map_location='cpu')['filtbackproj'].detach()

    net_kwargs = {
        'scales': dip_run_cfg.dip.net.scales,
        'channels': dip_run_cfg.dip.net.channels,
        'skip_channels': dip_run_cfg.dip.net.skip_channels,
        'use_norm': dip_run_cfg.dip.net.use_norm,
        'use_sigmoid': dip_run_cfg.dip.net.use_sigmoid,
        'sigmoid_saturation_thresh': dip_run_cfg.dip.net.sigmoid_saturation_thresh}

    reconstructor = DeepImagePriorReconstructor(
        ray_trafo, torch_manual_seed=dip_run_cfg.dip.torch_manual_seed,
        device=device, net_kwargs=net_kwargs,
        load_params_path=dip_run_cfg.load_pretrained_dip_params)

    dip_params_filepath = os.path.join(run_path, f'dip_model_{args.sample_idx}.pt')
    print(f'loading DIP network parameters from {dip_params_filepath}')
    reconstructor.load_params(dip_params_filepath)
    recon = reconstructor.nn_model(filtbackproj.to(device)).detach()

    print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
    print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

    prior_assignment_dict, hyperparams_init_dict = (
            get_default_unet_gaussian_prior_dicts(reconstructor.nn_model)
            if not dip_run_cfg.priors.use_gprior else
            get_default_unet_gprior_dicts(reconstructor.nn_model))

    num_samples = args.num_samples
    expected_tv_dict = {}
    for key in prior_assignment_dict.keys():
        prior_assignment_dict_red = {}
        expected_tv_per_layer = []
        if hyperparams_init_dict[key].get('lengthscale') is not None:
            prior_assignment_dict_red[key] = prior_assignment_dict[key]
            for i in np.logspace(-2, 2, args.sweep_grid_points):
                hyperparams_init_dict[key]['lengthscale'] = i 
                parameter_cov = ParameterCov(
                        reconstructor.nn_model,
                        prior_assignment_dict_red,
                        hyperparams_init_dict,
                        device=device
                )
                matmul_neural_basis_expansion = get_matmul_neural_basis_expansion(
                    nn_model=reconstructor.nn_model,
                    nn_input=filtbackproj.to(device),
                    ordered_nn_params=parameter_cov.ordered_nn_params,
                    nn_out_shape=filtbackproj.shape,
                    use_gprior=dip_run_cfg.priors.use_gprior,
                    trafo=ray_trafo,
                    scale_kwargs=OmegaConf.to_object(dip_run_cfg.priors.gprior.scale),
                )
                sigma_xx = parameter_cov(matmul_neural_basis_expansion._matrix) @ matmul_neural_basis_expansion._matrix.T
                sigma_xx = sigma_xx.detach()
                sigma_xx[np.diag_indices(sigma_xx.shape[0])] += 1e-6
                if not args.do_not_fix_marginal_1:
                    corr = sigma_xx.diag().sqrt().pow(-1).diag() # fixing marginal variance to be one
                    sigma_xx = corr @ sigma_xx @ corr
                cnt = 0
                succed = False
                while not succed:
                    try: 
                        dist = \
                            torch.distributions.multivariate_normal.MultivariateNormal(loc=recon.flatten().to(device),
                                scale_tril=torch.linalg.cholesky(sigma_xx))
                        succed = True
                    except: 
                        sigma_xx[np.diag_indices(sigma_xx.shape[0])] += 1e-4
                        cnt += 1
                    assert cnt < 100
                samples = dist.sample((num_samples, )).detach().cpu()
                expected_tv_per_layer.append(
                    tv_loss(samples.reshape(num_samples, 1, *recon.shape[2:])).numpy() / num_samples)
            expected_tv_dict[key] = np.asarray(expected_tv_per_layer)
    return expected_tv_dict

if args.load_data_from:
    print(f'loading data from {args.load_data_from}')
    data = torch.load(args.load_data_from)
else:
    data = collect_bijectivity_data()

if args.save_data_to:
    print(f'saving data to {args.save_data_to}')
    torch.save(data, args.save_data_to)

fig, axs = plt.subplots(1, 5, figsize=(10, 2), gridspec_kw={
    'width_ratios': [1., 1., 1., 1., 1.],  # includes spacer columns
    'wspace': 0.45, 'hspace': 0.2})

labels = {
    'inc': '$(\\ell_1, \sigma_1^2)$', 
    'down_0': '$(\\ell_2, \sigma_2^2)$',
    'down_1': '$(\\ell_3, \sigma_3^2)$', 
    'up_0': '$(\\ell_4, \sigma_4^2)$',
    'up_1': '$(\\ell_5, \sigma_5^2)$'
    }    

for i, (ax, key) in enumerate(zip(axs, data.keys())): 
    ax.plot(np.logspace(-2, 2, args.sweep_grid_points), data[key], DEFAULT_COLORS['bayes_dip'], linewidth=2)
    ax.set_xscale('log')
    ax.grid(0.3)
    if i ==0: ax.set_ylabel('$\kappa$', fontsize=12)
    if not args.do_not_print_titles: ax.set_title(labels[key])
    ax.set_xlabel('$\ell$')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

print('saving figures')
bool_flag = True if not args.do_not_fix_marginal_1 else False
fig.savefig(f'kmnist_bijectivity_fix_{bool_flag}_{args.sample_idx}_{args.noise}_{args.angles}.pdf', bbox_inches='tight', pad_inches=0.)
fig.savefig(f'kmnist_bijectivity_fix_{bool_flag}_{args.sample_idx}_{args.noise}_{args.angles}.png', bbox_inches='tight', pad_inches=0., dpi=600)

fig, axs = plt.subplots(1, 2, figsize=(5, 2), gridspec_kw={
    'width_ratios': [1., 1.],  # includes spacer columns
    'wspace': 0.25, 'hspace': 0.2})

for i, (ax, key) in enumerate(zip(axs, ['inc', 'up_1'])):

    ax.plot(
        np.logspace(-2, 2, args.sweep_grid_points),
        data[key], 
        DEFAULT_COLORS['bayes_dip'], 
        linewidth=2
    )
    ax.set_xscale('log')
    ax.grid(0.3)
    if i ==0: ax.set_ylabel('$\kappa$', fontsize=12)
    ax.set_title(labels[key])
    ax.set_xlabel('$\ell$') 
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

print('saving figures')
bool_flag = True if not args.do_not_fix_marginal_1 else False
fig.savefig(f'kmnist_mini_bijectivity_fix_{bool_flag}_{args.sample_idx}_{args.noise}_{args.angles}.pdf', bbox_inches='tight', pad_inches=0.)
fig.savefig(f'kmnist_mini_bijectivity_fix_{bool_flag}_{args.sample_idx}_{args.noise}_{args.angles}.png', bbox_inches='tight', pad_inches=0., dpi=600)
