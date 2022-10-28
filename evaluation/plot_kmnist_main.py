import os
import yaml
import argparse
import torch
import scipy
import matplotlib
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from bayes_dip.utils.utils import PSNR, SSIM
from bayes_dip.utils.evaluation_utils import (
        get_abs_diff, get_density_data, get_recon, get_ground_truth, get_observation, get_stddev, translate_path)
from bayes_dip.utils.plot_utils import (
        DEFAULT_COLORS, configure_matplotlib, plot_hist, plot_image, add_metrics)
from baselines.evaluation_utils import compute_mcdo_reconstruction, get_mcdo_density_data, get_mcdo_stddev

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_kmnist_exact_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--sample_idx', type=int, default=0)
parser.add_argument('--angles', type=int, default=20)
parser.add_argument('--noise', type=float, default=0.05)
parser.add_argument('--hist_xlim_max', type=float, default=0.25)
parser.add_argument('--baseline_mcdo_runs_file', type=str, default='runs_baseline_kmnist_mcdo_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--do_not_subtract_image_noise_correction', action='store_true', default=False, help='do not subtract the image noise correction term (if any) from the covariance diagonals')
parser.add_argument('--save_data_to', type=str, default='', help='path to cache the plot data, such that they can be loaded with --load_data_from')
parser.add_argument('--load_data_from', type=str, default='', help='load data cached from a previous run with --save_data_to')
args = parser.parse_args()

PATCH_SIZE = 28
experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

with open(args.baseline_mcdo_runs_file, 'r') as f:
    run_mcdo = yaml.safe_load(f)

cfg = OmegaConf.load(os.path.join(
        translate_path(runs[args.noise][args.angles]['include_predcp_False'], experiment_paths=experiment_paths),
        '.hydra', 'config.yaml'))  # use this cfg mainly, relevant settings should be the same

def collect_kmnist_main_figure_data(args, cfg, runs, run_mcdo):
    data = {}

    dip_mll_optim_run = OmegaConf.load(os.path.join(
            translate_path(runs[args.noise][args.angles]['include_predcp_False'], experiment_paths=experiment_paths),
            '.hydra', 'config.yaml')
            ).inference.load_path

    kwargs = {
        'sample_idx': args.sample_idx,
        'experiment_paths': experiment_paths,
        }
    stddev_kwargs = {
        'subtract_image_noise_correction_if_any': not args.do_not_subtract_image_noise_correction,
        }

    data['ground_truth'] = get_ground_truth(dip_mll_optim_run, **kwargs)
    data['observation_2d'] = get_observation(dip_mll_optim_run, **kwargs)

    print('collecting bayes_dip data')
    data['recon'] = get_recon(dip_mll_optim_run, **kwargs)
    data['abs_diff'] = get_abs_diff(dip_mll_optim_run, **kwargs)
    data['stddev'] = get_stddev(runs[args.noise][args.angles]['include_predcp_False'], **kwargs, **stddev_kwargs)
    data['stddev_predcp'] = get_stddev(runs[args.noise][args.angles]['include_predcp_True'], **kwargs, **stddev_kwargs)

    print('collecting mcdo data')
    try:
        data['recon_mcdo'] = torch.load(
                os.path.join(run_mcdo[args.noise][args.angles], 'recon_0.pt'), map_location='cpu').squeeze(1).squeeze(0)
    except FileNotFoundError:
        print('did not find mcdo reconstruction, will recompute')
        data['recon_mcdo'] = compute_mcdo_reconstruction(
                run_mcdo[args.noise][args.angles], sample_idx=args.sample_idx, device='cpu').squeeze(1).squeeze(0)
    data['abs_diff_mcdo'] = torch.abs(data['recon_mcdo'] - data['ground_truth'])
    data['stddev_mcdo'] = get_mcdo_stddev(run_mcdo[args.noise][args.angles], **kwargs, **stddev_kwargs)

    data['mask'] = torch.logical_not(torch.isnan(data['stddev']))
    print(f'Using {data["mask"].sum()} pixels.')

    print('computing bayes_dip log prob')
    cfg_predcp = OmegaConf.load(os.path.join(
            translate_path(runs[args.noise][args.angles]['include_predcp_True'], experiment_paths=experiment_paths),
            '.hydra', 'config.yaml'))
    assert cfg_predcp.inference.patch_size == PATCH_SIZE
    data['log_lik_predcp'] = get_density_data(
                    run_path=runs[args.noise][args.angles]['include_predcp_True'], sample_idx=args.sample_idx,
                    experiment_paths=experiment_paths)[0]['log_prob']
    cfg_no_predcp = OmegaConf.load(os.path.join(
            translate_path(runs[args.noise][args.angles]['include_predcp_False'], experiment_paths=experiment_paths),
            '.hydra', 'config.yaml'))
    assert cfg_no_predcp.inference.patch_size == PATCH_SIZE
    data['log_lik_no_predcp'] = get_density_data(
                    run_path=runs[args.noise][args.angles]['include_predcp_False'], sample_idx=args.sample_idx,
                    experiment_paths=experiment_paths)[0]['log_prob']
    print('computing mcdo log prob')
    data['log_lik_mcdo'] = get_mcdo_density_data(
                    run_path=run_mcdo[args.noise][args.angles], sample_idx=args.sample_idx,
                    experiment_paths=experiment_paths)['log_prob']

    data['psnr'] = PSNR(
            data['recon'].cpu().numpy(),
            data['ground_truth'].cpu().numpy())
    data['ssim'] = SSIM(
            data['recon'].cpu().numpy(),
            data['ground_truth'].cpu().numpy())
    data['psnr_mcdo'] = PSNR(
            data['recon_mcdo'].cpu().numpy(),
            data['ground_truth'].cpu().numpy())
    data['ssim_mcdo'] = SSIM(
            data['recon_mcdo'].cpu().numpy(),
            data['ground_truth'].cpu().numpy())

    print('getting pseudo 2d sinogram')
    return data

if args.load_data_from:
    print(f'loading data from {args.load_data_from}')
    data = torch.load(args.load_data_from)
else:
    data = collect_kmnist_main_figure_data(args, cfg, runs, run_mcdo)

if args.save_data_to:
    print(f'saving data to {args.save_data_to}')
    torch.save(data, args.save_data_to)

configure_matplotlib()
fig, axs = plt.subplots(2, 8, figsize=(14, 4.5), gridspec_kw={
    'width_ratios': [1., 0.05, 1., 1., 0.05, 1., 0.45, 1.],  # includes spacer columns
    'wspace': 0.01, 'hspace': 0.2})

ground_truth = data['ground_truth']
recon = data['recon']
abs_diff = data['abs_diff']
abs_diff_mcdo = data['abs_diff_mcdo']
stddev = data['stddev'].clone()
stddev_predcp = data['stddev_predcp'].clone()
stddev_mcdo = data['stddev_mcdo'].clone()

vmax_row_0 = max(torch.max(stddev), torch.max(abs_diff))
print('plotting images')
plot_image(fig, axs[0, 0], ground_truth, title='$x$', vmin=0.)
plot_image(fig, axs[0, 2], recon, title='$\\hat x$', vmin=0.)
add_metrics(axs[0, 2], data['psnr'], data['ssim'], **{'size': 'small'})
plot_image(fig, axs[1, 2], data['recon_mcdo'], vmin=0.)
add_metrics(axs[1, 2], data['psnr_mcdo'], data['ssim_mcdo'], **{'size': 'small'})
axs[0, 2].set_ylabel('\\textbf{lin.-DIP}',  fontsize=plt.rcParams['axes.titlesize'], )
axs[1, 2].set_ylabel('\\textbf{DIP-MCDO}',  fontsize=plt.rcParams['axes.titlesize'], )
# spacer
axs[0, 1].remove()
axs[1, 1].remove()
plot_image(fig, axs[0, 3], abs_diff, title='$|\\hat x - x|$', vmin=0., vmax=vmax_row_0, colorbar='invisible')
plot_image(fig, axs[1, 3], abs_diff_mcdo, vmin=0., colorbar=True)
# spacer
axs[0, 4].remove()
axs[1, 4].remove()
# spacer
axs[0, 6].remove()
axs[1, 6].remove()
plot_image(fig, axs[0, 5], stddev_predcp, vmin=0., vmax=vmax_row_0, title='std-dev', colorbar=True)
axs[0, 5].set_ylabel('TV-MAP', labelpad=2)
plot_image(fig, axs[1, 5], stddev_mcdo, vmin=0., colorbar=True)
axs[1, 0].imshow(data['observation_2d'].T, cmap='gray')
axs[1, 0].set_title('$y$')
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])

print('plotting histograms')
plot_hist(
    [abs_diff, stddev, stddev_predcp],
    ['$|\\hat x - x|$',  'std-dev -- MLL (LL: ${:.2f}$)'.format(data['log_lik_no_predcp']), 'std-dev -- TV-MAP (LL: ${:.2f}$)'.format(data['log_lik_predcp'])],
    title='marginal std-dev',
    ax=axs[0, 7],
    remove_ticks=True,
    color_list=[DEFAULT_COLORS['abs_diff'], DEFAULT_COLORS['bayes_dip'], DEFAULT_COLORS['bayes_dip_predcp']],
    legend_kwargs={'loc': 'upper right', 'bbox_to_anchor': (1.015, 0.99), 'prop': {'size': 'x-small'}},
    hist_kwargs_per_data={'zorder': [5, 4, 3]},
    xlim=(0., args.hist_xlim_max),
    ylim=(1e-2, 1000.),
    )
plot_hist(
    [abs_diff_mcdo, stddev_mcdo],
    ['$|\\hat x - x|$', 'std-dev -- MCDO (LL: ${:.2f}$)'.format(data['log_lik_mcdo'])],
    title='',
    ax=axs[1, 7],
    remove_ticks=False,
    color_list=[DEFAULT_COLORS['abs_diff'], DEFAULT_COLORS['mcdo']],
    legend_kwargs={'loc': 'upper right', 'bbox_to_anchor': (1.015,  0.99), 'prop':  {'size': 'x-small'}},
    hist_kwargs_per_data={'zorder': [5, 4, 3]},
    xlim=(0., args.hist_xlim_max),
    ylim=(1e-2, 1000.),
    )

print('saving figures')
fig.savefig(f'kmnist_{PATCH_SIZE}x{PATCH_SIZE}_{args.sample_idx}_{args.noise}_{args.angles}.pdf', bbox_inches='tight', pad_inches=0.)
fig.savefig(f'kmnist_{PATCH_SIZE}x{PATCH_SIZE}_{args.sample_idx}_{args.noise}_{args.angles}.png', bbox_inches='tight', pad_inches=0., dpi=600)
