import os
import yaml
import argparse
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from bayes_dip.data.walnut_utils import get_projection_data, get_single_slice_ray_trafo
from bayes_dip.data.datasets.walnut import get_walnut_2d_inner_part_defined_by_patch_size
from bayes_dip.utils.evaluation_utils import (
        get_abs_diff, get_density_data, get_recon, get_ground_truth, get_stddev, restrict_sample_based_density_data_to_new_patch_idx_list, translate_path)
from bayes_dip.utils.plot_utils import (
        DEFAULT_COLORS, configure_matplotlib, plot_hist, plot_qq)
from baselines.evaluation_utils import compute_mcdo_reconstruction, get_mcdo_density_data, get_mcdo_stddev

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_walnut_sample_based_density_reweight_off_diagonal_entries/patch_size_1.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--runs_file_approx', type=str, default='runs_walnut_sample_based_density_approx_jacs_cg_reweight_off_diagonal_entries/patch_size_1.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--baseline_mcdo_runs_file', type=str, default='runs_baseline_walnut_mcdo_density_reweight_off_diagonal_entries/patch_size_1.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--do_not_use_predcp', action='store_true', default=False, help='use the run without PredCP (i.e., use MLL instead of TV-MAP)')
parser.add_argument('--include_outer_part', action='store_true', default=False, help='include the outer part of the walnut image (that only contains background)')
parser.add_argument('--patch_size', type=int, default=1, help='patch size')
parser.add_argument('--walnut_data_path', type=str, default='', help='"/path/to/Walnuts/", which should contain e.g. a sub-folder "Walnut1/"; if not specified, the path configured in a hydra config is used')
parser.add_argument('--do_not_subtract_image_noise_correction', action='store_true', default=False, help='do not subtract the image noise correction term (if any) from the covariance diagonals')
parser.add_argument('--save_data_to', type=str, default='', help='path to cache the plot data, such that they can be loaded with --load_data_from')
parser.add_argument('--load_data_from', type=str, default='', help='load data cached from a previous run with --save_data_to')
args = parser.parse_args()

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

with open(args.runs_file_approx, 'r') as f:
    runs_approx = yaml.safe_load(f)

with open(args.baseline_mcdo_runs_file, 'r') as f:
    run_mcdo = yaml.safe_load(f)

cfg = OmegaConf.load(os.path.join(
        translate_path(runs['include_predcp_False'], experiment_paths=experiment_paths),
        '.hydra', 'config.yaml'))  # use this cfg mainly, relevant settings should be the same
assert cfg.inference.patch_size == args.patch_size


def get_walnut_pseudo_2d_sinogram(cfg, walnut_data_path):
    observation_full = get_projection_data(
            data_path=walnut_data_path,
            walnut_id=cfg.dataset.walnut_id,
            orbit_id=cfg.trafo.orbit_id,
            angular_sub_sampling=cfg.trafo.angular_sub_sampling,
            proj_col_sub_sampling=cfg.trafo.proj_col_sub_sampling)
    # WalnutRayTrafo instance needed for selecting and masking the projections
    walnut_ray_trafo = get_single_slice_ray_trafo(
            walnut_data_path,
            walnut_id=cfg.dataset.walnut_id,
            orbit_id=cfg.trafo.orbit_id,
            angular_sub_sampling=cfg.trafo.angular_sub_sampling,
            proj_col_sub_sampling=cfg.trafo.proj_col_sub_sampling)
    observation_2d = np.take_along_axis(
            walnut_ray_trafo.projs_from_full(observation_full),
            walnut_ray_trafo.proj_mask_first_row_inds[None],
            axis=0).squeeze()
    return observation_2d

def normalized_error_for_qq_plot(recon, image, std):
    normalized_error = (recon - image) / std
    return normalized_error

def collect_walnut_main_figure_data(args, cfg, runs, run_mcdo):
    data = {}

    walnut_data_path = args.walnut_data_path or cfg.dataset.data_path

    dip_mll_optim_run = OmegaConf.load(os.path.join(
            translate_path(runs['include_predcp_False'], experiment_paths=experiment_paths),
            '.hydra', 'config.yaml')
            ).inference.load_path

    kwargs = {
        'sample_idx': 0,
        'experiment_paths': experiment_paths,
        }
    stddev_kwargs = {
        'patch_idx_list': None if args.include_outer_part else 'walnut_inner',
        'subtract_image_noise_correction_if_any': not args.do_not_subtract_image_noise_correction,
        }

    data['ground_truth'] = get_ground_truth(dip_mll_optim_run, **kwargs)

    print('collecting bayes_dip data')
    data['recon'] = get_recon(dip_mll_optim_run, **kwargs)
    data['abs_diff'] = get_abs_diff(dip_mll_optim_run, **kwargs)
    data['stddev'] = get_stddev(runs['include_predcp_False'], **kwargs, **stddev_kwargs)
    data['stddev_approx'] = get_stddev(runs_approx['include_predcp_False'], **kwargs, **stddev_kwargs)
    data['stddev_predcp'] = get_stddev(runs['include_predcp_True'], **kwargs, **stddev_kwargs)
    data['stddev_predcp_approx'] = get_stddev(runs_approx['include_predcp_True'], **kwargs, **stddev_kwargs)

    print('collecting mcdo data')
    try:
        data['recon_mcdo'] = torch.load(
                os.path.join(run_mcdo, 'recon_0.pt'), map_location='cpu').squeeze(1).squeeze(0)
    except FileNotFoundError:
        print('did not find mcdo reconstruction, will recompute')
        data['recon_mcdo'] = compute_mcdo_reconstruction(
                run_mcdo, sample_idx=0, device='cpu').squeeze(1).squeeze(0)
    data['abs_diff_mcdo'] = torch.abs(data['recon_mcdo'] - data['ground_truth'])
    data['stddev_mcdo'] = get_mcdo_stddev(run_mcdo, **kwargs, **stddev_kwargs)

    data['mask'] = torch.logical_not(torch.isnan(data['stddev']))
    print(f'Using {data["mask"].sum()} pixels.')

    slice_0, slice_1 = (
            (slice(0, data['ground_truth'].shape[0]), slice(0, data['ground_truth'].shape[1]))
            if args.include_outer_part else
            get_walnut_2d_inner_part_defined_by_patch_size(args.patch_size))
    assert data['mask'].sum() == (slice_0.stop - slice_0.start) * (slice_1.stop - slice_1.start)
    data['slice_0'], data['slice_1'] = slice_0, slice_1


    print('computing bayes_dip log prob')
    cfg_predcp = OmegaConf.load(os.path.join(
            translate_path(runs['include_predcp_True'], experiment_paths=experiment_paths),
            '.hydra', 'config.yaml'))
    assert cfg_predcp.inference.patch_size == args.patch_size
    data['log_lik_predcp'] = restrict_sample_based_density_data_to_new_patch_idx_list(
            data=get_density_data(
                    run_path=runs['include_predcp_True'], sample_idx=0,
                    experiment_paths=experiment_paths)[0],
            patch_idx_list=stddev_kwargs['patch_idx_list'],
            orig_patch_idx_list=cfg_predcp.inference.patch_idx_list,
            patch_size=args.patch_size,
            im_shape=data['ground_truth'].shape)['log_prob']
    data['log_lik_predcp_approx'] = restrict_sample_based_density_data_to_new_patch_idx_list(
            data=get_density_data(
                    run_path=runs_approx['include_predcp_True'], sample_idx=0,
                    experiment_paths=experiment_paths)[0],
            patch_idx_list=stddev_kwargs['patch_idx_list'],
            orig_patch_idx_list=cfg_predcp.inference.patch_idx_list,
            patch_size=args.patch_size,
            im_shape=data['ground_truth'].shape)['log_prob']
    data['log_lik_no_predcp'] = restrict_sample_based_density_data_to_new_patch_idx_list(
            data=get_density_data(
                    run_path=runs['include_predcp_False'], sample_idx=0,
                    experiment_paths=experiment_paths)[0],
            patch_idx_list=stddev_kwargs['patch_idx_list'],
            orig_patch_idx_list=cfg_predcp.inference.patch_idx_list,
            patch_size=args.patch_size,
            im_shape=data['ground_truth'].shape)['log_prob']
    data['log_lik_no_predcp_approx'] = restrict_sample_based_density_data_to_new_patch_idx_list(
            data=get_density_data(
                    run_path=runs_approx['include_predcp_False'], sample_idx=0,
                    experiment_paths=experiment_paths)[0],
            patch_idx_list=stddev_kwargs['patch_idx_list'],
            orig_patch_idx_list=cfg_predcp.inference.patch_idx_list,
            patch_size=args.patch_size,
            im_shape=data['ground_truth'].shape)['log_prob']

    print('computing mcdo log prob')
    data['log_lik_mcdo'] = restrict_sample_based_density_data_to_new_patch_idx_list(
            data=get_mcdo_density_data(
                    run_path=run_mcdo, sample_idx=0,
                    experiment_paths=experiment_paths),
            patch_idx_list=stddev_kwargs['patch_idx_list'],
            orig_patch_idx_list=OmegaConf.load(os.path.join(
                    translate_path(run_mcdo, experiment_paths=experiment_paths),
                    '.hydra', 'config.yaml')).inference.patch_idx_list,
            patch_size=args.patch_size,
            im_shape=data['ground_truth'].shape)['log_prob']

    data['qq_err_mll'] = normalized_error_for_qq_plot(
            data['recon'][slice_0, slice_1],
            data['ground_truth'][slice_0, slice_1],
            data['stddev'][slice_0, slice_1])
    data['qq_err_mll_approx'] = normalized_error_for_qq_plot(
            data['recon'][slice_0, slice_1],
            data['ground_truth'][slice_0, slice_1],
            data['stddev_approx'][slice_0, slice_1])
    data['qq_err_map'] = normalized_error_for_qq_plot(
            data['recon'][slice_0, slice_1],
            data['ground_truth'][slice_0, slice_1],
            data['stddev_predcp'][slice_0, slice_1])
    data['qq_err_map_approx'] = normalized_error_for_qq_plot(
            data['recon'][slice_0, slice_1],
            data['ground_truth'][slice_0, slice_1],
            data['stddev_predcp_approx'][slice_0, slice_1])
    data['qq_err_mcdo'] = normalized_error_for_qq_plot(
            data['recon_mcdo'][slice_0, slice_1],
            data['ground_truth'][slice_0, slice_1],
            data['stddev_mcdo'][slice_0, slice_1])

    print('getting pseudo 2d sinogram')
    data['observation_2d'] = get_walnut_pseudo_2d_sinogram(cfg, walnut_data_path=walnut_data_path)

    return data

if args.load_data_from:
    print(f'loading data from {args.load_data_from}')
    data = torch.load(args.load_data_from)
else:
    data = collect_walnut_main_figure_data(args, cfg, runs, run_mcdo)

if args.save_data_to:
    print(f'saving data to {args.save_data_to}')
    torch.save(data, args.save_data_to)


configure_matplotlib()

fig, axs = plt.subplots(2, 2, figsize=(6, 6), gridspec_kw={
    'width_ratios': [1., 1.],  # includes spacer columns
    'wspace': 0.45, 'hspace': 0.4})

ground_truth = data['ground_truth']
recon = data['recon']
abs_diff = data['abs_diff']
abs_diff_mcdo = data['abs_diff_mcdo']

# nan parts black
stddev = data['stddev'].clone()
stddev[torch.logical_not(data['mask'])] = 0.
stddev_approx = data['stddev_approx'].clone()
stddev_approx[torch.logical_not(data['mask'])] = 0.
stddev_predcp = data['stddev_predcp'].clone()
stddev_predcp[torch.logical_not(data['mask'])] = 0.
stddev_predcp_approx = data['stddev_predcp_approx'].clone()
stddev_predcp_approx[torch.logical_not(data['mask'])] = 0.
stddev_mcdo = data['stddev_mcdo'].clone()
stddev_mcdo[torch.logical_not(data['mask'])] = 0.

slice_0, slice_1 = data['slice_0'], data['slice_1']
vmax_row_0 = max(torch.max(stddev), torch.max(abs_diff))

print('plotting histograms')

plot_hist(
    [abs_diff[slice_0, slice_1], stddev[slice_0, slice_1], stddev_approx[slice_0, slice_1]],
    ['$|\\hat x - x|$',  'std-dev (LL: ${:.2f})$'.format(data['log_lik_no_predcp']), 'std-dev -- $\\tilde J$ $\&$ PCG (LL: ${:.2f})$'.format(data['log_lik_no_predcp_approx'])],
    title='marginal std-dev \n lin.-DIP -- MLL',
    ax=axs[0, 0],
    xlim=(0., 1),
    yscale='log',
    ylim=(1e-4, 1000.),
    remove_ticks=False,
    color_list=[DEFAULT_COLORS['abs_diff'], DEFAULT_COLORS['bayes_dip'], DEFAULT_COLORS['bayes_dip_approx']],
    hist_kwargs_per_data={'zorder': [5, 4, 3]},
    legend_kwargs={'loc': 'upper right', 'prop': {'size': 'x-small'}},
    )

plot_hist(
    [abs_diff[slice_0, slice_1], stddev_predcp[slice_0, slice_1], stddev_predcp_approx[slice_0, slice_1]],
    ['$|\\hat x - x|$', 'std-dev (LL: ${:.2f})$'.format(data['log_lik_predcp']), 'std-dev -- $\\tilde J$ $\&$ PCG (LL: ${:.2f})$'.format(data['log_lik_predcp_approx'])],
    title='marginal std-dev \n lin.-DIP -- TV-MAP',
    ax=axs[1, 0],
    xlim=(0., 1),
    yscale='log',
    ylim=(1e-4, 1000.),
    remove_ticks=False,
    color_list=[DEFAULT_COLORS['abs_diff'], DEFAULT_COLORS['bayes_dip_predcp'], DEFAULT_COLORS['bayes_dip_predcp_approx']],
    hist_kwargs_per_data={'zorder': [5, 4, 3]},
    legend_kwargs={'loc': 'upper right', 'prop': {'size': 'x-small'}},
    )

plot_hist(
    [abs_diff_mcdo[slice_0, slice_1], stddev_mcdo[slice_0, slice_1]],
    ['$|\\hat x - x|$', 'std-dev (LL: ${:.2f})$'.format(data['log_lik_mcdo'])],
    title='marginal std-dev \n DIP-MCDO',
    ax=axs[0, 1],
    xlim=(0., 1),
    yscale='log',
    ylim=(1e-4, 1000.),
    remove_ticks=False,
    color_list=[DEFAULT_COLORS['abs_diff'], DEFAULT_COLORS['mcdo']],
    hist_kwargs_per_data={'zorder': [5, 4, 3]},
    legend_kwargs={'loc': 'upper right', 'prop': {'size': 'x-small'}},
    )

print('plotting Q-Q plot')

osm_mll, osr_mll = scipy.stats.probplot(data['qq_err_mll'].flatten(), fit=False)
osm_mll_approx, osr_mll_approx = scipy.stats.probplot(data['qq_err_mll_approx'].flatten(), fit=False)
osm_map, osr_map = scipy.stats.probplot(data['qq_err_map'].flatten(), fit=False)
osm_map_approx, osr_map_approx = scipy.stats.probplot(data['qq_err_map_approx'].flatten(), fit=False)
osm_mcdo, osr_mcdo = scipy.stats.probplot(data['qq_err_mcdo'].flatten(), fit=False)

plot_qq(
        ax=axs[1, 1],
        data=[(osm_mll, osr_mll), (osm_mll_approx, osr_mll_approx), (osm_map, osr_map), (osm_map_approx, osr_map_approx), (osm_mcdo, osr_mcdo)],
        label_list=['lin.-DIP -- MLL', 'lin.-DIP -- MLL -- $\\tilde J$ $\&$ PCG', 'lin.-DIP -- TV-MAP', 'lin.-DIP -- TV-MAP -- $\\tilde J$ $\&$ PCG', 'DIP-MCDO'],
        color_list=[DEFAULT_COLORS['bayes_dip'], DEFAULT_COLORS['bayes_dip_approx'], DEFAULT_COLORS['bayes_dip_predcp'], DEFAULT_COLORS['bayes_dip_predcp_approx'], DEFAULT_COLORS['mcdo']],
        legend_kwargs='off')
axs[1, 1].set_title('calibration: Q-Q')
axs[1, 1].set_xlabel('prediction quantiles', labelpad=2)
axs[1, 1].set_ylabel('error quantiles', labelpad=2)
handles, labels = axs[1, 1].get_legend_handles_labels()
handles_leg1, labels_leg1 = handles.copy(), labels.copy()
del handles_leg1[4]
del labels_leg1[4]
handles_leg1.insert(2, Line2D([0],[0],color="w"))
labels_leg1.insert(2, '')
legend_kwargs = {'loc': 'lower center', 'bbox_to_anchor': (0.5, -0.085), 'ncol': 2, 'markerscale': 3}
legend = fig.legend(handles_leg1, labels_leg1, **legend_kwargs)
fig.add_artist(legend)
fig.legend([handles[-1]], [labels[-1]], **legend_kwargs, frameon=False, facecolor='none')

print('saving figures')
fig.savefig(f'walnut_{args.patch_size}x{args.patch_size}_uq_main.pdf', bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0.1)
fig.savefig(f'walnut_{args.patch_size}x{args.patch_size}_uq_main.png', bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0.1, dpi=600)
