import os
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from bayes_dip.data.walnut_utils import get_projection_data, get_single_slice_ray_trafo
from bayes_dip.data.datasets.walnut import get_walnut_2d_inner_part_defined_by_patch_size
from bayes_dip.utils.utils import PSNR, SSIM
from bayes_dip.utils.evaluation_utils import (
        get_abs_diff, get_density_data, get_recon, get_ground_truth, get_stddev, restrict_sample_based_density_data_to_new_patch_idx_list, translate_path)
from bayes_dip.utils.plot_utils import (
        configure_matplotlib,  plot_image, add_inner_rect, add_metrics,
        )
from baselines.evaluation_utils import compute_mcdo_reconstruction, get_mcdo_density_data, get_mcdo_stddev

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_walnut_sample_based_density_reweight_off_diagonal_entries/patch_size_1.yaml', help='path of yaml file containing hydra output directory names')
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
    data['stddev_predcp'] = get_stddev(runs['include_predcp_True'], **kwargs, **stddev_kwargs)

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


    data['psnr'] = PSNR(
            data['recon'][slice_0, slice_1].cpu().numpy(),
            data['ground_truth'][slice_0, slice_1].cpu().numpy())
    data['ssim'] = SSIM(
            data['recon'][slice_0, slice_1].cpu().numpy(),
            data['ground_truth'][slice_0, slice_1].cpu().numpy())
    data['psnr_mcdo'] = PSNR(
            data['recon_mcdo'][slice_0, slice_1].cpu().numpy(),
            data['ground_truth'][slice_0, slice_1].cpu().numpy())
    data['ssim_mcdo'] = SSIM(
            data['recon_mcdo'][slice_0, slice_1].cpu().numpy(),
            data['ground_truth'][slice_0, slice_1].cpu().numpy())


    data['qq_err_map'] = normalized_error_for_qq_plot(
            data['recon'][slice_0, slice_1],
            data['ground_truth'][slice_0, slice_1],
            data['stddev_predcp'][slice_0, slice_1])
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


fig, axs = plt.subplots(2, 6, figsize=(16, 7.25), gridspec_kw={
    'width_ratios': [1., 0.01, 1., 1., 0.01, 1.],  # includes spacer columns
    'wspace': 0.01, 'hspace': 0.25})


ground_truth = data['ground_truth']
recon = data['recon']
abs_diff = data['abs_diff']
abs_diff_mcdo = data['abs_diff_mcdo']

# nan parts black
stddev = data['stddev'].clone()
stddev[torch.logical_not(data['mask'])] = 0.
stddev_predcp = data['stddev_predcp'].clone()
stddev_predcp[torch.logical_not(data['mask'])] = 0.
stddev_mcdo = data['stddev_mcdo'].clone()
stddev_mcdo[torch.logical_not(data['mask'])] = 0.

slice_0, slice_1 = data['slice_0'], data['slice_1']


insets = [
    {
        'rect': [190, 345, 133, 70],
        'axes_rect': [0.8, 0.62, 0.2, 0.38],
        'frame_path': [[0., 1.], [0., 0.], [1., 0.]],
        'clip_path_closing': [[1., 1.]],
    },
    {
        'rect': [220, 200, 55, 65],
        'axes_rect': [0., 0., 0.39, 0.33],
        'frame_path': [[1., 0.], [1., 0.45], [0.3, 1.], [0., 1.]],
        'clip_path_closing': [[0., 0.]],
    },
]


vmax_row_0 = max(torch.max(stddev_predcp), torch.max(abs_diff))


print('plotting images')
plot_image(fig, axs[0, 0], ground_truth, title='$x$', vmin=0., insets=insets, insets_mark_in_orig=True)
add_inner_rect(axs[0, 0], slice_0, slice_1)
plot_image(fig, axs[0, 2], recon, title='$\\hat x$', vmin=0., insets=insets)
add_inner_rect(axs[0, 2], slice_0, slice_1, )
add_metrics(axs[0, 2], data['psnr'], data['ssim'], **{'fontsize': 14})
plot_image(fig, axs[1, 2], data['recon_mcdo'], vmin=0., insets=insets)
add_inner_rect(axs[1, 2], slice_0, slice_1)
add_metrics(axs[1, 2], data['psnr_mcdo'], data['ssim_mcdo'], **{'fontsize': 14})
axs[0, 2].set_ylabel('\\textbf{lin.-DIP}', fontsize=16)
axs[1, 2].set_ylabel('\\textbf{DIP-MCDO}', fontsize=16)
# spacer
axs[0, 1].remove()
axs[1, 1].remove()
plot_image(fig, axs[0, 3], abs_diff, title='$|\\hat x - x|$', vmin=0., vmax=vmax_row_0, insets=insets, colorbar='invisible')
add_inner_rect(axs[0, 3], slice_0, slice_1)
plot_image(fig, axs[1, 3], abs_diff_mcdo, vmin=0., insets=insets, colorbar=True)
add_inner_rect(axs[1, 3], slice_0, slice_1)
# spacer
axs[0, 4].remove()
axs[1, 4].remove()
plot_image(fig, axs[0, 5], stddev_predcp, vmin=0., vmax=vmax_row_0, title='std-dev', insets=insets, colorbar=True)
add_inner_rect(axs[0, 5], slice_0, slice_1)
axs[0, 5].set_ylabel('TV-MAP', fontsize=14, labelpad=2)
plot_image(fig, axs[1, 5], stddev_mcdo, vmin=0., insets=insets, colorbar=True)
add_inner_rect(axs[1, 5], slice_0, slice_1)

axs[1, 0].imshow(data['observation_2d'].T, cmap='gray')
axs[1, 0].set_title('$y$')
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])

print('saving figures')
fig.savefig(f'walnut_{args.patch_size}x{args.patch_size}.pdf', bbox_inches='tight', pad_inches=0.)
fig.savefig(f'walnut_{args.patch_size}x{args.patch_size}.png', bbox_inches='tight', pad_inches=0., dpi=600)
