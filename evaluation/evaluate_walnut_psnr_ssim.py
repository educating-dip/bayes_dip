import os
import yaml
import argparse
from math import ceil
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from bayes_dip.utils import PSNR, SSIM
from bayes_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_dataset
from bayes_dip.utils.evaluation_utils import translate_path, recompute_reconstruction
from bayes_dip.data.datasets.walnut import get_walnut_2d_inner_part_defined_by_patch_size

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, help='hydra output directory')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--include_outer_part', action='store_true', default=False, help='include the outer part of the walnut image (that only contains background)')
parser.add_argument('--define_inner_part_by_patch_size', type=int, default=1, help='patch size defining the effective inner part (due to not necessarily aligned patches)')
parser.add_argument('--save_to', type=str, nargs='?', default='')
args = parser.parse_args()

run = args.run

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

run = translate_path(
        run,
        experiment_paths=experiment_paths)

try:
    ground_truth = torch.load(
            os.path.join(run, f'sample_0.pt'), map_location='cpu')['ground_truth']
except FileNotFoundError:
    print('did not find sample data, will recompute')
    cfg = OmegaConf.load(os.path.join(run, '.hydra', 'config.yaml'))
    dtype = torch.double if cfg.use_double else torch.float
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    ray_trafo = get_standard_ray_trafo(cfg)
    ray_trafo.to(dtype=dtype, device=device)
    dataset = get_standard_dataset(
            cfg, ray_trafo, fold=cfg.dataset.fold, use_fixed_seeds_starting_from=cfg.seed,
            device=device)
    data_sample = next(DataLoader(dataset))
    # data: observation, ground_truth, filtbackproj
    _, ground_truth, _ = data_sample

try:
    recon = torch.load(os.path.join(run, 'recon_0.pt'), map_location='cpu')
except FileNotFoundError:
    print('did not find reconstruction, will recompute')
    recon = recompute_reconstruction(run, sample_idx=0, experiment_paths=experiment_paths)

if not args.include_outer_part:
    slice_0, slice_1 = get_walnut_2d_inner_part_defined_by_patch_size(
            args.define_inner_part_by_patch_size)
    print(f'restricting to image pixels {slice_0}, {slice_1}')
    recon = recon[:, :, slice_0, slice_1]
    ground_truth = ground_truth[:, :, slice_0, slice_1]
else:
    print('using full image')

psnr = PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())
ssim = SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy())

result = {
    'psnr': psnr.item(),
    'ssim': ssim.item(),
}
print(f'walnut')
print(f'psnr / ssim: {psnr} / {ssim}')
print()

if args.save_to:
    with open(args.save_to, 'w') as f:
        yaml.dump(result, f)
