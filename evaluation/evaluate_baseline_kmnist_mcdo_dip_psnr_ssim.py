import os
import yaml
import argparse
import torch
import numpy as np
from bayes_dip.utils import PSNR, SSIM
from bayes_dip.utils.evaluation_utils import translate_path
from baselines.evaluation_utils import compute_mcdo_reconstruction

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_baseline_kmnist_mcdo_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--save_to', type=str, nargs='?', default='')
args = parser.parse_args()

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

NOISE_LIST = [0.05, 0.1]
ANGLES_LIST = [5, 10, 20, 30]
NUM_IMAGES = 50

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

stats = {}

for noise in NOISE_LIST:
    stats.setdefault(noise, {})
    for angles in ANGLES_LIST:
        stats[noise].setdefault(angles, {})
        run = runs[noise][angles]
        run = translate_path(
                run,
                experiment_paths=experiment_paths)
        psnrs = []
        ssims = []
        has_recons = os.path.isfile(os.path.join(run, f'mcdo_recon_0.pt'))
        if not has_recons:
            print('did not find reconstructions, will recompute')
        for i in range(NUM_IMAGES):
            ground_truth = torch.load(
                    os.path.join(run, f'sample_{i}.pt'), map_location='cpu')['ground_truth']
            recon = (
                    torch.load(os.path.join(run, f'mcdo_recon_{i}.pt'), map_location='cpu') if has_recons
                    else compute_mcdo_reconstruction(run, sample_idx=i))
            psnrs.append(PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
            ssims.append(SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

        mean_psnr = np.mean(psnrs)
        mean_ssim = np.mean(ssims)
        stderr_psnr = np.std(psnrs) / np.sqrt(len(psnrs))
        stderr_ssim = np.std(ssims) / np.sqrt(len(ssims))
        stats[noise][angles] = {
            'psnr': {
                'mean': mean_psnr.item(),
                'stderr': stderr_psnr.item(),
            },
            'ssim': {
                'mean': mean_ssim.item(),
                'stderr': stderr_ssim.item(),
            },
        }
        print(f'kmnist noise={noise} angles={angles}')
        print(f'mean psnr / mean ssim: {mean_psnr} / {mean_ssim}')
        print()

if args.save_to:
    with open(args.save_to, 'w') as f:
        yaml.dump(stats, f)
