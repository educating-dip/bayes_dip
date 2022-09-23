import os
import yaml
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from bayes_dip.utils.evaluation_utils import (
        translate_path, restrict_sample_based_density_data_to_new_patch_idx_list)

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_baseline_walnut_mcdo_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--include_outer_part', action='store_true', default=False, help='include the outer part of the walnut image (that only contains background)')
parser.add_argument('--save_to', type=str, nargs='?', default='')
args = parser.parse_args()

with open(args.runs_file, 'r') as f:
    run = yaml.safe_load(f)

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

result = None

run = translate_path(
        run,
        experiment_paths=experiment_paths)
data = torch.load(os.path.join(run, f'sample_based_mcdo_predictive_posterior_0.pt'),
        map_location='cpu')
log_prob_original = data['log_prob']
patch_idx_list = None if args.include_outer_part else 'walnut_inner'
cfg = OmegaConf.load(os.path.join(run, '.hydra', 'config.yaml'))
data_restricted = restrict_sample_based_density_data_to_new_patch_idx_list(
        data=data,
        patch_idx_list=patch_idx_list,
        orig_patch_idx_list=cfg.inference.patch_idx_list,
        patch_size=cfg.inference.patch_size,
        im_shape=(cfg.dataset.im_size,) * 2)
log_prob_restricted = data_restricted['log_prob']
result = log_prob_restricted.item()
print(f'walnut mcdo sample based density')
print(f'(original log prob saved in file, with potentially different patch selection: {log_prob_original})')
print(f'log prob: {log_prob_restricted}')

if args.save_to:
    with open(args.save_to, 'w') as f:
        yaml.dump(result, f)
