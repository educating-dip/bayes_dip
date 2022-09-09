import os
import yaml
import argparse
import torch
import numpy as np
from bayes_dip.utils.evaluation_utils import (
        translate_output_path, restrict_sample_based_density_data_to_new_patch_idx_list)

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_walnut_sample_based_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--include_outer_part', action='store_true', default=False, help='include the outer part of the walnut image (that only contains background)')
args = parser.parse_args()

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

INCLUDE_PREDCP_LIST = [True, False]

for include_predcp in INCLUDE_PREDCP_LIST:
    run = runs[f'include_predcp_{include_predcp}']
    run = translate_output_path(run, outputs_path=args.experiments_outputs_path)
    data = torch.load(os.path.join(run, f'sample_based_predictive_posterior_0.pt'),
            map_location='cpu')
    log_prob_original = data['log_prob']
    patch_idx_list = None if args.include_outer_part else 'walnut_inner'
    data_restricted = restrict_sample_based_density_data_to_new_patch_idx_list(
            run_path=run, data=data, patch_idx_list=patch_idx_list,
            outputs_path=args.experiments_outputs_path)
    log_prob_restricted = data_restricted['log_prob']
    print(f'walnut sample based density for include_predcp={include_predcp}')
    print(f'(original log prob saved in file, with potentially different patch selection: {log_prob_original})')
    print(f'log prob: {log_prob_restricted}')
