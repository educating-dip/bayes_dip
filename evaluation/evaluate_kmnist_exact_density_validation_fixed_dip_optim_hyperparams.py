import os
import yaml
import argparse
from itertools import product
import torch
import numpy as np
from bayes_dip.utils.evaluation_utils import translate_path

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_kmnist_exact_density_validation_fixed_dip_optim_hyperparams.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--experiments_multirun_path', type=str, default='../experiments/multirun', help='base path containing the hydra multirun directories (usually "[...]/multirun/")')
parser.add_argument('--print_individual_log_probs', action='store_true', default=False)
args = parser.parse_args()

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

experiment_paths = {
        'outputs_path': args.experiments_outputs_path,
        'multirun_path': args.experiments_multirun_path,
}

NOISE_LIST = [0.05, 0.1]
ANGLES_LIST = [5, 10, 20, 30]
INCLUDE_PREDCP = True
PREDCP_SCALE_LIST = [0.01, 0.1, 1., 10., 100.]
NUM_IMAGES = 10

mean_log_prob_dict = {}

for noise in NOISE_LIST:
    for angles in ANGLES_LIST:
        mean_log_prob_dict[(noise, angles)] = {}
        for predcp_scale in PREDCP_SCALE_LIST:
            run = runs[noise][angles][f'include_predcp_{INCLUDE_PREDCP}'][f'predcp_scale_{predcp_scale}']
            run = translate_path(run, experiment_paths=experiment_paths)
            log_probs = []
            for i in range(NUM_IMAGES):
                d = torch.load(os.path.join(run, f'exact_predictive_posterior_{i}.pt'), map_location='cpu')
                log_probs.append(d['log_prob'])
            mean_log_prob = np.mean(log_probs)
            print(f'kmnist exact density for noise={noise} angles={angles} include_predcp={INCLUDE_PREDCP} predcp_scale={predcp_scale}')
            print(f'mean log prob: {mean_log_prob}')
            if args.print_individual_log_probs:
                print(f'individual log probs: {log_probs}')
            print()
            mean_log_prob_dict[(noise, angles)][predcp_scale] = mean_log_prob

print('\nCompare different PredCP scales for each setting:\n')
for noise, angles in product(NOISE_LIST, ANGLES_LIST):
    predcp_scales = list(mean_log_prob_dict[(noise, angles)].keys())
    mean_log_probs = list(mean_log_prob_dict[(noise, angles)].values())
    print(f'noise={noise} angles={angles}:')
    print('\t'.join(('<max>' if mean_log_prob == max(mean_log_probs) else '').rjust(7)
            for mean_log_prob in mean_log_probs))
    print('\t'.join(f'{predcp_scale}'.rjust(7) for predcp_scale in predcp_scales))
    print('\t'.join(f'{mean_log_prob:.4f}'.rjust(7) for mean_log_prob in mean_log_probs))
    print()
