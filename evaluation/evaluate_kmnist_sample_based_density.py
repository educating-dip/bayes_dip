import os
import yaml
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--runs_file', type=str, default='runs_kmnist_sample_based_density.yaml', help='path of yaml file containing hydra output directory names')
parser.add_argument('--experiments_outputs_path', type=str, default='../experiments/outputs', help='base path containing the hydra output directories (usually "[...]/outputs/")')
parser.add_argument('--print_individual_log_probs', action='store_true', default=False)
args = parser.parse_args()

with open(args.runs_file, 'r') as f:
    runs = yaml.safe_load(f)

NOISE_LIST = [0.05, 0.1]
ANGLES_LIST = [20]
INCLUDE_PREDCP_LIST = [False]
NUM_IMAGES = 10

for noise in NOISE_LIST:
    for angles in ANGLES_LIST:
        for include_predcp in INCLUDE_PREDCP_LIST:
            run = runs[noise][angles][f'include_predcp_{include_predcp}']
            run = os.path.join(args.experiments_outputs_path, os.path.basename(run.rstrip('/')))
            log_probs = []
            for i in range(NUM_IMAGES):
                d = torch.load(os.path.join(run, f'sample_based_predictive_posterior_{i}.pt'), map_location='cpu')
                log_probs.append(d['log_prob'])
            mean_log_prob = np.mean(log_probs)
            print(f'kmnist sample based density for noise={noise} angles={angles} include_predcp={include_predcp}')
            print(f'mean log prob: {mean_log_prob}')
            if args.print_individual_log_probs:
                print(f'individual log probs: {log_probs}')
            print()