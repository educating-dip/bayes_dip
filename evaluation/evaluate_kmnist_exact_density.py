import os
import yaml
import torch
import numpy as np

RUNS_FILE = 'runs_kmnist_exact_density.yaml'
EXPERIMENTS_OUTPUTS_PATH = '../experiments/outputs'

with open(RUNS_FILE, 'r') as f:
    runs = yaml.safe_load(f)

NOISE_LIST = [0.05, 0.1]
ANGLES_LIST = [5, 10, 20, 30]
INCLUDE_PREDCP_LIST = [False]  # [True, False]
NUM_IMAGES = 50

for noise in NOISE_LIST:
    for angles in ANGLES_LIST:
        for include_predcp in INCLUDE_PREDCP_LIST:
            run = runs[noise][angles][f'include_predcp_{include_predcp}']
            run = os.path.join(EXPERIMENTS_OUTPUTS_PATH, os.path.basename(run.rstrip('/')))
            log_probs = []
            for i in range(NUM_IMAGES):
                d = torch.load(os.path.join(run, f'exact_predictive_posterior_{i}.pt'), map_location='cpu')
                log_probs.append(d['log_prob'])
            mean_log_prob = np.mean(log_probs)
            print(f'kmnist exact density for noise={noise} angles={angles} include_predcp={include_predcp}')
            print(f'mean log prob: {mean_log_prob}')
            print(f'individual log probs: {log_probs}')
            print()
