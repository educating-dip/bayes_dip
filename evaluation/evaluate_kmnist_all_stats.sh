#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts
# Bayes-DIP
python evaluate_kmnist_exact_density.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --save_stats_to $yaml_root_path/stats_kmnist_exact_density.yaml
# baselines
python evaluate_baseline_kmnist_mcdo_density.py --runs_file $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml --save_stats_to $yaml_root_path/stats_baseline_kmnist_mcdo_density.yaml
python evaluate_baseline_kmnist_deterministic_density.py --runs_file $yaml_root_path/runs_baseline_kmnist_deterministic_density.yaml --save_stats_to $yaml_root_path/stats_baseline_kmnist_deterministic_density.yaml
