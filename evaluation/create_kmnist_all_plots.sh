#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts

python plot_kmnist_histogram.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --sample_idx 1
python plot_kmnist_histogram.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --sample_idx 1 --do_not_use_log_yscale
