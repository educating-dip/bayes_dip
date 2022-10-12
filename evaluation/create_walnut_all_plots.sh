#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts

python plot_walnut_mini.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml
python plot_walnut_mini.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --do_not_use_predcp

python plot_walnut_histogram.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml
python plot_walnut_histogram.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --do_not_use_log_yscale
