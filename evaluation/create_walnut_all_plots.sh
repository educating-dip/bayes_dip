#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts
walnut_data_path=../../../bayes_dip/scripts/walnuts

python plot_walnut_main.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_reweight_off_diagonal_entries/patch_size_1.yaml --baseline_mcdo_runs_file $yaml_root_path/runs_baseline_walnut_mcdo_density_reweight_off_diagonal_entries/patch_size_1.yaml --walnut_data_path $walnut_data_path

python plot_walnut_mini.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml
python plot_walnut_mini.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --do_not_use_predcp

python plot_walnut_histogram.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml
python plot_walnut_histogram.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --do_not_use_log_yscale
