#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts
walnut_data_path=../../../bayes_dip/scripts/walnuts

python plot_walnut_main.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_reweight_off_diagonal_entries/patch_size_1.yaml --baseline_mcdo_runs_file $yaml_root_path/runs_baseline_walnut_mcdo_density_reweight_off_diagonal_entries/patch_size_1.yaml --walnut_data_path $walnut_data_path --save_data_to walnut_main_figure_data.pt

python plot_walnut_mini.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --save_data_to walnut_mini_figure_data_include_predcp_True.pt
python plot_walnut_mini.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --do_not_use_predcp --save_data_to walnut_mini_figure_data_include_predcp_False.pt

python plot_walnut_histogram.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --save_data_to walnut_histogram_figure_data_log_yscale.pt
python plot_walnut_histogram.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --do_not_use_log_yscale --save_data_to walnut_histogram_figure_data_linear_yscale.pt
