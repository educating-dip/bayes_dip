#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts

python plot_kmnist_histogram.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --sample_idx 1 --save_data_to kmnist_histogram_figure_data.pt
python plot_kmnist_histogram.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --sample_idx 1 --do_not_use_log_yscale --load_data_from kmnist_histogram_figure_data.pt
