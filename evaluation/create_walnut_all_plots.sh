#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts
walnut_data_path=../../../bayes_dip/scripts/walnuts

python plot_walnut_main.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_reweight_off_diagonal_entries/patch_size_1.yaml --baseline_mcdo_runs_file $yaml_root_path/runs_baseline_walnut_mcdo_density_reweight_off_diagonal_entries/patch_size_1.yaml --walnut_data_path $walnut_data_path --save_data_to walnut_main_figure_data.pt

python plot_walnut_mini.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --save_data_to walnut_mini_figure_data_include_predcp_True.pt
python plot_walnut_mini.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --do_not_use_predcp --save_data_to walnut_mini_figure_data_include_predcp_False.pt

python plot_walnut_histogram.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --save_data_to walnut_histogram_figure_data.pt
python plot_walnut_histogram.py --runs_file $yaml_root_path/runs_walnut_sample_based_density.yaml --do_not_use_log_yscale --load_data_from walnut_histogram_figure_data.pt

python plot_walnut_hyperparams.py --runs_file $yaml_root_path/runs_walnut_dip_mll_optim.yaml --tag_list GPprior_lengthscale_{0..10} GPprior_variance_{0..10} --suffix _gppriors --rows 4 --skip_sub_plots 6 18 --legend_pos 23 --legend_loc "upper right" --legend_bbox_to_anchor 1.05 1. --wspace 0.4 --save_data_to walnut_hyperparams_figure_data.pt
python plot_walnut_hyperparams.py --runs_file $yaml_root_path/runs_walnut_dip_mll_optim.yaml --tag_list NormalPrior_variance_{0..2} observation_noise_variance --suffix _normalpriors --rows 1 --skip_sub_plots 4 --legend_pos 3 --legend_loc "upper right" --legend_bbox_to_anchor 1.05 1. --wspace 0.4 --load_data_from walnut_hyperparams_figure_data.pt
python plot_walnut_hyperparams.py --runs_file $yaml_root_path/runs_walnut_dip_mll_optim.yaml --tag_list GPprior_lengthscale_0 GPprior_lengthscale_10 NormalPrior_variance_2 GPprior_variance_0 GPprior_variance_10 observation_noise_variance --suffix _small --rows 2 --legend_pos 5 --load_data_from walnut_hyperparams_figure_data.pt

python plot_walnut_uq_main.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_reweight_off_diagonal_entries/patch_size_1.yaml --runs_file_approx $yaml_root_path/runs_walnut_sample_based_density_approx_jacs_cg_reweight_off_diagonal_entries/patch_size_1.yaml --baseline_mcdo_runs_file $yaml_root_path/runs_baseline_walnut_mcdo_density_reweight_off_diagonal_entries/patch_size_1.yaml --walnut_data_path $walnut_data_path --save_data_to walnut_uq_main_figure_data.pt
