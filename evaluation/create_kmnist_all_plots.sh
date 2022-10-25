#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts
hmc_priors_data_file=../../../bayes_dip/experiments/old_results/ICML_HMC/prior_HMC/prior_samples.pickle

python plot_kmnist_histogram.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --sample_idx 1 --save_data_to kmnist_histogram_figure_data.pt
python plot_kmnist_histogram.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --sample_idx 1 --do_not_use_log_yscale --load_data_from kmnist_histogram_figure_data.pt

python plot_kmnist_main.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --baseline_mcdo_runs_file  $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml  --sample_idx 4 --noise 0.05 --angles 5  --save_data_to kmnist_main_figure_data_4_0.05_5.pt --hist_xlim_max 0.6
python plot_kmnist_main.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --baseline_mcdo_runs_file  $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml  --sample_idx 4 --noise 0.05 --angles 10  --save_data_to kmnist_main_figure_data_4_0.05_10.pt --hist_xlim_max 0.4
python plot_kmnist_main.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --baseline_mcdo_runs_file  $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml  --sample_idx 4 --noise 0.05 --angles 20 --save_data_to kmnist_main_figure_data_4_0.05_20.pt --hist_xlim_max 0.3
python plot_kmnist_main.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --baseline_mcdo_runs_file  $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml  --sample_idx 4 --noise 0.05 --angles 30  --save_data_to kmnist_main_figure_data_4_0.05_30.pt --hist_xlim_max 0.2
python plot_kmnist_main.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --baseline_mcdo_runs_file  $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml  --sample_idx 4 --noise 0.1 --angles 5  --save_data_to kmnist_main_figure_data_4_0.1_5.pt --hist_xlim_max 0.7
python plot_kmnist_main.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --baseline_mcdo_runs_file  $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml  --sample_idx 4 --noise 0.1 --angles 10  --save_data_to kmnist_main_figure_data_4_0.1_10.pt --hist_xlim_max 0.7
python plot_kmnist_main.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --baseline_mcdo_runs_file  $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml  --sample_idx 4 --noise 0.1 --angles 20  --save_data_to kmnist_main_figure_data_4_0.1_20.pt --hist_xlim_max 0.5
python plot_kmnist_main.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --baseline_mcdo_runs_file  $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml  --sample_idx 4 --noise 0.1 --angles 30  --save_data_to kmnist_main_figure_data_4_0.1_30.pt --hist_xlim_max 0.5

<<<<<<< Updated upstream
python plot_kmnist_hyperparams.py --runs_file $yaml_root_path/runs_kmnist_exact_dip_mll_optim.yaml --tag_list GPprior_lengthscale_0 GPprior_lengthscale_1 GPprior_lengthscale_2 GPprior_lengthscale_3 GPprior_lengthscale_4 GPprior_variance_0 GPprior_variance_1 GPprior_variance_2 GPprior_variance_3 GPprior_variance_4 --suffix _gppriors --rows 2 --wspace 0.4 --noise 0.05 --angles 5 --load_data_from kmnist_hyperparams_figure_data_0.05_5.pt
python plot_kmnist_hyperparams.py --runs_file $yaml_root_path/runs_kmnist_exact_dip_mll_optim.yaml --tag_list NormalPrior_variance_{0..1} NormalPrior_variance_2 observation_noise_variance --suffix _normalpriors --rows 1 --wspace 0.4 --noise 0.05 --angles 5 --load_data_from kmnist_hyperparams_figure_data_0.05_5.pt

python plot_kmnist_sample_based_vs_exact.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --runs_folder_sample_based $yaml_root_path/runs_kmnist_sample_based_density_varying_num_samples/ --num_subplots 2 --ylim_min 2.5 --save_data_to kmnist_sample_based_vs_exact_figure_data.pt

python plot_kmnist_tv_hists_and_samples_from_dists.py --runs_file $yaml_root_path/runs_kmnist_exact_dip_mll_optim.yaml --hmc_priors_data_file $hmc_priors_data_file --save_data_to kmnist_tv_hists_and_samples_from_dists_figure_data.pt
=======
python plot_kmnist_hyperparams.py --runs_file $yaml_root_path/runs_kmnist_exact_dip_mll_optim.yaml --tag_list GPprior_lengthscale_0 GPprior_lengthscale_1 GPprior_lengthscale_2 GPprior_lengthscale_3 GPprior_lengthscale_4 GPprior_variance_0 GPprior_variance_1 GPprior_variance_2 GPprior_variance_3 GPprior_variance_4 --suffix _gppriors --rows 2 --wspace 0.4 --noise 0.05 --angles 5 --load_data_from kmnist_hyperparams_figure_data_0.05_5.pt 
python plot_kmnist_hyperparams.py --runs_file $yaml_root_path/runs_kmnist_exact_dip_mll_optim.yaml --tag_list NormalPrior_variance_{0..1} NormalPrior_variance_2 observation_noise_variance --suffix _normalpriors --rows 1 --wspace 0.4 --noise 0.05 --angles 5 --load_data_from kmnist_hyperparams_figure_data_0.05_5.pt 

python linear_sweep_bijectivity.py --runs_file $yaml_root_path/runs_kmnist_dip.yaml --do_not_fix_marginal_1 --num_samples 10000 --sweep_grid_points 100 --save_data_to biject_False.pt
python linear_sweep_bijectivity.py --runs_file $yaml_root_path/runs_kmnist_dip.yaml --num_samples 10000 --sweep_grid_points 100 --save_data_to biject_True.pt
>>>>>>> Stashed changes
