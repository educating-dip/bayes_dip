#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts

## exact density
python create_kmnist_density_table.py --root_path=$yaml_root_path --save_to kmnist_density_table.tex


## sample based density for different patch sizes

# sample based density only for noise 0.05 and patch size up to 10, but with MCDO baseline
python create_kmnist_sample_based_density_table.py --noise_list 0.05 --patch_sizes 1 2 3 4 5 6 7 8 9 10 --root_path=$yaml_root_path --bayes_dip_folder results_kmnist_sample_based_density --save_to kmnist_sample_based_density_table_with_mcdo_baseline.tex
python create_kmnist_sample_based_density_table.py --noise_list 0.05 --patch_sizes 1 2 3 4 5 6 7 8 9 10 --root_path=$yaml_root_path --bayes_dip_folder results_kmnist_sample_based_density_reweight_off_diagonal_entries --save_to kmnist_sample_based_density_table_with_mcdo_baseline_reweight_off_diagonal_entries.tex

# sample based density
python create_kmnist_sample_based_density_table.py --root_path=$yaml_root_path --bayes_dip_folder results_kmnist_sample_based_density --baseline_mcdo_folder "" --save_to kmnist_sample_based_density_table.tex
python create_kmnist_sample_based_density_table.py --root_path=$yaml_root_path --bayes_dip_folder results_kmnist_sample_based_density_reweight_off_diagonal_entries --baseline_mcdo_folder "" --save_to kmnist_sample_based_density_table_reweight_off_diagonal_entries.tex

# sample based density with approx Jacs
python create_kmnist_sample_based_density_table.py --root_path=$yaml_root_path --bayes_dip_folder results_kmnist_sample_based_density_approx_jacs --baseline_mcdo_folder "" --save_to kmnist_sample_based_density_table_approx_jacs.tex
python create_kmnist_sample_based_density_table.py --root_path=$yaml_root_path --bayes_dip_folder results_kmnist_sample_based_density_approx_jacs_reweight_off_diagonal_entries --baseline_mcdo_folder "" --save_to kmnist_sample_based_density_table_approx_jacs_reweight_off_diagonal_entries.tex

# sample based density with approx Jacs and CG inversion
python create_kmnist_sample_based_density_table.py --root_path=$yaml_root_path --bayes_dip_folder results_kmnist_sample_based_density_approx_jacs_cg --baseline_mcdo_folder "" --save_to kmnist_sample_based_density_table_approx_jacs_cg.tex
python create_kmnist_sample_based_density_table.py --root_path=$yaml_root_path --bayes_dip_folder results_kmnist_sample_based_density_approx_jacs_cg_reweight_off_diagonal_entries --baseline_mcdo_folder "" --save_to kmnist_sample_based_density_table_approx_jacs_cg_reweight_off_diagonal_entries.tex
