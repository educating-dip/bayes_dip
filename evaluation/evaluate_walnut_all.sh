#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts
# Bayes-DIP sample based density
mkdir -p $yaml_root_path/results_walnut_sample_based_density
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_walnut_sample_based_density_add_image_noise_correction_term
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_add_image_noise_correction_term/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_add_image_noise_correction_term/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_walnut_sample_based_density_reweight_off_diagonal_entries
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_walnut_sample_based_density_add_image_noise_correction_term_reweight_off_diagonal_entries
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_add_image_noise_correction_term_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_add_image_noise_correction_term_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done
# Bayes-DIP sample based density with approx Jac
mkdir -p $yaml_root_path/results_walnut_sample_based_density_approx_jacs
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_approx_jacs/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_approx_jacs/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_walnut_sample_based_density_approx_jacs_add_image_noise_correction_term
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_approx_jacs_add_image_noise_correction_term/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_approx_jacs_add_image_noise_correction_term/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_walnut_sample_based_density_approx_jacs_reweight_off_diagonal_entries
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_approx_jacs_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_approx_jacs_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_walnut_sample_based_density_approx_jacs_add_image_noise_correction_term_reweight_off_diagonal_entries
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_approx_jacs_add_image_noise_correction_term_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_approx_jacs_add_image_noise_correction_term_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done
# Bayes-DIP sample based density with approx Jac and CG inversion
mkdir -p $yaml_root_path/results_walnut_sample_based_density_approx_jacs_cg
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_approx_jacs_cg/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_approx_jacs_cg/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_walnut_sample_based_density_approx_jacs_cg_add_image_noise_correction_term
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_approx_jacs_cg_add_image_noise_correction_term/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_approx_jacs_cg_add_image_noise_correction_term/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_walnut_sample_based_density_approx_jacs_cg_reweight_off_diagonal_entries
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_approx_jacs_cg_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_approx_jacs_cg_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_walnut_sample_based_density_approx_jacs_cg_add_image_noise_correction_term_reweight_off_diagonal_entries
for patch_size in $(seq 1 10); do python evaluate_walnut_sample_based_density.py --runs_file $yaml_root_path/runs_walnut_sample_based_density_approx_jacs_cg_add_image_noise_correction_term_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_walnut_sample_based_density_approx_jacs_cg_add_image_noise_correction_term_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done
# baselines
mkdir -p $yaml_root_path/results_baseline_walnut_mcdo_density
for patch_size in $(seq 1 10); do python evaluate_baseline_walnut_mcdo_density.py --runs_file $yaml_root_path/runs_baseline_walnut_mcdo_density/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_baseline_walnut_mcdo_density/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_baseline_walnut_mcdo_density_add_image_noise_correction_term
for patch_size in $(seq 1 10); do python evaluate_baseline_walnut_mcdo_density.py --runs_file $yaml_root_path/runs_baseline_walnut_mcdo_density_add_image_noise_correction_term/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_baseline_walnut_mcdo_density_add_image_noise_correction_term/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_baseline_walnut_mcdo_density_reweight_off_diagonal_entries
for patch_size in $(seq 1 10); do python evaluate_baseline_walnut_mcdo_density.py --runs_file $yaml_root_path/runs_baseline_walnut_mcdo_density_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_baseline_walnut_mcdo_density_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_baseline_walnut_mcdo_density_add_image_noise_correction_term_reweight_off_diagonal_entries
for patch_size in $(seq 1 10); do python evaluate_baseline_walnut_mcdo_density.py --runs_file $yaml_root_path/runs_baseline_walnut_mcdo_density_add_image_noise_correction_term_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_baseline_walnut_mcdo_density_add_image_noise_correction_term_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done


# DIP PSNR and SSIM
# (use a dip_mll_optim run instead of the dip run, because the dip run is from an old code base with different hydra config structure)
python evaluate_walnut_psnr_ssim.py --run $(python -c "import yaml; import os; f = open('$yaml_root_path/runs_walnut_dip_mll_optim.yaml', 'r'); d = yaml.safe_load(f); f.close(); print(d['include_predcp_False'])") --save_to $yaml_root_path/psnr_ssim_walnut_dip.yaml
python evaluate_walnut_psnr_ssim.py --run $(python -c "import yaml; import os; f = open('$yaml_root_path/runs_walnut_dip_mll_optim.yaml', 'r'); d = yaml.safe_load(f); f.close(); print(d['include_predcp_False'])") --include_outer_part --save_to $yaml_root_path/psnr_ssim_walnut_dip_include_outer_part.yaml

# baseline MCDO-DIP PSNR and SSIM
python evaluate_baseline_walnut_mcdo_dip_psnr_ssim.py --runs_file $yaml_root_path/runs_baseline_walnut_mcdo_density/patch_size_1.yaml --save_to $yaml_root_path/psnr_ssim_baseline_walnut_mcdo_dip.yaml
python evaluate_baseline_walnut_mcdo_dip_psnr_ssim.py --runs_file $yaml_root_path/runs_baseline_walnut_mcdo_density/patch_size_1.yaml --include_outer_part --save_to $yaml_root_path/psnr_ssim_baseline_walnut_mcdo_dip_include_outer_part.yaml
