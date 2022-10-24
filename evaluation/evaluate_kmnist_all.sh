#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts

# Bayes-DIP exact density
python evaluate_kmnist_exact_density.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --save_to $yaml_root_path/results_kmnist_exact_density.yaml
# Bayes-DIP exact density, using angles 20 and 10 images to compare with sample based density
python evaluate_kmnist_exact_density.py --runs_file $yaml_root_path/runs_kmnist_exact_density.yaml --angles_list 20 --num_images 10 --save_to $yaml_root_path/results_kmnist_exact_density_first_10_images.yaml

# Bayes-DIP sample based density
python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density.yaml
# Bayes-DIP sample based density for different patch sizes
mkdir -p $yaml_root_path/results_kmnist_sample_based_density
for patch_size in $(seq 1 28); do python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_kmnist_sample_based_density_reweight_off_diagonal_entries
for patch_size in $(seq 1 28); do python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done
# Bayes-DIP sample based density for different patch sizes and different numbers of samples
mkdir -p $yaml_root_path/results_kmnist_sample_based_density_varying_num_samples
for num_samples in 32 64 128 256 512 1024 2048 4096 8192 16384; do for patch_size in 1 2 4 7 14 28; do python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density_varying_num_samples/patch_size_${patch_size}_num_samples_${num_samples}.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density_varying_num_samples/patch_size_${patch_size}_num_samples_${num_samples}.yaml; done; done
mkdir -p $yaml_root_path/results_kmnist_sample_based_density_varying_num_samples_reweight_off_diagonal_entries
for num_samples in 32 64 128 256 512 1024 2048 4096 8192 16384; do for patch_size in 1 2 4 7 14 28; do python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density_varying_num_samples_reweight_off_diagonal_entries/patch_size_${patch_size}_num_samples_${num_samples}.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density_varying_num_samples_reweight_off_diagonal_entries/patch_size_${patch_size}_num_samples_${num_samples}.yaml; done; done

# Bayes-DIP sample based density with approx Jacs
python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density_approx_jacs.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density_approx_jacs.yaml
# Bayes-DIP sample based density with approx Jacs for different patch sizes
mkdir -p $yaml_root_path/results_kmnist_sample_based_density_approx_jacs
for patch_size in $(seq 1 28); do python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density_approx_jacs/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density_approx_jacs/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_kmnist_sample_based_density_approx_jacs_reweight_off_diagonal_entries
for patch_size in $(seq 1 28); do python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density_approx_jacs_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density_approx_jacs_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done

# Bayes-DIP sample based density with approx Jacs and CG inversion
python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density_approx_jacs_cg.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density_approx_jacs_cg.yaml
# Bayes-DIP sample based density with approx Jacs and CG inversion for different patch sizes
mkdir -p $yaml_root_path/results_kmnist_sample_based_density_approx_jacs_cg
for patch_size in $(seq 1 28); do python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density_approx_jacs_cg/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density_approx_jacs_cg/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_kmnist_sample_based_density_approx_jacs_cg_reweight_off_diagonal_entries
for patch_size in $(seq 1 28); do python evaluate_kmnist_sample_based_density.py --runs_file $yaml_root_path/runs_kmnist_sample_based_density_approx_jacs_cg_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_kmnist_sample_based_density_approx_jacs_cg_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done

# baselines
python evaluate_baseline_kmnist_mcdo_density.py --runs_file $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml --save_to $yaml_root_path/results_baseline_kmnist_mcdo_density.yaml
python evaluate_baseline_kmnist_deterministic_density.py --runs_file $yaml_root_path/runs_baseline_kmnist_deterministic_density.yaml --save_to $yaml_root_path/results_baseline_kmnist_deterministic_density.yaml
# baseline MCDO for different patch sizes, using noise 0.05, angles 20 and 10 images to compare with Bayes-DIP sample based density for different patch sizes
mkdir -p $yaml_root_path/results_baseline_kmnist_mcdo_density
for patch_size in $(seq 1 10); do python evaluate_baseline_kmnist_mcdo_density.py --noise_list 0.05 --angles_list 20 --num_images 10 --runs_file $yaml_root_path/runs_baseline_kmnist_mcdo_density/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_baseline_kmnist_mcdo_density/patch_size_$patch_size.yaml; done
mkdir -p $yaml_root_path/results_baseline_kmnist_mcdo_density_reweight_off_diagonal_entries
for patch_size in $(seq 1 10); do python evaluate_baseline_kmnist_mcdo_density.py --noise_list 0.05 --angles_list 20 --num_images 10 --runs_file $yaml_root_path/runs_baseline_kmnist_mcdo_density_reweight_off_diagonal_entries/patch_size_$patch_size.yaml --save_to $yaml_root_path/results_baseline_kmnist_mcdo_density_reweight_off_diagonal_entries/patch_size_$patch_size.yaml; done


# DIP PSNR and SSIM
python evaluate_kmnist_psnr_ssim.py --runs_file $yaml_root_path/runs_kmnist_dip.yaml --save_to $yaml_root_path/psnr_ssim_kmnist_dip.yaml

# baseline MCDO-DIP PSNR and SSIM
python evaluate_baseline_kmnist_mcdo_dip_psnr_ssim.py --runs_file $yaml_root_path/runs_baseline_kmnist_mcdo_density.yaml --save_to $yaml_root_path/psnr_ssim_baseline_kmnist_mcdo_dip.yaml
