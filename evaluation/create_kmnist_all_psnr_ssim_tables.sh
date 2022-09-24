#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts

python create_kmnist_psnr_ssim_table.py --dip_file=$yaml_root_path/psnr_ssim_kmnist_dip.yaml --mcdo_dip_file=$yaml_root_path/psnr_ssim_baseline_kmnist_mcdo_dip.yaml --save_to kmnist_psnr_ssim_table.tex
