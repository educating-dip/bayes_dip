#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts

python create_walnut_psnr_ssim_table.py --dip_file=$yaml_root_path/psnr_ssim_walnut_dip.yaml --mcdo_dip_file=$yaml_root_path/psnr_ssim_baseline_walnut_mcdo_dip.yaml --save_to walnut_psnr_ssim_table.tex
