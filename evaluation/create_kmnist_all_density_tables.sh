#!/bin/sh
yaml_root_path=../../../bayes_dip/scripts
python create_kmnist_density_table.py --root_path=$yaml_root_path --save_to kmnist_density_table.tex
python create_kmnist_sample_based_density_table.py --noise_list 0.05 --root_path=$yaml_root_path --save_to kmnist_sample_based_density_table.tex
