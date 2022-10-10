#!/bin/bash

if [[ $- != *i* ]]
then
    echo "Please run in interactive mode, i.e. bash -i ...; aborting."
    exit 1
fi

print_usage () {
    echo "usage:"
    echo -e "\tbash -i $0 -n environment_name "
    echo -e "\tbash -i $0 -p path_to_env "
}

if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
fi
if [[ "$1" != "-n" ]] && [[ "$1" != "-p" ]]; then
    print_usage
    exit 1
fi
if [[ $# -ge 3 ]]; then
    print_usage
    exit 1
fi

flag=$1
env_path=$2

# exit when any command fails
set -e

# create and activate conda env
conda create $flag $env_path
conda activate $env_path

# setup channels
conda config --prepend channels pytorch
conda config --append channels gpytorch
conda config --append channels astra-toolbox
conda config --append channels conda-forge

# install pytorch
conda install pytorch=1.12 torchvision cudatoolkit=11.6 -c pytorch

# install other conda packages
conda install tensorboard tensorboardx scikit-image imageio opt_einsum tqdm gpytorch astra-toolbox

# install pip packages
pip install hydra-core https://github.com/odlgroup/odl/archive/master.zip functorch

echo -e "created env ${env_path}; to activate it, use:\n\tconda activate ${env_path}"
