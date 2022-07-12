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

# install torch as suggested on https://pytorch.org/get-started/locally/
conda install pytorch torchvision -c pytorch-nightly

# install other conda packages
conda install tensorboardx scikit-image imageio opt_einsum tqdm tabulate

# install gpytorch package
conda install gpytorch -c gpytorch

# install astra package (should install version >= 2.0.0)
conda install astra-toolbox -c astra-toolbox

# install pip packages
pip install hydra-core tensorly==0.6.0 https://github.com/odlgroup/odl/archive/master.zip bios functorch xitorch tensorboard

echo -e "created env ${env_path}; to activate it, use:\n\tconda activate ${env_path}"