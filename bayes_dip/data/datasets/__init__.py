"""
Provides synthetic image datasets and access to stored datasets.
"""

from .rectangles import RectanglesDataset
from .mnist import get_mnist_testset, get_mnist_trainset, get_kmnist_testset, get_kmnist_trainset
from .walnut import (
        get_walnut_2d_observation, get_walnut_2d_ground_truth, get_walnut_2d_inner_patch_indices)
