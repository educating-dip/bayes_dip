"""
Provides the MNIST and KMNIST image datasets.
"""

import os
import torch
from torch import Tensor
from torchvision import datasets, transforms
from bayes_dip.utils import get_original_cwd


class MNISTImageDataset(torch.utils.data.Dataset):
    """
    Torch dataset wrapper for the (K)MNIST images.
    """
    def __init__(self, dataset_type, path, train):
        """
        Parameters
        ----------
        dataset_type : callable
            Either ``torchvision.datasets.MNIST`` or ``torchvision.datasets.KMNIST``.
        path : str
            Root path for storing the dataset, either absolute or relative to the original current
            working directory.
        train : bool
            Whether to use the training images (or the test images otherwise).
        """
        path = os.path.join(get_original_cwd(), path)

        self.dataset = dataset_type(
                root=path, train=train, download=True,
                transform=transforms.ToTensor())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tensor:
        return self.dataset[idx][0]  # only the image, not the label


def get_mnist_testset(path='.'):
    """
    Return the MNIST image test dataset.

    Parameters
    ----------
    path : str, optional
        Root path for storing the dataset, either absolute or relative to the original current
        working directory. The default is ``'.'``, i.e. the original current working directory.
    """
    return MNISTImageDataset(datasets.MNIST, path, train=False)


def get_mnist_trainset(path='.'):
    """
    Return the MNIST image training dataset.

    Parameters
    ----------
    path : str, optional
        Root path for storing the dataset, either absolute or relative to the original current
        working directory. The default is ``'.'``, i.e. the original current working directory.
    """
    return MNISTImageDataset(datasets.MNIST, path, train=True)


def get_kmnist_testset(path='.'):
    """
    Return the KMNIST image test dataset.

    Parameters
    ----------
    path : str, optional
        Root path for storing the dataset, either absolute or relative to the original current
        working directory. The default is ``'.'``, i.e. the original current working directory.
    """
    return MNISTImageDataset(datasets.KMNIST, path, train=False)


def get_kmnist_trainset(path='.'):
    """
    Return the KMNIST image training dataset.

    Parameters
    ----------
    path : str, optional
        Root path for storing the dataset, either absolute or relative to the original current
        working directory. The default is ``'.'``, i.e. the original current working directory.
    """
    return MNISTImageDataset(datasets.KMNIST, path, train=True)
