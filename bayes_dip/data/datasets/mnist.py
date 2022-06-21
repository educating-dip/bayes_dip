"""
Provides :class:`RectanglesDataset`.
"""

import os
import torch
from torch import Tensor
from torchvision import datasets, transforms
from bayes_dip.utils import get_original_cwd


class MNISTImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type, path, train):

        path = os.path.join(get_original_cwd(), path)

        self.dataset = dataset_type(
                root=path, train=train, download=True,
                transform=transforms.ToTensor())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tensor:
        return self.dataset[idx][0]  # only the image, not the label


def get_mnist_testset(path='.'):
    return MNISTImageDataset(datasets.MNIST, path, train=False)


def get_mnist_trainset(path='.'):
    return MNISTImageDataset(datasets.MNIST, path, train=True)


def get_kmnist_testset(path='.'):
    return MNISTImageDataset(datasets.KMNIST, path, train=False)


def get_kmnist_trainset(path='.'):
    return MNISTImageDataset(datasets.KMNIST, path, train=True)
