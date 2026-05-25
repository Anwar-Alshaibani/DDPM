"""
CIFAR-10 loader normalized to [-1, 1] for diffusion training.
"""
from pathlib import Path

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def get_cifar10(root="./data", train=True, augment=True):
    Path(root).mkdir(parents=True, exist_ok=True)
    ops = []
    if train and augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    return torchvision.datasets.CIFAR10(
        root=root, train=train, transform=transforms.Compose(ops), download=True
    )


def make_loader(ds, batch_size, shuffle, num_workers=2):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=shuffle)
