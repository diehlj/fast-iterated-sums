from .cifar import CIFAR10DataModule, CIFAR100DataModule
from .mnist import MNISTDataModule, FashionMNISTDataModule

__all__ = [
    "CIFAR10DataModule",
    "CIFAR100DataModule", 
    "MNISTDataModule",
    "FashionMNISTDataModule"
]
