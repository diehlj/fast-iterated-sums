from torchvision.datasets import MNIST, FashionMNIST

from .base_datamodule import BaseDataModule


class MNISTAlbumentation(MNIST):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class FashionMNISTAlbumentation(FashionMNIST):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class MNISTDataModule(BaseDataModule):
    num_classes = 10

    def __init__(self, **params):
        super(MNISTDataModule, self).__init__(**params)

    def setup(self, stage: str):
        if "albumentations" in str(self.train_transforms.__class__):
            self.train_dataset = MNISTAlbumentation(
                self.data_path, train=True, transform=self.train_transforms, download=True
            )
        else:
            self.train_dataset = MNIST(self.data_path, train=True, transform=self.train_transforms, download=True)

        if "albumentations" in str(self.test_transforms.__class__):
            self.val_dataset = MNISTAlbumentation(
                self.data_path, train=False, transform=self.test_transforms, download=True
            )
            self.test_dataset = MNISTAlbumentation(
                self.data_path, train=False, transform=self.test_transforms, download=True
            )
        else:
            self.val_dataset = MNIST(self.data_path, train=False, transform=self.test_transforms, download=True)
            self.test_dataset = MNIST(self.data_path, train=False, transform=self.test_transforms, download=True)


class FashionMNISTDataModule(BaseDataModule):
    num_classes = 10

    def __init__(self, **params):
        super(FashionMNISTDataModule, self).__init__(**params)

    def setup(self, stage: str):
        if "albumentations" in str(self.train_transforms.__class__):
            self.train_dataset = FashionMNISTAlbumentation(
                self.data_path, train=True, transform=self.train_transforms, download=True
            )
        else:
            self.train_dataset = FashionMNIST(self.data_path, train=True, transform=self.train_transforms, download=True)

        if "albumentations" in str(self.test_transforms.__class__):
            self.val_dataset = FashionMNISTAlbumentation(
                self.data_path, train=False, transform=self.test_transforms, download=True
            )
            self.test_dataset = FashionMNISTAlbumentation(
                self.data_path, train=False, transform=self.test_transforms, download=True
            )
        else:
            self.val_dataset = FashionMNIST(self.data_path, train=False, transform=self.test_transforms, download=True)
            self.test_dataset = FashionMNIST(self.data_path, train=False, transform=self.test_transforms, download=True)
