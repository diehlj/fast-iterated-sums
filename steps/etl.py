from dataclasses import dataclass

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, default_collate, random_split
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import v2

from shared.defintions import (
    DataName,
    MVTECCategory,
)


def create_transform(mean: tuple, std: tuple, train: bool) -> transforms.Compose:
    fns = []

    if train:
        fns.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    fns.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transforms.Compose(fns)


def collate_fn(batch: torch.Tensor, num_classes: int) -> torch.Tensor:
    cutmix = v2.CutMix(num_classes=num_classes)
    return cutmix(*default_collate(batch))


def load_data(data_name: str, batch_size: int = 256) -> tuple[DataLoader, DataLoader]:
    DATA_DIR = "./data"

    # mean, std obtained from https://github.com/chenyaofo/pytorch-cifar-models
    if data_name == DataName.CIFAR10:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        dataset_builder = CIFAR10

    elif data_name == DataName.CIFAR100:
        mean = (0.507, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2761)
        dataset_builder = CIFAR100

    else:
        raise ValueError(f"data_name must be one of {', '.join(DataName)}")

    train_dataset = dataset_builder(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=create_transform(mean=mean, std=std, train=True),
    )
    validation_dataset = dataset_builder(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=create_transform(mean=mean, std=std, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    print(f"Number of train data examples: {len(train_dataset)}")
    print(f"Number of validation data examples: {len(validation_dataset)}")

    return train_loader, validation_loader


@dataclass(frozen=True)
class MVTECDataSplit:
    train_loader: DataLoader
    validation_loader: DataLoader | None
    transform: transforms.Compose
    img_size: tuple[int, int]


def load_mvtec(
    batch_size: int, category: str, include_val: bool = True
) -> MVTECDataSplit:
    if category not in MVTECCategory:
        raise ValueError(
            f"invalid category: category must be one of {', '.join(MVTECCategory)}"
        )

    ROOT = f"./data/mvtec/{category}"
    MVTEC_IMG_SIZE = (256, 256)
    MVTEC_IMG_CROP_SIZE = (224, 224)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = transforms.Compose(
        [
            transforms.Resize(MVTEC_IMG_SIZE),
            transforms.CenterCrop(MVTEC_IMG_CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # NOTE: ImageFolder converts all images to RGB, so we don't need to worry about 1-channel -> 3-channel conversion
    train_set = ImageFolder(root=f"{ROOT}/train", transform=transform)

    if include_val:
        # FIXME: mvtec dataset paper recommend 10% validation set
        VAL_SIZE = 0.1
        rng = torch.Generator().manual_seed(42)
        train_set, validation_set = random_split(
            train_set, [1.0 - VAL_SIZE, VAL_SIZE], rng
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(
            validation_set, batch_size=batch_size, shuffle=True
        )

    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        validation_loader = None

    return MVTECDataSplit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        transform=transform,
        img_size=MVTEC_IMG_CROP_SIZE,
    )
