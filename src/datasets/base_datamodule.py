import random
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from hydra.utils import get_original_cwd

class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_root_dir,
        name,
        batch_size,
        train_transform=None, # XXX Only used in subclasses
        test_transform=None,  # XXX Only used in subclasses
        num_workers=4,      # TODO should this not be taken care of by pytorch lightning?
        shuffle=True,     # Shuffle train dataset
        persistent_workers=True,
        # prepare_data_per_node,
        *args,
        **kwargs
    ):
        super(BaseDataModule, self).__init__()

        self.data_path = Path(get_original_cwd()) / Path(data_root_dir) / name
        self.batch_size = batch_size
        if train_transform is not None:
            self.train_transforms = train_transform()
        else:
            self.train_transforms = None
        if test_transform is not None:
            self.test_transforms = test_transform()
        else:
            self.test_transforms = None
        # self.random_batches = random_batches
        self.num_workers = num_workers
        # self.prepare_data_per_node = prepare_data_per_node
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        trainloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=self.persistent_workers,
        )

        return trainloader

    def val_dataloader(self):
        valloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=self.persistent_workers,
        )

        return valloader

    def test_dataloader(self):
        testloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=self.persistent_workers,
        )

        return testloader
    
    def predict_dataloader(self):
        predictloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        return predictloader


def seed_worker(worker_id):
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    ensures different random numbers each batch with each worker every epoch while keeping reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
