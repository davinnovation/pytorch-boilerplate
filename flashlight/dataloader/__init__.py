import numpy as np

import albumentations as AB
from albumentations.pytorch import ToTensor, ToTensorV2
import torchvision
import torch
from torch.utils.data import random_split
import pytorch_lightning as pl

from ..utils import func

from . import coi_dataset

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str, train_batchsize=32, val_batchsize=32, test_batchsize=32, workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform = torchvision.transforms.Compose(
            [torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.ToTensor()]
        )
        self.train_batchsize = train_batchsize
        self.val_batchsize = val_batchsize
        self.test_batchsize = test_batchsize
        self.workers = workers

    def prepare_data(self):
        # download
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = torchvision.datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=self.train_batchsize)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=self.val_batchsize)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.test_batchsize)

""" Dataset """
# Add {Dataset Name : torch.utils.data.Dataset}
DATA_DICT = {
    "MNIST": MNISTDataModule
}

def check_data_option(data, data_option_dict):
    print(data, data_option_dict)
    if len(data_option_dict.keys()) > 0:
        func.function_arg_checker(DATA_DICT[data].__init__, data_option_dict)
    return data_option_dict

def get_data(data, data_option_dict):
    return DATA_DICT[data](**data_option_dict)