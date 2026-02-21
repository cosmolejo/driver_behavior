"""
Drive and act Dataloader

"""


import imageio
import torch
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import csv

from torch.utils.data import DataLoader, TensorDataset, Dataset

class DnADataloader(Dataset):
    def __init__(self, config):
        self.config = config

        if config.data_mode == "train_test_val":
            self.dataset = datasets.ImageFolder(config.data_path)
            data = csv.reader(open(config.data_path + "/train_test_val.csv"))
        elif config.data_mode == "train_test":
            self.dataset = datasets.ImageFolder(config.data_path)

        elif config.data_mode == "kfolds":
            self.dataset = datasets.ImageFolder(config.data_path)

        elif config.data_mode == "random":
            train_data = torch.randn(self.config.batch_size, self.config.input_channels, self.config.img_size,
                                     self.config.img_size)
            train_labels = torch.ones(self.config.batch_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.test_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")