import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset

class DataModule:
    def __init__(self, config, dataset ):
        self.config = config
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.dataset = dataset



    def setup(self):
        full_dataset = self.dataset(self.config)
        label_path = self.config.label_path
        train_idx = torch.load(label_path+'train_indices.pt')
        val_idx = torch.load(label_path+'val_indices.pt')
        test_idx = torch.load(label_path+'test_indices.pt')

        if self.config.data_mode == "train_test_val":
            self.train_ds = Subset(full_dataset, train_idx)
            self.val_ds = Subset(full_dataset, val_idx)
            self.test_ds = Subset(full_dataset, test_idx)
        elif self.config.data_mode == "train_test":
            self.train_ds = Subset(full_dataset, train_idx)
            self.test_ds = Subset(full_dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

    def val_dataloader(self):
        if self.val_ds:
            return DataLoader(
                self.val_ds,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
        else:
            raise Exception("Validation dataset not specified in configuration")

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

