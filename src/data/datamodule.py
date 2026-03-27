import torch
from torch.utils.data import DataLoader, random_split
from .drive_and_act import DriveAndAct

class DataModule:
    def __init__(self, config, dataset ):
        self.config = config
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.dataset = dataset



    def setup(self):
        full_dataset = self.dataset(self.config)

        if self.config.data_mode == "train_test_val":
            train_size = int(0.8 * len(full_dataset))
            val_size = int(0.1 * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            self.train_ds, self.test_ds, self.val_ds = random_split(
                full_dataset, [train_size, test_size, val_size]
            )
        elif self.config.data_mode == "train_test":
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            self.train_ds, self.test_ds = random_split(
                full_dataset, [train_size, test_size]
            )

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

