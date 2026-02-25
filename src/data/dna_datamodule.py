import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from .drive_and_act import DriveAndAct



class DnADataModule:
    def __init__(self, config):
        self.config = config
        # Atributos que se llenarán en setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        """
        Paso 1: Operaciones de una sola vez.
        Ej: Descargar videos, descomprimir archivos, procesar CSVs.
        No asigna estado a la clase (self.algo = ...)
        """
        print("Preparando datos en disco...")

    def setup(self):
        """
        Paso 2: Transformaciones, splits y creación de Datasets.
        Se ejecuta en cada GPU individualmente.
        """
        # Aquí crearías tu instancia de la clase VideoDataset que ya tienes
        full_dataset = DriveAndAct(self.config)



        # Podrías cargar un test_ds por separado si tienes otro dataframe
        # self.test_ds = TuCustomVideoDataset(self.test_df)
        if self.config.data_mode == "train_test_val":

            train_size = int(0.8 * len(full_dataset))
            val_size = int(0.2 * len(full_dataset) - train_size)
            test_size = len(full_dataset) - train_size - val_size
            self.train_ds,self.test_ds, self.val_ds = random_split(
                full_dataset, [train_size, test_size, val_size]
            )

        elif self.config.data_mode == "train_test":

            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            self.train_ds, self.test_ds = random_split(
                full_dataset, [train_size, test_size]
            )

        elif self.config.data_mode == "kfolds":
            skf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed)
            index = list(skf.split(X, y))

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True  # Recomendado para entrenamiento en GPU
        )

    def val_dataloader(self):
        if self.val_ds:
            return DataLoader(
                self.val_ds,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers']
            )
        else:
            raise Exception(" Validation dataset not specified in configuration")

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )