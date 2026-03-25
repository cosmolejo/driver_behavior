import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split, StratifiedKFold

from .drive_and_act_padding import DriveAndAct



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
        full_dataset = DriveAndAct(self.config)

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
            raise Exception("Not implemented yet")
            #skf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed)
            #index = list(skf.split(X, y))

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,  # Recomendado para entrenamiento en GPU
            collate_fn=self.custom_collate_fn
        )

    def val_dataloader(self):
        if self.val_ds:
            return DataLoader(
                self.val_ds,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                collate_fn=self.custom_collate_fn
            )
        else:
            raise Exception(" Validation dataset not specified in configuration")

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=self.custom_collate_fn
        )

    @staticmethod
    def custom_collate_fn(batch):
        # 1. Separamos los datos del batch
        videos = [v for v, _ in batch]
        labels = [label for _, label in batch]


        lengths = [v.shape[0] for v in videos]
        lengths_tensor = torch.tensor(lengths)

        # 2. Rellenamos los videos con ceros (Padding)
        videos_padded = pad_sequence(videos, batch_first=True)

        # 3. Convertimos las etiquetas a un tensor matemático
        labels_tensor = torch.tensor(labels)

        # Retornamos la tupla que espera el bucle de entrenamiento
        return videos_padded, labels_tensor, lengths_tensor

