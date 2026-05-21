"""
DataModule para la estructura de particiones generada por split_dataset.py.

Cambios respecto a la version anterior
--------------------------------------
  * Ya NO se usa random_split ni los indices train/val/test_indices.pt:
    las particiones estan separadas FISICAMENTE en carpetas
    <data_path>/{TRAIN,VALIDATION,TEST}/...
  * Se crea un Dataset por split, pasando el nombre del split como argumento.
    El split se determina por la clave de configuracion data_mode:
        - "train_test_val": crea TRAIN, VALIDATION y TEST
        - "train_test"    : crea TRAIN y TEST (sin VALIDATION)
"""
from torch.utils.data import DataLoader


class DataModule:
    def __init__(self, config, dataset):
        """
        Args:
            config : configuracion del experimento.
            dataset: la CLASE del Dataset (p.ej. DMD). Se instancia una vez por
                     split con la firma dataset(config, split=<SPLIT>).
        """
        self.config = config
        self.dataset = dataset
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self):
        mode = self.config.data_mode

        if mode == "train_test_val":
            self.train_ds = self.dataset(self.config, split="TRAIN")
            self.val_ds = self.dataset(self.config, split="VALIDATION")
            self.test_ds = self.dataset(self.config, split="TEST")
        elif mode == "train_test":
            self.train_ds = self.dataset(self.config, split="TRAIN")
            self.test_ds = self.dataset(self.config, split="TEST")
        else:
            raise ValueError(
                "data_mode no soportado: {!r} (usa 'train_test_val' o 'train_test')".format(mode)
            )

    def train_dataloader(self):
        if self.train_ds is None:
            raise RuntimeError("Llama a setup() antes de pedir el train_dataloader().")
        return DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            raise Exception("Validation dataset not specified in configuration")
        return DataLoader(
            self.val_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            raise RuntimeError("Llama a setup() antes de pedir el test_dataloader().")
        return DataLoader(
            self.test_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )