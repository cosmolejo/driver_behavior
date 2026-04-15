import torch
from torch.utils.data import random_split
from src.data.dmd import DMD
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
       full_dataset = DMD(cfg)
       train_size = int(0.8 * len(full_dataset))
       val_size = int(0.1 * len(full_dataset))
       test_size = len(full_dataset) - train_size - val_size

       # Obliga a usar un generador con semilla fija (por si acaso)
       generador = torch.Generator().manual_seed(306638)
       train_ds, val_ds, test_ds = random_split(
           full_dataset, [train_size, val_size, test_size], generator=generador
       )

       # GUARDA LOS ÍNDICES EN ARCHIVOS
       torch.save(train_ds.indices, 'train_indices.pt')
       torch.save(val_ds.indices, 'val_indices.pt')
       torch.save(test_ds.indices, 'test_indices.pt')

if __name__ == '__main__':
    main()