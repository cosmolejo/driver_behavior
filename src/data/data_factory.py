"""


"""
from typing import Callable

from omegaconf import DictConfig

from .datamodule  import DataModule
from .drive_and_act import DriveAndAct
from .dmd import DMD
class DataFactory:
    data_dict = {
        'drive_and_act': DriveAndAct,
        'dmd': DMD,
    }



    @staticmethod
    def get_data(cfg: DictConfig) -> DataModule:
        datamodule = DataModule(cfg, DataFactory.data_dict[cfg.data_module])
        return datamodule
