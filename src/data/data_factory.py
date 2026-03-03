"""


"""
from typing import Callable

from .dna_datamodule  import DnADataModule

class DataFactory:
    data_dict = {
        'drive_and_act': DnADataModule,
    }



    @staticmethod
    def get_data(data_name: str) -> Callable:
        return DataFactory.data_dict[data_name]
