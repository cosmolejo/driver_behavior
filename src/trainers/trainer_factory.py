"""


"""
from typing import Callable

from .trainer import SafeDrivingTrainer

class TrainerFactory:
    models_dict = {
        'safe_driving': SafeDrivingTrainer,
    }


    @staticmethod
    def get_trainer(trainer_name: str) -> Callable:
        return TrainerFactory.models_dict[trainer_name]
