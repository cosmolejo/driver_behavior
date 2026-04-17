"""


"""
from typing import Callable

from .trainer import SafeDrivingTrainer
from .trainer_multicam import MultiCamTrainer


class TrainerFactory:
    models_dict = {
        'safe_driving': SafeDrivingTrainer,
        'safe_driving_multicam': MultiCamTrainer,
    }


    @staticmethod
    def get_trainer(trainer_name: str) -> Callable:
        return TrainerFactory.models_dict[trainer_name]
