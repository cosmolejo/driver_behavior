"""


"""
from typing import Callable

from ..nn.mobileNet import mobilenet_v3_small_local, mobilenet_v3_large_local
from torch.nn import LSTM
from .classifier_model import ClassifierModel
from .static_classifier_model import StaticClassifierModel
class ModelFactory:
    models_dict = {
        'mobilenet_v3_small': mobilenet_v3_small_local,
        'mobilenet_v3_large': mobilenet_v3_large_local,
        'LSTM': LSTM,
        'classifier': ClassifierModel,
        'static_classifier': StaticClassifierModel,

    }

    @staticmethod
    def get_model(model_name: str) -> Callable:
        return ModelFactory.models_dict[model_name]
