"""


"""
from ..nn.mobileNet import mobilenet_v3_small_local, mobilenet_v3_large_local
class ModelFactory:
    models_dict = {
        'mobilenet_v3_small': mobilenet_v3_small_local,
        'mobilenet_v3_large': mobilenet_v3_large_local,

    }

    @staticmethod
    def get_model(model_name):
        return ModelFactory.models_dict[model_name]
