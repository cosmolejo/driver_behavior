"""


"""
from typing import Callable
from omegaconf import DictConfig
from nn.nn_factory import ModelFactory

class SetupFactory:

    @staticmethod
    def _base_model(cfg: DictConfig) -> Callable:
        backbone_class = ModelFactory.get_model(cfg.backbone_model)
        backbone = backbone_class()

        temporal_class = ModelFactory.get_model(cfg.temporal_model.model)
        input_size = backbone.output_channels if cfg.temporal_model.input_size == 'default' else cfg.temporal_model.input_size
        hidden_size = backbone.output_channels if cfg.temporal_model.hidden_size == 'default' else cfg.temporal_model.hidden_size
        temporal = temporal_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=cfg.temporal_model.num_layers,
            batch_first=cfg.temporal_model.batch_first,
            bidirectional=cfg.temporal_model.bidirectional
        )

        model = ModelFactory.get_model(cfg.model.name)(
            backbone, temporal, input_size, cfg.model.num_classes)

        return model

    @staticmethod
    def _multi_model(cfg: DictConfig) -> Callable:

        face_model = SetupFactory._base_model(cfg)
        body_model = SetupFactory._base_model(cfg)

        model = ModelFactory.get_model(cfg.model.name)(
            face_model, body_model, cfg.model.input_size, cfg.model.num_classes)




    mode_dict = {
        'face': _base_model,
    }