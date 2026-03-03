"""

Main
-Capture the config file
-Create an agent instance
-Run the agent
"""






import hydra
import torch.optim as optim
from omegaconf import DictConfig
from torch import nn
from data.data_factory import DataFactory
from trainers.trainer_factory import TrainerFactory
from models.nn.nn_factory import ModelFactory



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):


    # Create the Agent and pass all the configuration to it then run it.

    data_module_class = DataFactory.get_data(cfg.data_module)
    datamodule = data_module_class(cfg)

    backbone_class = ModelFactory.get_model(cfg.backbone_model)
    backbone = backbone_class()

    temporal_class = ModelFactory.get_model(cfg.temporal_model.model)
    input_size = backbone.output_channels if cfg.temporal_model.input_size == 'default' else cfg.temporal_model.input_size
    hidden_size = backbone.output_channels if cfg.temporal_model.hidden_size == 'default' else cfg.temporal_model.hidden_size
    temporal = temporal_class(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = cfg.temporal_model.num_layers,
        batch_first = cfg.temporal_model.batch_first,
        bidirectional = cfg.temporal_model.bidirectional
    )

    model = ModelFactory.get_model(cfg.model.name)(
        backbone, temporal, input_size, cfg.model.num_classes)


    loss = nn.NLLLoss()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)

    trainer_class = TrainerFactory.get_trainer(cfg.trainer)
    trainer = trainer_class(cfg, datamodule, model, loss, optimizer)

    trainer.run()
    trainer.finalize()


if __name__ == '__main__':
    main()
