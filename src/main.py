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
from models.setup_factory import SetupFactory

import mlflow

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    mlflow.set_tracking_uri('file://' + hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/mlruns')
    mlflow.set_experiment(cfg.exp_name)


    datamodule = DataFactory.get_data(cfg)

    model = SetupFactory.get_model(cfg.setup.mode)()

    loss = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    trainer_class = TrainerFactory.get_trainer(cfg.trainer)
    trainer = trainer_class(cfg, datamodule, model, loss, optimizer)

    match cfg.setup.setep:
        case 'train':
            trainer.run()
        case 'test':
            trainer.validate()
        case 'predict':
            raise NotImplementedError
    trainer.finalize()


if __name__ == '__main__':
    main()
