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

import mlflow

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    mlflow.set_tracking_uri('file://' + hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/mlruns')
    mlflow.set_experiment(cfg.exp_name)

    # Create the Agent and pass all the configuration to it then run it.


    datamodule = DataFactory.get_data(cfg)




    loss = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    trainer_class = TrainerFactory.get_trainer(cfg.trainer)
    trainer = trainer_class(cfg, datamodule, model, loss, optimizer)

    trainer.run()
    trainer.finalize()


if __name__ == '__main__':
    main()
