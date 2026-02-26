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

from models.nn.nn_factory import ModelFactory


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    #print(f"Working directory : {os.getcwd()}")
    #print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    # Create the Agent and pass all the configuration to it then run it..

    data_module_class = globals()[cfg.data_module]
    datamodule = data_module_class(cfg)
    trainer_class = globals()[cfg.trainer]
    models_class = ModelFactory.get_model(cfg.model)
    model = models_class()

    loss = nn.NLLLoss()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)


    agent = trainer_class(cfg, datamodule, model, loss, optimizer)
    print(agent.train_loader)
    #agent.run()
    #agent.finalize()


if __name__ == '__main__':
    main()
