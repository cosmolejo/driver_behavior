"""

Main
-Capture the config file
-Create an agent instance
-Run the agent
"""






from omegaconf import DictConfig, OmegaConf
import hydra
import os
from pprint import pprint
from data import *
from trainers import *

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    #print(f"Working directory : {os.getcwd()}")
    #print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    # Create the Agent and pass all the configuration to it then run it..

    data_module_class = globals()[cfg.data_module]
    datamodule = data_module_class(cfg)
    trainer_class = globals()[cfg.trainer]
    agent = trainer_class(cfg, datamodule)
    print(agent.config, agent.tr)
    #agent.run()
    #agent.finalize()


if __name__ == '__main__':
    main()
