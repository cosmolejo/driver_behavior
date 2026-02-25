"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""




from trainers import *

from omegaconf import DictConfig, OmegaConf
import hydra
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    # Create the Agent and pass all the configuration to it then run it..

    
    trainer_class = globals()[cfg.trainer]
    agent = trainer_class(cfg)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
