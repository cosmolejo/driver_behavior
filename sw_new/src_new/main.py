from pathlib import Path
import sys
from omegaconf import OmegaConf
from trainer import train_pipeline

if __name__ == "__main__":
    conf = OmegaConf.load("config.yaml")
    cli_config = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_config)
    if conf.optuna.study_name is not None:
        del conf.optuna
    train_pipeline(
        **conf
        # data_dir=Path(__file__).parent.parent.resolve() / "dmd",   # Your data folder
        # num_classes=3,
        # sequence_length=32,  # Or -1 for full video
        # sample_one_each=2,
        # batch_size=2,
        # num_epochs=500,
        # loss_fn="CrossEntropyLoss",
        # lr=1e-4
    )