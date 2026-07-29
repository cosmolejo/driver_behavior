from omegaconf import OmegaConf
from trainer import train_pipeline

if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()

    # Permite elegir el archivo de config sin editar este script:
    #   python main.py config_path=config_sanity.yaml
    # `pop` lo saca del override para que no se reenvie a train_pipeline
    # como kwarg inexistente.
    config_path = cli_config.pop("config_path", "config.yaml")

    conf = OmegaConf.load(config_path)
    conf = OmegaConf.merge(conf, cli_config)

    if conf.optuna.study_name is not None:
        del conf.optuna

    print(f"Config: {config_path}")
    print(OmegaConf.to_yaml(conf))

    train_pipeline(**conf)