"""

Main
-Capture the config file
-Create an agent instance
-Run the agent
"""






from pathlib import Path
from utils.bot_telegram import notify
import hydra
import torch.optim as optim
from omegaconf import DictConfig
from torch import nn
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from data.data_factory import DataFactory
from trainers.trainer_factory import TrainerFactory
from models.setup_factory import SetupFactory
import mlflow

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Store de MLflow:
    #  - en --multirun (barrido): UNO compartido en la raiz del sweep, de modo que
    #    todos los trials queden como runs comparables del mismo experimento.
    #  - en ejecucion normal: dentro de la carpeta del run.
    # .as_uri() construye un file:// valido en Windows y Linux por igual.
    hc = HydraConfig.get()
    if hc.mode == RunMode.MULTIRUN:
        mlruns_dir = Path(hc.runtime.cwd) / hc.sweep.dir / "mlruns"
    else:
        mlruns_dir = Path(hc.runtime.output_dir) / "mlruns"
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())

    mlflow.set_experiment(cfg.exp_name)


    datamodule = DataFactory.get_data(cfg)


    model = SetupFactory.get_model(cfg.setup.mode)(cfg)

    loss = nn.CrossEntropyLoss()

    # define optimizer
    optim_param = cfg.setup.optimizer
    optimizer = optim.Adam(model.parameters(), lr=optim_param.learning_rate, betas= (optim_param.beta1, optim_param.beta2), weight_decay=optim_param.weight_decay)

    trainer_class = TrainerFactory.get_trainer(cfg.setup.trainer)
    trainer = trainer_class(cfg, datamodule, model, loss, optimizer)

    objective = None
    match cfg.setup.step:
        case 'train':
            objective = trainer.run()   # best_loss (test-loss minimo) -> objetivo de Optuna
        case 'test':
            trainer.test()
        case 'predict':
            raise NotImplementedError
    trainer.finalize()

    job_num = HydraConfig.get().job.num  # indice del trial en el barrido (0..n-1)
    notify(f"[{cfg.exp_name}] run {job_num} OK | best_test_loss={objective}")

    # Optuna minimiza el valor retornado por main(); solo se usa en --multirun.
    return objective


if __name__ == '__main__':
    main()