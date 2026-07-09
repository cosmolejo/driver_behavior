"""
Batería de experimentos con Optuna.

Uso:
    python optuna_search.py --n-trials 50
    python optuna_search.py --n-trials 50 --timeout 36000   # 10 hs máx

Requiere: pip install optuna
Persiste en SQLite (config.yaml -> optuna.storage), así que se puede cortar
y retomar el estudio en cualquier momento sin perder los trials ya corridos:
    optuna study load driver_behavior_search --storage sqlite:///optuna_study.db
"""
import argparse

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from omegaconf import OmegaConf

from trainer import train_pipeline

BASE_CONF = OmegaConf.load("config.yaml")
OPT_CONF = BASE_CONF.optuna


def objective(trial: optuna.Trial) -> float:
    # --- Optimización / scheduler ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    pct_start = trial.suggest_float("pct_start", 0.05, 0.3)
    div_factor = trial.suggest_categorical("div_factor", [10, 25, 50])

    # --- Batch size (limitado por memoria de GPU con video) ---
    batch_size = trial.suggest_categorical("batch_size", [2, 4])

    # --- Arquitectura ---
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])

    # --- Loss function + kwargs condicionales ---
    loss_fn = trial.suggest_categorical(
        "loss_fn", ["cross_entropy", "CE_label_smoothing", "CE_weight", "sigmoid_focal_loss"]
    )
    loss_kwargs = {}
    if loss_fn == "CE_label_smoothing":
        loss_kwargs["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.3)
    elif loss_fn == "CE_weight":
        loss_kwargs["class_weight_power"] = trial.suggest_float("class_weight_power", 0.5, 1.5)
    elif loss_fn == "sigmoid_focal_loss":
        loss_kwargs["alpha"] = trial.suggest_float("alpha", 0.1, 0.9)
        loss_kwargs["gamma"] = trial.suggest_float("gamma", 1.0, 3.0)

    val_loss = train_pipeline(
        data_dir=BASE_CONF.data_dir,
        num_classes=BASE_CONF.num_classes,
        sequence_length=BASE_CONF.sequence_length,
        sample_one_each=BASE_CONF.sample_one_each,
        batch_size=batch_size,
        num_epochs=OPT_CONF.search_epochs,
        loss_fn=loss_fn,
        loss_kwargs=loss_kwargs,
        lr=lr,
        weight_decay=weight_decay,
        pct_start=pct_start,
        div_factor=div_factor,
        model_kwargs=dict(
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        ),
        trial=trial,
        run_name=f"optuna/trial_{trial.number}",
        save_checkpoints=False,  # evita saturar disco durante la búsqueda
    )
    return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=OPT_CONF.n_trials)
    parser.add_argument("--timeout", type=int, default=None, help="segundos, opcional")
    args = parser.parse_args()

    pruner = HyperbandPruner(
        min_resource=OPT_CONF.pruner_min_resource,
        max_resource=OPT_CONF.search_epochs,
        reduction_factor=OPT_CONF.pruner_reduction_factor,
    )
    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        study_name=OPT_CONF.study_name,
        storage=OPT_CONF.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,  # permite retomar el estudio si se cortó
    )

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nTrials completados: {len(completed)} | podados: {len(pruned)}")
    print(f"Mejor val_loss: {study.best_value:.4f}")
    print("Mejores hiperparámetros:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
