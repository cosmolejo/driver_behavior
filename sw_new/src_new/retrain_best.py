"""
Reentrena el mejor trial del estudio con el número completo de épocas
(sin pruning, guardando checkpoints). Correr después de optuna_search.py.

Uso:
    python retrain_best.py
"""
import optuna
from omegaconf import OmegaConf

from trainer import train_pipeline

BASE_CONF = OmegaConf.load("config.yaml")
OPT_CONF = BASE_CONF.optuna

study = optuna.load_study(study_name=OPT_CONF.study_name, storage=OPT_CONF.storage)
best = study.best_params
print(f"Mejor trial: #{study.best_trial.number} | val_loss={study.best_value:.4f}")
print("Hiperparámetros:", best)

loss_kwargs = {}
if best["loss_fn"] == "CE_label_smoothing":
    loss_kwargs["label_smoothing"] = best["label_smoothing"]
elif best["loss_fn"] == "CE_weight":
    loss_kwargs["class_weight_power"] = best["class_weight_power"]
elif best["loss_fn"] == "sigmoid_focal_loss":
    loss_kwargs["alpha"] = best["alpha"]
    loss_kwargs["gamma"] = best["gamma"]

val_loss = train_pipeline(
    data_dir=BASE_CONF.data_dir,
    num_classes=BASE_CONF.num_classes,
    sequence_length=BASE_CONF.sequence_length,
    sample_one_each=BASE_CONF.sample_one_each,
    batch_size=best["batch_size"],
    num_epochs=OPT_CONF.final_epochs,
    loss_fn=best["loss_fn"],
    loss_kwargs=loss_kwargs,
    lr=best["lr"],
    weight_decay=best["weight_decay"],
    pct_start=best["pct_start"],
    div_factor=best["div_factor"],
    model_kwargs=dict(
        hidden_dim=best["hidden_dim"],
        lstm_layers=best["lstm_layers"],
        dropout=best["dropout"],
        freeze_backbone=best["freeze_backbone"],
    ),
    run_name="best_full_retrain",
    save_checkpoints=True,
)

print(f"\nEntrenamiento final completo. Mejor val_loss: {val_loss:.4f}")
print("Checkpoints en models/best_full_retrain/, logs en runs/best_full_retrain/")
