"""
Exporta el modelo actual a ExecuTorch (.pte) sin Hydra.

La configuración se carga únicamente con OmegaConf. El checkpoint esperado
es el formato actual del proyecto: un diccionario que contiene
`model_state_dict` (por ejemplo, model_best_segment.pth).

Entrada del modelo actual:
    (B, C, T, H, W)

Con la configuración usada en entrenamiento:
    sequence_length = 32
    sample_one_each = 2
    Resize = 112 x 112

la entrada efectiva es:
    (1, 3, 16, 112, 112)
"""

import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)

from model import get_model


# ---------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------
CONFIG_PATH = "configs/config_binary_phone_balanced.yaml"
CHECKPOINT_PATH = "../models/binary_phone_lad13_augmentation/model_8.pth"
OUTPUT_PATH = "model_binary_phone.pte"

# El Resize actual del pipeline de entrenamiento/evaluación es 112x112.
# Si más adelante lo mueves al YAML, este valor se puede eliminar y leer
# directamente desde la configuración.
DEFAULT_INPUT_SIZE = 112


def infer_num_classes(state_dict: dict) -> int:
    """
    Infiere el número de clases desde la última capa Linear del classifier.
    Evita depender de que `num_classes` del YAML coincida con el checkpoint.
    """
    classifier_keys = sorted(
        key
        for key in state_dict
        if key.startswith("classifier.") and key.endswith(".weight")
    )

    if not classifier_keys:
        raise RuntimeError(
            "No se pudo inferir num_classes: no se encontraron pesos "
            "`classifier.*.weight` en el checkpoint."
        )

    return int(state_dict[classifier_keys[-1]].shape[0])


def infer_lstm_architecture(state_dict: dict) -> tuple[int, int]:
    """
    Infiere hidden_dim y número de capas LSTM desde el state_dict.
    """
    key = "lstm.weight_ih_l0"
    if key not in state_dict:
        raise RuntimeError(
            f"No se encontró {key!r}; el checkpoint no parece corresponder "
            "al modelo MobileNet + BiLSTM actual."
        )

    hidden_dim = int(state_dict[key].shape[0] // 4)

    layer_ids = set()
    prefix = "lstm.weight_ih_l"
    for name in state_dict:
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            suffix = suffix.replace("_reverse", "")
            if suffix.isdigit():
                layer_ids.add(int(suffix))

    if not layer_ids:
        raise RuntimeError("No se pudo inferir el número de capas LSTM.")

    lstm_layers = max(layer_ids) + 1
    return hidden_dim, lstm_layers


def build_model_from_checkpoint(cfg, state_dict: dict):
    """
    Construye el modelo usando la configuración actual, pero valida contra
    la arquitectura realmente almacenada en el checkpoint.
    """
    num_classes = infer_num_classes(state_dict)
    hidden_dim_ckpt, lstm_layers_ckpt = infer_lstm_architecture(state_dict)

    model_kwargs = {}
    if "model_kwargs" in cfg and cfg.model_kwargs is not None:
        model_kwargs = OmegaConf.to_container(
            cfg.model_kwargs,
            resolve=True,
        )

    # La arquitectura del checkpoint manda sobre esos dos parámetros.
    # Dropout/freeze_bn/etc. pueden seguir viniendo del YAML.
    model_kwargs["hidden_dim"] = hidden_dim_ckpt
    model_kwargs["lstm_layers"] = lstm_layers_ckpt

    model = get_model(
        num_classes=num_classes,
        **model_kwargs,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            "El checkpoint no coincide exactamente con el modelo construido.\n"
            f"Parámetros faltantes: {missing}\n"
            f"Parámetros inesperados: {unexpected}"
        )

    return model, num_classes, model_kwargs


def main():
    config_path = Path(CONFIG_PATH)
    checkpoint_path = Path(CHECKPOINT_PATH)
    output_path = Path(OUTPUT_PATH)

    if not config_path.exists():
        raise FileNotFoundError(f"No existe el config: {config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Configuración: OmegaConf solamente, sin Hydra
    # ------------------------------------------------------------------
    cfg = OmegaConf.load(config_path)

    # ------------------------------------------------------------------
    # Checkpoint actual del proyecto
    #
    # Los checkpoints recientes pueden contener optimizer, scheduler,
    # RNG y metadata Python además del state_dict. Como es un checkpoint
    # propio/de confianza, se carga explícitamente con weights_only=False.
    # ------------------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if not isinstance(state_dict, dict):
        raise TypeError(
            "El checkpoint no contiene un state_dict reconocible."
        )

    model, num_classes, model_kwargs = build_model_from_checkpoint(
        cfg,
        state_dict,
    )
    model.eval()

    # ------------------------------------------------------------------
    # Forma correcta de la entrada
    #
    # Dataset:
    #   frames_per_window = ceil(sequence_length / sample_one_each)
    #
    # Config actual:
    #   32 / 2 -> 16 frames efectivos
    #
    # Modelo actual:
    #   (B, C, T, H, W), NO (B, T, C, H, W)
    # ------------------------------------------------------------------
    sequence_length = int(cfg.sequence_length)
    sample_one_each = int(cfg.sample_one_each)

    frames_per_window = math.ceil(
        sequence_length / max(1, sample_one_each)
    )

    input_size = int(cfg.get("input_size", DEFAULT_INPUT_SIZE))

    example_inputs = (
        torch.rand(
            1,                  # B
            3,                  # C = RGB
            frames_per_window,  # T
            input_size,         # H
            input_size,         # W
            dtype=torch.float32,
        ),
    )

    print("Configuración de exportación")
    print("--------------------------")
    print(f"Config       : {config_path}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Salida       : {output_path}")
    print(f"num_classes  : {num_classes}")
    print(f"model_kwargs : {model_kwargs}")
    print(f"Input shape  : {tuple(example_inputs[0].shape)}")
    print()

    # Sanity check antes de exportar.
    with torch.no_grad():
        output = model(*example_inputs)

    print(f"Forward OK   : output shape = {tuple(output.shape)}")

    # ------------------------------------------------------------------
    # PyTorch Export -> ExecuTorch/XNNPACK
    # ------------------------------------------------------------------
    exported_program = torch.export.export(
        model,
        example_inputs,
    )

    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        file.write(program.buffer)

    print(f"Modelo ExecuTorch guardado en: {output_path}")


if __name__ == "__main__":
    main()
