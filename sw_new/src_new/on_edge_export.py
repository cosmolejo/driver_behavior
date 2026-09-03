"""
Exporta el checkpoint actual a ExecuTorch (.pte) sin Hydra.

CRITICO:
La reconstruccion del modelo es EXACTAMENTE la misma usada por
eval_model_fixed.py y por el nuevo base_predictor.py:

    state_dict
        -> infer hidden_dim, lstm_layers, num_classes
        -> get_model(num_classes, hidden_dim=..., lstm_layers=...)
        -> load_state_dict(...)
        -> model.eval()
        -> torch.export
        -> ExecuTorch/XNNPACK

El YAML NO redefine model_kwargs. Solo determina la forma temporal/espacial
del tensor de ejemplo para exportacion.
"""

import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from executorch.exir import (
    to_edge_transform_and_lower,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)

from model import get_model


# ---------------------------------------------------------------------
# Rutas: editar segun la corrida que se quiera exportar
# ---------------------------------------------------------------------
CONFIG_PATH = "configs/config_binary_phone_balanced.yaml"
CHECKPOINT_PATH = (
    "../models/binary_phone_lad13_augmentation/"
    "model_best_segment.pth"
)
OUTPUT_PATH = "../model_binary_phone.pte"

DEFAULT_INPUT_SIZE = 112


def infer_model_kwargs(state_dict: dict) -> dict:
    """
    Misma inferencia arquitectonica de eval_model_fixed.py.
    """
    lstm_key = "lstm.weight_ih_l0"
    if lstm_key not in state_dict:
        raise RuntimeError(
            f"No se encontro {lstm_key!r}. "
            "Checkpoint incompatible con el modelo actual."
        )

    hidden_dim = int(
        state_dict[lstm_key].shape[0] // 4
    )

    layer_ids = set()
    for key in state_dict:
        if key.startswith("lstm.weight_ih_l"):
            suffix = key[len("lstm.weight_ih_l"):]
            suffix = suffix.replace("_reverse", "")
            if suffix.isdigit():
                layer_ids.add(int(suffix))

    if not layer_ids:
        raise RuntimeError(
            "No se pudo inferir lstm_layers."
        )

    lstm_layers = max(layer_ids) + 1

    classifier_keys = sorted(
        key
        for key in state_dict
        if key.startswith("classifier.")
        and key.endswith(".weight")
    )

    if not classifier_keys:
        raise RuntimeError(
            "No se pudo inferir num_classes "
            "desde classifier.*.weight."
        )

    num_classes = int(
        state_dict[
            classifier_keys[-1]
        ].shape[0]
    )

    return {
        "hidden_dim": hidden_dim,
        "lstm_layers": lstm_layers,
        "num_classes": num_classes,
    }


def build_model_from_state_dict(
    state_dict: dict,
):
    """
    Copia deliberadamente el patron de eval_model_fixed.py.
    """
    inferred = infer_model_kwargs(state_dict)
    num_classes = inferred.pop("num_classes")

    model = get_model(
        num_classes,
        **inferred,
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "El checkpoint no coincide con el "
            "modelo reconstruido.\n"
            f"Parametros faltantes: {missing}\n"
            f"Parametros inesperados: {unexpected}"
        )

    model.eval()

    return model, num_classes, inferred


def main():
    config_path = Path(CONFIG_PATH)
    checkpoint_path = Path(CHECKPOINT_PATH)
    output_path = Path(OUTPUT_PATH)

    if not config_path.exists():
        raise FileNotFoundError(
            f"No existe el config: {config_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint: "
            f"{checkpoint_path}"
        )

    cfg = OmegaConf.load(config_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    if not isinstance(state_dict, dict):
        raise TypeError(
            "El checkpoint no contiene un "
            "state_dict reconocible."
        )

    # Seguridad adicional para este checkpoint:
    # el usuario verifico que NO contiene Transformer.
    transformer_keys = [
        key
        for key in state_dict
        if key.startswith("transformer.")
    ]
    if transformer_keys:
        raise RuntimeError(
            "Este exportador esta configurado para la "
            "arquitectura sin Transformer, pero el "
            "checkpoint contiene parametros transformer.*"
        )

    model, num_classes, inferred = (
        build_model_from_state_dict(state_dict)
    )

    sequence_length = int(
        cfg.sequence_length
    )
    sample_one_each = max(
        1,
        int(cfg.sample_one_each),
    )
    frames_per_window = math.ceil(
        sequence_length / sample_one_each
    )

    input_size = int(
        cfg.get(
            "input_size",
            DEFAULT_INPUT_SIZE,
        )
    )

    example_tensor = torch.rand(
        1,
        3,
        frames_per_window,
        input_size,
        input_size,
        dtype=torch.float32,
    )

    example_inputs = (example_tensor,)

    print("Configuracion de exportacion")
    print("---------------------------")
    print(f"Config          : {config_path}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Salida          : {output_path}")
    print(
        "Reconstruccion  : "
        "igual a eval_model_fixed.py"
    )
    print(f"num_classes     : {num_classes}")
    print(f"Arquitectura    : {inferred}")
    print(
        f"Input shape     : "
        f"{tuple(example_tensor.shape)}"
    )

    if "epoch" in checkpoint:
        print(
            f"Epoca checkpoint: "
            f"{checkpoint['epoch']}"
        )

    # Forward PyTorch antes de exportar.
    with torch.no_grad():
        pytorch_output = model(
            example_tensor
        )

    print(
        f"Forward PyTorch : "
        f"{tuple(pytorch_output.shape)}"
    )
    print(
        "Logits ejemplo  : "
        f"{pytorch_output[0].tolist()}"
    )

    # Export PyTorch -> ExecuTorch/XNNPACK.
    exported_program = torch.export.export(
        model,
        example_inputs,
    )

    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[
            XnnpackPartitioner()
        ],
    ).to_executorch()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("wb") as file:
        file.write(program.buffer)

    print()
    print(
        "Modelo ExecuTorch guardado en: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
