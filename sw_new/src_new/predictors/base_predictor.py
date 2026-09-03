"""
Predictor para MobileNetV3-Large + BiLSTM + Temporal Attention.

Esta version reconstruye el modelo EXACTAMENTE con la misma logica usada
por eval_model_fixed.py:

    state_dict
        -> infer hidden_dim, lstm_layers, num_classes
        -> get_model(num_classes, hidden_dim=..., lstm_layers=...)
        -> load_state_dict(...)
        -> model.eval()

El YAML NO se usa para reconstruir la arquitectura del modelo. Solo se usa
para parametros de entrada/preprocesamiento y nombres de clase.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision import transforms

from model import get_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_INPUT_SIZE = 112


def infer_model_kwargs(state_dict: dict) -> dict:
    """
    Misma inferencia arquitectonica usada por eval_model_fixed.py.

    Solo se infieren los parametros que quedan codificados directamente
    en el state_dict:
      - hidden_dim
      - lstm_layers
      - num_classes

    El checkpoint confirmado no contiene Transformer; get_model() usa
    use_transformer=False por defecto.
    """
    lstm_key = "lstm.weight_ih_l0"
    if lstm_key not in state_dict:
        raise RuntimeError(
            f"No se encontro {lstm_key!r}. "
            "El checkpoint no parece corresponder al modelo actual."
        )

    hidden_dim = int(state_dict[lstm_key].shape[0] // 4)

    layer_ids = set()
    for key in state_dict:
        if key.startswith("lstm.weight_ih_l"):
            suffix = key[len("lstm.weight_ih_l"):]
            suffix = suffix.replace("_reverse", "")
            if suffix.isdigit():
                layer_ids.add(int(suffix))

    if not layer_ids:
        raise RuntimeError("No se pudo inferir lstm_layers.")

    lstm_layers = max(layer_ids) + 1

    classifier_keys = sorted(
        key
        for key in state_dict
        if key.startswith("classifier.") and key.endswith(".weight")
    )
    if not classifier_keys:
        raise RuntimeError(
            "No se pudo inferir num_classes desde classifier.*.weight."
        )

    num_classes = int(state_dict[classifier_keys[-1]].shape[0])

    return {
        "hidden_dim": hidden_dim,
        "lstm_layers": lstm_layers,
        "num_classes": num_classes,
    }


def build_model_from_state_dict(
    state_dict: dict,
    device: torch.device | str = "cpu",
):
    """
    Construye el modelo exactamente como eval_model_fixed.py.

    Importante:
    - NO usa cfg.model_kwargs.
    - NO fuerza parametros arquitectonicos desde el YAML.
    - NO intenta inferir Transformer.
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
            "El checkpoint no coincide con el modelo reconstruido.\n"
            f"Parametros faltantes: {missing}\n"
            f"Parametros inesperados: {unexpected}"
        )

    model = model.to(device)
    model.eval()

    return model, num_classes, inferred


def resolve_class_names(cfg: DictConfig, num_classes: int) -> list[str]:
    label_mode = str(cfg.get("label_mode", "")).lower()

    known = {
        "binary_phone": ["safe", "phone"],
        "macro": ["reaching", "safe", "unsafe"],
        "ternary_clean": ["reaching", "safe", "unsafe"],
    }

    if label_mode in known and len(known[label_mode]) == num_classes:
        return known[label_mode]

    if label_mode == "fine":
        try:
            from fine_labels import get_scheme

            _, names = get_scheme(
                label_mode,
                cfg.get("partition_report", None),
            )
            names = list(names)
            if len(names) == num_classes:
                return names
        except Exception as exc:
            print(
                "AVISO: no fue posible resolver nombres de clases finas: "
                f"{exc}"
            )

    return [f"class_{i}" for i in range(num_classes)]


class ClipPreprocessor:
    """
    Preprocesamiento compartido por PyTorch y ExecuTorch.

    Entrada:
        (sequence_length, H, W, 3), OpenCV/BGR.

    Salida:
        (1, C, T, H, W), float32 normalizado.
    """

    def __init__(self, config: DictConfig):
        self.cfg = config
        self.sequence_length = int(self.cfg.sequence_length)
        self.sample_one_each = max(
            1,
            int(self.cfg.sample_one_each),
        )
        self.frames_per_window = max(
            1,
            math.ceil(
                self.sequence_length / self.sample_one_each
            ),
        )
        self.input_size = int(
            self.cfg.get("input_size", DEFAULT_INPUT_SIZE)
        )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (self.input_size, self.input_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ])

    def select_temporal_frames(
        self,
        raw_frames: np.ndarray,
    ) -> np.ndarray:
        if len(raw_frames) < self.sequence_length:
            raise ValueError(
                "Buffer temporal incompleto: "
                f"{len(raw_frames)} frames recibidos; "
                f"se requieren {self.sequence_length}."
            )

        raw_frames = raw_frames[-self.sequence_length:]

        last = len(raw_frames) - 1
        indices = [
            min(i * self.sample_one_each, last)
            for i in range(self.frames_per_window)
        ]

        return raw_frames[indices]

    def __call__(
        self,
        raw_frames: np.ndarray,
    ) -> torch.Tensor:
        sampled_frames = self.select_temporal_frames(
            raw_frames
        )

        frames = []
        for frame_bgr in sampled_frames:
            frame_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )
            frames.append(
                self.transform(frame_rgb)
            )

        clip = torch.stack(frames, dim=1)
        return clip.unsqueeze(0).contiguous()


class Predictor:
    def __init__(
        self,
        checkpoint_path: str,
        config: DictConfig,
        device: str = "auto",
    ):
        self.cfg = config
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No existe el checkpoint: "
                f"{self.checkpoint_path}"
            )

        self.preprocessor = ClipPreprocessor(self.cfg)
        self.sequence_length = (
            self.preprocessor.sequence_length
        )
        self.sample_one_each = (
            self.preprocessor.sample_one_each
        )
        self.frames_per_window = (
            self.preprocessor.frames_per_window
        )
        self.input_size = self.preprocessor.input_size

        self.device = self._resolve_device(device)

        (
            self.model,
            self.num_classes,
            self.inferred_architecture,
        ) = self._load_model()

        self.class_names = resolve_class_names(
            self.cfg,
            self.num_classes,
        )

        print("Predictor inicializado")
        print("----------------------")
        print(f"Checkpoint       : {self.checkpoint_path}")
        print(f"Device           : {self.device}")
        print(
            "Reconstruccion   : "
            "igual a eval_model_fixed.py"
        )
        print(
            f"Arquitectura     : "
            f"{self.inferred_architecture}"
        )
        print(f"num_classes      : {self.num_classes}")
        print(
            f"sequence_length  : {self.sequence_length}"
        )
        print(
            f"sample_one_each  : {self.sample_one_each}"
        )
        print(
            f"frames al modelo : {self.frames_per_window}"
        )
        print(
            "Input esperado   : "
            f"(1, 3, {self.frames_per_window}, "
            f"{self.input_size}, {self.input_size})"
        )
        print(f"Clases           : {self.class_names}")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        requested = torch.device(device)

        if (
            requested.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "Se solicito CUDA, pero "
                "torch.cuda.is_available() es False."
            )

        return requested

    def _load_model(self):
        checkpoint = torch.load(
            self.checkpoint_path,
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

        model, num_classes, inferred = (
            build_model_from_state_dict(
                state_dict,
                device=self.device,
            )
        )

        if "epoch" in checkpoint:
            print(
                f"Epoca checkpoint : "
                f"{checkpoint['epoch']}"
            )

        return model, num_classes, inferred

    def preprocess(
        self,
        raw_frames: np.ndarray,
    ) -> torch.Tensor:
        return self.preprocessor(raw_frames)

    @torch.no_grad()
    def predict_with_confidence(
        self,
        raw_frames: np.ndarray,
    ) -> tuple[str, float, np.ndarray]:
        input_tensor = self.preprocess(
            raw_frames
        ).to(
            self.device,
            dtype=torch.float32,
        )

        logits = self.model(input_tensor)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        pred_idx = int(
            probabilities.argmax(dim=1).item()
        )
        confidence = float(
            probabilities[0, pred_idx].item()
        )

        label = (
            self.class_names[pred_idx]
            if pred_idx < len(self.class_names)
            else f"class_{pred_idx}"
        )

        return (
            label,
            confidence,
            probabilities[0].cpu().numpy(),
        )

    def predict(
        self,
        raw_frames: np.ndarray,
    ) -> str:
        label, _, _ = self.predict_with_confidence(
            raw_frames
        )
        return label


if __name__ == "__main__":
    raise SystemExit(
        "Este archivo define Predictor. "
        "Ejecuta demo.py para inferencia."
    )
