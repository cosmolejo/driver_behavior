"""
Predictor para la arquitectura actual MobileNetV3-Large + BiLSTM +
Temporal Attention.

Cambios respecto al predictor antiguo:
- No usa Hydra.
- No usa SetupFactory.
- No usa label_encoder.joblib.
- Carga checkpoints actuales con `model_state_dict`.
- Reproduce el preprocesamiento de entrenamiento:
    BGR -> RGB -> Resize(112, 112) -> ToTensor -> ImageNet Normalize.
- Reproduce el muestreo temporal del SegmentDataset.
- Entrega al modelo tensores (B, C, T, H, W).
"""

import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from model import get_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_INPUT_SIZE = 112


def infer_model_kwargs(state_dict: dict) -> dict:
    """
    Infiere del checkpoint los parámetros arquitectónicos que sí quedan
    codificados directamente en el state_dict.
    """
    lstm_key = "lstm.weight_ih_l0"
    if lstm_key not in state_dict:
        raise RuntimeError(
            f"No se encontró {lstm_key!r}. "
            "El checkpoint no parece pertenecer a la arquitectura actual."
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
        raise RuntimeError("No se pudo inferir el número de capas LSTM.")

    lstm_layers = max(layer_ids) + 1

    classifier_keys = sorted(
        key
        for key in state_dict
        if key.startswith("classifier.") and key.endswith(".weight")
    )
    if not classifier_keys:
        raise RuntimeError(
            "No se encontró la capa final del classifier en el checkpoint."
        )

    num_classes = int(state_dict[classifier_keys[-1]].shape[0])

    inferred = {
        "hidden_dim": hidden_dim,
        "lstm_layers": lstm_layers,
        "num_classes": num_classes,
        "use_transformer": any(
            key.startswith("transformer.") for key in state_dict
        ),
    }

    # Si existe Transformer, algunos parámetros adicionales sí pueden
    # deducirse del state_dict. `nhead` no se puede inferir de forma fiable,
    # por lo que se conserva el valor del YAML.
    if inferred["use_transformer"]:
        transformer_layer_ids = set()
        pattern = re.compile(r"^transformer\.layers\.(\d+)\.")
        for key in state_dict:
            match = pattern.match(key)
            if match:
                transformer_layer_ids.add(int(match.group(1)))

        if transformer_layer_ids:
            inferred["transformer_layers"] = max(transformer_layer_ids) + 1

        ff_key = "transformer.layers.0.linear1.weight"
        if ff_key in state_dict:
            inferred["feedforward_dim"] = int(state_dict[ff_key].shape[0])

    return inferred


def resolve_class_names(cfg: DictConfig, num_classes: int) -> list[str]:
    """
    Obtiene nombres de clase coherentes con `label_mode`.

    Para los modos usados actualmente no hace falta construir DMD ni
    serializar un LabelEncoder.
    """
    label_mode = str(cfg.get("label_mode", "")).lower()

    known = {
        "binary_phone": ["safe", "phone"],
        "macro": ["reaching", "safe", "unsafe"],
        "ternary_clean": ["reaching", "safe", "unsafe"],
    }

    if label_mode in known and len(known[label_mode]) == num_classes:
        return known[label_mode]

    # Para esquemas finos, reutiliza la definición central del proyecto
    # cuando esté disponible.
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
                "AVISO: no fue posible obtener nombres de clases finas "
                f"desde fine_labels.py: {exc}"
            )

    return [f"class_{i}" for i in range(num_classes)]


class Predictor:
    def __init__(
        self,
        checkpoint_path: str,
        config: DictConfig,
        device: str = "auto",
    ):
        """
        Prepara el modelo una sola vez.

        Parameters
        ----------
        checkpoint_path:
            Ruta a un checkpoint actual del proyecto (.pth).
        config:
            Configuración cargada con OmegaConf.
        device:
            "auto", "cpu", "cuda" o un device PyTorch válido.
        """
        self.cfg = config
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No existe el checkpoint: {self.checkpoint_path}"
            )

        self.sequence_length = int(self.cfg.sequence_length)
        self.sample_one_each = max(1, int(self.cfg.sample_one_each))
        self.frames_per_window = max(
            1,
            math.ceil(self.sequence_length / self.sample_one_each),
        )
        self.input_size = int(
            self.cfg.get("input_size", DEFAULT_INPUT_SIZE)
        )

        self.device = self._resolve_device(device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ])

        self.model, self.num_classes = self._load_model()
        self.class_names = resolve_class_names(
            self.cfg,
            self.num_classes,
        )

        print("Predictor inicializado")
        print("----------------------")
        print(f"Checkpoint       : {self.checkpoint_path}")
        print(f"Device           : {self.device}")
        print(f"sequence_length  : {self.sequence_length}")
        print(f"sample_one_each  : {self.sample_one_each}")
        print(f"frames al modelo : {self.frames_per_window}")
        print(
            f"Input esperado   : "
            f"(1, 3, {self.frames_per_window}, "
            f"{self.input_size}, {self.input_size})"
        )
        print(f"Clases           : {self.class_names}")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        requested = torch.device(device)

        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Se solicitó CUDA, pero torch.cuda.is_available() es False."
            )

        return requested

    def _load_model(self):
        # Los checkpoints actuales pueden contener optimizer, scheduler,
        # RNG y metadata Python. Para checkpoints propios/de confianza se
        # carga explícitamente con weights_only=False.
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)

        if not isinstance(state_dict, dict):
            raise TypeError(
                "El checkpoint no contiene un state_dict reconocible."
            )

        inferred = infer_model_kwargs(state_dict)
        num_classes = inferred.pop("num_classes")

        model_kwargs = {}
        if (
            "model_kwargs" in self.cfg
            and self.cfg.model_kwargs is not None
        ):
            model_kwargs = OmegaConf.to_container(
                self.cfg.model_kwargs,
                resolve=True,
            )

        # Solo pasar argumentos que entiende get_model().
        allowed = {
            "hidden_dim",
            "lstm_layers",
            "dropout",
            "freeze_backbone",
            "freeze_bn",
            "use_transformer",
            "transformer_layers",
            "nhead",
            "feedforward_dim",
        }
        model_kwargs = {
            key: value
            for key, value in model_kwargs.items()
            if key in allowed
        }

        # Para los parámetros deducibles, manda el checkpoint.
        model_kwargs["hidden_dim"] = inferred["hidden_dim"]
        model_kwargs["lstm_layers"] = inferred["lstm_layers"]
        model_kwargs["use_transformer"] = inferred["use_transformer"]

        if "transformer_layers" in inferred:
            model_kwargs["transformer_layers"] = inferred[
                "transformer_layers"
            ]
        if "feedforward_dim" in inferred:
            model_kwargs["feedforward_dim"] = inferred[
                "feedforward_dim"
            ]

        model = get_model(
            num_classes=num_classes,
            **model_kwargs,
        )

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )

        if missing or unexpected:
            raise RuntimeError(
                "El checkpoint no coincide con el modelo construido.\n"
                f"Parámetros faltantes: {missing}\n"
                f"Parámetros inesperados: {unexpected}"
            )

        model = model.to(self.device)
        model.eval()

        print(f"Arquitectura      : {model_kwargs}")
        if "epoch" in checkpoint:
            print(f"Época checkpoint  : {checkpoint['epoch']}")

        return model, num_classes

    def _select_temporal_frames(
        self,
        raw_frames: np.ndarray,
    ) -> np.ndarray:
        """
        Reproduce `_window_indices()` del SegmentDataset.

        El buffer mantiene `sequence_length` frames crudos. Después se
        seleccionan `frames_per_window` frames separados por
        `sample_one_each`.

        Ejemplo actual:
            sequence_length=32
            sample_one_each=2
            -> índices 0, 2, 4, ..., 30
            -> 16 frames al modelo.
        """
        if len(raw_frames) < self.sequence_length:
            raise ValueError(
                "Buffer temporal incompleto: "
                f"se recibieron {len(raw_frames)} frames, "
                f"se necesitan {self.sequence_length}."
            )

        # Si llega un buffer más grande, usamos la ventana más reciente.
        raw_frames = raw_frames[-self.sequence_length:]

        last = len(raw_frames) - 1
        indices = [
            min(i * self.sample_one_each, last)
            for i in range(self.frames_per_window)
        ]

        return raw_frames[indices]

    def preprocess(self, raw_frames: np.ndarray) -> torch.Tensor:
        """
        Convierte frames OpenCV BGR al tensor usado durante entrenamiento.

        Entrada:
            (sequence_length, H, W, 3), BGR uint8

        Salida:
            (1, C, T, H, W), float32 normalizado
        """
        sampled_frames = self._select_temporal_frames(raw_frames)

        frames = []
        for frame_bgr in sampled_frames:
            frame_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )
            frames.append(self.transform(frame_rgb))

        # Cada frame es (C, H, W). Se apilan sobre T:
        # (C, T, H, W), igual que SegmentDataset._load_window().
        clip = torch.stack(frames, dim=1)

        # Batch:
        # (1, C, T, H, W)
        return clip.unsqueeze(0)

    @torch.no_grad()
    def predict_with_confidence(
        self,
        raw_frames: np.ndarray,
    ) -> tuple[str, float, np.ndarray]:
        input_tensor = self.preprocess(raw_frames).to(
            self.device,
            dtype=torch.float32,
        )

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        pred_idx = int(probabilities.argmax(dim=1).item())
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

    def predict(self, raw_frames: np.ndarray) -> str:
        label, _, _ = self.predict_with_confidence(raw_frames)
        return label


if __name__ == "__main__":
    raise SystemExit(
        "Este archivo define Predictor. "
        "Ejecuta demo.py para hacer inferencia sobre video."
    )
