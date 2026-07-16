import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from typing import Callable, Optional


class FocalLossWrapper(nn.Module):
    """
    torchvision.ops.sigmoid_focal_loss espera:
      - inputs: logits (N, C)
      - targets: one-hot float (N, C)
    pero el resto del pipeline maneja labels como índices de clase (N,).
    Este wrapper hace el one-hot internamente para que la loss tenga la
    misma firma que CrossEntropyLoss: loss(outputs, labels) -> (N,).

    reduction siempre es 'none': la agregación (promedio por segmento) se
    hace en trainer.py, no acá.
    """

    def __init__(self, num_classes: int, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        per_class_loss = ops.sigmoid_focal_loss(
            inputs, targets_onehot, alpha=self.alpha, gamma=self.gamma, reduction="none"
        )  # (N, C)
        return per_class_loss.sum(dim=1)  # (N,) — una loss escalar por muestra


class LossFactory:
    """
    Interfaz unificada: toda loss devuelta es un nn.Module con firma
    loss(outputs, labels) -> tensor (N,), SIN reducir. La agregación
    (promedio por segmento, luego por batch de segmentos) se hace siempre
    en trainer.py, para poder implementar el esquema de loss-por-segmento.
    """

    @staticmethod
    def get_loss(
        loss_name: str,
        num_classes: int = 3,
        weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Callable:
        if loss_name in ("cross_entropy", "CE_label_smoothing", "CE_weight"):
            ce_kwargs = dict(kwargs)
            ce_kwargs["reduction"] = "none"
            if weight is not None:
                ce_kwargs["weight"] = weight
            return nn.CrossEntropyLoss(**ce_kwargs)
        elif loss_name == "sigmoid_focal_loss":
            return FocalLossWrapper(num_classes=num_classes, **kwargs)
        else:
            raise ValueError(f"loss_fn desconocida: {loss_name}")