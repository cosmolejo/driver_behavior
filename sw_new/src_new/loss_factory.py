import torch.nn as nn
import torchvision.ops as ops
from typing import Callable
class LossFactory:
    loss_dict = {
        'cross_entropy': nn.CrossEntropyLoss,
        'CE_label_smoothing': nn.CrossEntropyLoss,
        'CE_weight': nn.CrossEntropyLoss,
        'sigmoid_focal_loss': ops.sigmoid_focal_loss,



    }

    @staticmethod
    def get_loss(loss_name: str) -> Callable:

        return LossFactory.loss_dict[loss_name]
