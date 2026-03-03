"""


"""
from torch import nn


class ClassifierModel(nn.Module):
    def __init__(
            self,
            backbone_model: nn.Module,
            temporal_model: nn.Module,
            input_size: int,
            num_classes: int = 5,

    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.temporal_model = temporal_model
        self.last_layer = nn.Linear(2 * input_size, num_classes)
        #TODO: add sigmoid for classification
    def forward(self, x_input):
        x = self.backbone_model(x_input)
        x = self.temporal_model(x)
        x = self.last_layer(x)
        return x
