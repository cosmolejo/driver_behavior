import torch
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
        self.relu = nn.ReLU()
        self.temporal_model = temporal_model
        self.last_layer = nn.Linear(2 * input_size, num_classes)


    def forward(self, x_input):
        if x_input.dim() != 5:
            raise ValueError(
                f"Expected input with shape (batch, frames, channels, height, width), "
                f"but got shape {tuple(x_input.shape)}"
            )

        batch_size, num_frames, channels, height, width = x_input.shape

        # Move frames into the batch dimension: (B, T, C, H, W) -> (B*T, C, H, W)
        x_input = x_input.reshape(batch_size * num_frames, channels, height, width)

        # Extract spatial features for every frame
        x = self.relu(self.backbone_model(x_input))

        # Restore temporal structure: (B*T, F) -> (B, T, F)
        x = x.reshape(batch_size, num_frames, -1)

        # Temporal processing
        x = self.temporal_model(x)[0]


        # Take the last time step
        x_last_frame = x[:, -1, :]

        out = self.last_layer(x_last_frame)


        return out