"""


"""
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn


class ClassifierModel(nn.Module):
    def __init__(
            self,
            backbone_model: nn.Module,
            temporal_model: nn.Module,
            input_size: int,
            num_classes: int = 5,

    ):
        #TODO agregar window_size, tambien en image loader
        super().__init__()
        self.backbone_model = backbone_model
        #self.temporal_model = temporal_model
        self.last_layer = nn.Linear( input_size, num_classes) #2 *
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x_input, lengths):
        batch_size = x_input.size(0)
        num_frames = x_input.size(1)

        x_input = x_input.view(-1, x_input.shape[2], x_input.shape[3], x_input.shape[4])
        x = self.backbone_model(x_input)

        x = x.view(batch_size, num_frames, -1)
        """
        x_packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        x_packed, (h_n, c_n) = self.temporal_model(x_packed)

        x, _ = pad_packed_sequence(x_packed, batch_first=True)
        """
        x = x[torch.arange(batch_size), lengths - 1]

        x = self.last_layer(x)
        #x = self.sigmoid(x)

        return x
