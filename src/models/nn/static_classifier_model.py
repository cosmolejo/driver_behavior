"""


"""
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn


class StaticClassifierModel(nn.Module):
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
        self.relu = nn.ReLU()
        #self.temporal_model = temporal_model
        self.last_layer = nn.Linear( input_size, num_classes) #2 *

    def forward(self, x_input, lengths):
        batch_size = x_input.size(0)
        num_frames = x_input.size(1)
        device = x_input.device  # Aseguramos que todo se calcule en la misma tarjeta

        # 1. Extraemos características de TODOS los frames
        x_input = x_input.view(-1, x_input.size(2), x_input.size(3), x_input.size(4))
        x = self.relu(self.backbone_model(x_input))

        # Recuperamos la forma: (Batch, Frames, Features)
        x = x.view(batch_size, num_frames, -1)

        # 2. Clasificamos cada frame de forma independiente
        # PyTorch es inteligente: si le pasas un tensor 3D a una capa Linear,
        # aplica la operación automáticamente a lo largo de la última dimensión.
        logits = self.last_layer(x)  # Nueva forma: (Batch, Frames, num_classes)

        # 3. Creamos una "Máscara" para el Padding 🎭
        # Esto crea un tensor de True/False donde False representa los ceros artificiales
        arange_tensor = torch.arange(num_frames, device=device).unsqueeze(0)
        lengths_tensor = lengths.unsqueeze(1).to(device)
        mask = arange_tensor < lengths_tensor

        # Ajustamos la forma de la máscara a (Batch, Frames, 1) para multiplicarla
        mask = mask.unsqueeze(-1)

        # 4. Apagamos las predicciones de los frames falsos (las volvemos cero)
        logits_masked = logits * mask

        # 5. Calculamos el promedio temporal real
        # Sumamos las predicciones a lo largo del eje del tiempo (dim=1)
        sum_logits = logits_masked.sum(dim=1)  # Forma: (Batch, num_classes)

        # Dividimos por la cantidad real de frames de cada video para tener el promedio exacto
        mean_logits = sum_logits / lengths.unsqueeze(1).to(device)

        return mean_logits