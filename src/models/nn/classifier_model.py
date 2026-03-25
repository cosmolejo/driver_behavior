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
        self.temporal_model = temporal_model
        # El 2 * input_size sugiere que tu modelo temporal es bidireccional
        self.last_layer = nn.Linear(2 * input_size, num_classes) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_input):
        batch_size = x_input.size(0)
        num_frames = x_input.size(1)

        # 1. Redimensionar para pasar todo el batch de frames por el backbone espacial (CNN)
        # Pasa de (Batch, Frames, C, H, W) a (Batch * Frames, C, H, W)
        x_input = x_input.view(-1, x_input.shape, x_input.shape, x_input.shape)
        x = self.backbone_model(x_input)
        
        # 2. Restaurar la dimensión temporal
        # Pasa de (Batch * Frames, Features) a (Batch, Frames, Features)
        x = x.view(batch_size, num_frames, -1)
        
        # 3. Procesamiento temporal
        # Ya no hay necesidad de empacar/desempacar. El modelo temporal recibe el tensor 3D directamente
        x, (h_n, c_n) = self.temporal_model(x)

        # 4. Extraer el último frame de la secuencia para la clasificación final
        # Antes buscabas el índice exacto con 'lengths - 1', ahora siempre es la última posición [-1]
        x_last_frame = x[:, -1, :] 
        
        # 5. Capa de salida y activación
        out = self.last_layer(x_last_frame)
        out = self.sigmoid(out)
        
        return out