import torch
from torch import nn

class MultiCamModel(nn.Module):
    def __init__(
            self,
            face_model: nn.Module,
            body_model: nn.Module,
            input_size: int,
            num_classes: int = 5,
    ):
        super().__init__()
        self.face_model = face_model
        self.body_model = body_model
        self.fusion_fc1 = nn.Linear(4 * input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.final_classifier = nn.Linear(128, num_classes)  # num_classes = 3



    def forward(self, x_face, x_body):
        feat_cam1 = self.face_model(x_face)  # tamaño: [batch, 2 * input_size]
        feat_cam2 = self.body_model(x_body)  # tamaño: [batch, 2 * input_size]

        # 2. Concatenar en la dimensión de las características (dim=1)
        combined_features = torch.cat((feat_cam1, feat_cam2), dim=1)  # tamaño: [batch, 4 * input_size]

        # 3. Pasar por la red de fusión
        x = self.fusion_fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.final_classifier(x)  # tamaño: [batch, 3]

        # Retornas esto y lo metes directamente a tu CrossEntropyLoss habitual
        return logits



        return out