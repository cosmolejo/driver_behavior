import torch
from torch import nn

class SoftmaxModel(nn.Module):
    def __init__(
            self,
            face_model: nn.Module,
            body_model: nn.Module,
    ):
        super().__init__()
        self.face_model = face_model
        self.body_model = body_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_face, x_body):
        feat_cam1 = self.face_model(x_face)
        feat_cam2 = self.body_model(x_body)

        combined_features = feat_cam1 + feat_cam2

        probs = self.softmax(combined_features)

        return probs
