import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, T, D)

        weights = self.attn(x)
        weights = torch.softmax(weights, dim=1)

        pooled = (weights * x).sum(dim=1)

        return pooled

class MobileNetLSTMAttention(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        lstm_layers=2,
        dropout=0.3,
        freeze_backbone=False,
        use_transformer=False,
        transformer_layers=2,
        nhead=8,
        feedforward_dim=1024,
    ):
        super().__init__()

        backbone = mobilenet_v3_large(weights="DEFAULT")

        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feature_dim = 960

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # --- Optional Transformer ---
        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim*2,
                nhead=nhead,
                dim_feedforward=feedforward_dim,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers
            )

        # --- Temporal attention ---
        self.temporal_attention = TemporalAttention(hidden_dim)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x:
            (B, C, T, H, W)
        """

        B, C, T, H, W = x.shape

        # Merge time dimension into batch for CNN
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        features = self.feature_extractor(x)
        features = self.pool(features).flatten(1)  # (B*T, D)
        features = features.view(B, T, -1)         # (B, T, D)

        # LSTM
        lstm_out, _ = self.lstm(features)          # (B, T, 2*hidden_dim)

        # Optional Transformer
        if self.use_transformer:
            lstm_out = self.transformer(lstm_out)

        # Temporal Attention Pooling
        temporal_feature = self.temporal_attention(lstm_out)  # (B, 2*hidden_dim)

        # Classification
        logits = self.classifier(temporal_feature)

        return logits
    


def get_model(
    num_classes,
    hidden_dim=256,
    lstm_layers=2,
    dropout=0.3,
    freeze_backbone=True,
    use_transformer=False,
    transformer_layers=2,
    nhead=8,
    feedforward_dim=1024,
):
    return MobileNetLSTMAttention(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        lstm_layers=lstm_layers,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        use_transformer=use_transformer,
        transformer_layers=transformer_layers,
        nhead=nhead,
        feedforward_dim=feedforward_dim,
    )