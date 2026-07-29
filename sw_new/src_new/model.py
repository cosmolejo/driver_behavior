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
        freeze_bn=True,
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

        # Ver docstring de `train()` mas abajo.
        self.freeze_bn = freeze_bn

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

    def train(self, mode: bool = True):
        """
        Con `freeze_bn=True`, las 46 capas BatchNorm del backbone quedan
        SIEMPRE en modo eval, incluso cuando el modelo esta entrenando.

        Motivo
        ------
        Cada forward procesa `max_windows_per_forward * sequence_length`
        frames (por defecto 8*32 = 256) que provienen todos del MISMO
        segmento: mismo sujeto, misma actividad, misma iluminacion, casi
        el mismo encuadre. En modo train, BatchNorm normaliza con las
        estadisticas de ese batch, o sea que hace normalizacion POR
        SEGMENTO y le filtra al modelo informacion del propio segmento que
        esta clasificando. Eso hace que train/loss baje a ~0 sin que el
        modelo haya aprendido nada transferible, y produce una brecha
        enorme al pasar a model.eval() (que usa las running_stats
        globales).

        Ademas, `freeze_backbone=True` NO alcanza para evitar esto:
        poner requires_grad=False congela los gradientes, pero BatchNorm
        sigue actualizando running_mean/running_var en modo train. Hay que
        forzar el modo eval explicitamente, que es lo que hace este
        override.

        Este metodo se llama en cada `model.train()` de train_one_epoch,
        asi que la garantia se mantiene durante todo el entrenamiento.
        """
        super().train(mode)
        if self.freeze_bn:
            for m in self.feature_extractor.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

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
    freeze_bn=True,
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
        freeze_bn=freeze_bn,
        use_transformer=use_transformer,
        transformer_layers=transformer_layers,
        nhead=nhead,
        feedforward_dim=feedforward_dim,
    )