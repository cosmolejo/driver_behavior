import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple


class SegmentDataset(Dataset):
    """
    Cada __getitem__ devuelve UN segmento completo (ej. una carpeta
    seg_001248-001326_reaching), con TODAS sus windows apiladas en un solo
    tensor (num_windows, C, T, H, W) y un único label (todas las windows de
    un segmento comparten la misma actividad).

    Esto reemplaza el esquema anterior (una window = un item independiente
    mezclado con windows de otros segmentos en el mismo batch) por uno donde
    el "bag" de windows de un segmento se mantiene junto, para poder
    promediar su loss antes del backward (ver trainer.py).

    Reglas de ventaneo:
      - Segmento con >= sequence_length frames: sliding window normal con
        `stride` fijo; todas las windows de ESE segmento tienen
        T = sequence_length.
      - Segmento con < sequence_length frames: una única window que usa
        TODOS los frames disponibles (T = num_frames del segmento). Antes,
        estos segmentos cortos no generaban ninguna window y quedaban
        completamente fuera del entrenamiento (varios en el DMD tienen
        15-25 frames con sequence_length=32).
    """

    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 32,
        sample_one_each: int = 1,
        stride: int = 8,
        transform=None,
    ):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.sample_one_each = sample_one_each
        self.stride = stride
        self.transform = transform

        # Cada entrada: (video_path, label, starts, window_len)
        #   starts: lista de frames iniciales de cada window del segmento
        #   window_len: largo (en frames) de esas windows (igual para todas
        #               las windows de un mismo segmento)
        self.segments: List[Tuple[str, int, List[int], int]] = []

        self.class_to_idx = {
            cls_name: i
            for i, cls_name in enumerate(sorted(os.listdir(root_dir)))
        }

        self._build_index()

    def _build_index(self):
        for cls_name, label in self.class_to_idx.items():
            cls_path = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for session_name in os.listdir(cls_path):
                session_path = os.path.join(cls_path, session_name)
                if not os.path.isdir(session_path):
                    continue
                for video_name in os.listdir(session_path):
                    video_path = os.path.join(session_path, video_name, "face")
                    if not os.path.isdir(video_path):
                        continue

                    num_frames = len(os.listdir(video_path))
                    if num_frames == 0:
                        continue

                    if num_frames >= self.sequence_length:
                        starts = list(range(
                            0, num_frames - self.sequence_length + 1, self.stride
                        ))
                        window_len = self.sequence_length
                    else:
                        # Segmento corto: una sola window con todos sus frames
                        starts = [0]
                        window_len = num_frames

                    if starts:
                        self.segments.append((video_path, label, starts, window_len))

    def __len__(self):
        return len(self.segments)

    def _load_window(self, video_path: str, start: int, window_len: int) -> torch.Tensor:
        frame_files = sorted(os.listdir(video_path))
        end = start + window_len
        sequence_files = frame_files[start:end:self.sample_one_each]
        frames = []
        for f in sequence_files:
            img = cv2.imread(os.path.join(video_path, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return torch.stack(frames, dim=1)  # (C, T, H, W)

    def __getitem__(self, idx):
        video_path, label, starts, window_len = self.segments[idx]
        windows = [self._load_window(video_path, s, window_len) for s in starts]
        windows_tensor = torch.stack(windows, dim=0)  # (num_windows, C, T, H, W)
        # (session_name, segment_name) — útil para debug/logging
        name = Path(video_path).parts[-3:-1]
        return windows_tensor, torch.tensor(label, dtype=torch.long), name


def segment_collate_fn(batch):
    """
    El DataLoader se usa con batch_size=1 (un segmento por iteración): cada
    __getitem__ ya devuelve el "batch" de windows de ese segmento. Este
    collate simplemente desempaqueta la lista de un solo elemento.
    """
    return batch[0]