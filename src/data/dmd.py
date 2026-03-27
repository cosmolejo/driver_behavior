"""
Drive and act Dataloader

"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from utils.video_slicer import slice_frame


class DMD(Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.window_size = config.window_size
        self.model_type = config.model_type
        self.dataset = config.data_path

        # Cargar los datos crudos
        conv = {1: lambda x: int(x), 2: lambda x: int(x), 4: lambda x: int(x)}
        data = np.loadtxt(self.config.label_path + "distracted_driving.csv", dtype=str, delimiter=",", skiprows=1,
                          converters=conv)

        self.samples = []
        raw_labels = []

        # Procesar los slices
        for row in data:
            file_id = row.strip()
            f_start = row[1]
            f_end = row[2]
            label = row[3].strip()

            delta = f_end - f_start

            if self.config.model_type == "static":
                self.samples.append({'file_id': file_id, 'frame_start': 0, 'frame_end': self.window_size})
                raw_labels.append(label)

            elif self.config.model_type == "temporal":

                # Si la actividad es más corta que window_size, retrocedemos el inicio para asegurar el tamaño
                if delta < self.window_size:
                    start = max(0, f_end - self.window_size)
                    self.samples.append({'file_id': file_id, 'frame_start': start, 'frame_end': start + self.window_size})
                    raw_labels.append(label)
                else:
                    # Crear ventanas consecutivas (puedes ajustar el 'stride' si quieres solapamiento)
                    stride = self.window_size
                    for start in range(f_start, f_end - self.window_size + 1, stride):
                        self.samples.append(
                            {'file_id': file_id, 'frame_start': start, 'frame_end': start + self.window_size})
                        raw_labels.append(label)

        # Codificar las etiquetas resultantes
        le = LabelEncoder()
        self.y = le.fit_transform(raw_labels)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        # slice_frame ahora siempre retornará exactamente la cantidad de frames = window_size
        video = slice_frame(self.dataset, sample)

        video_tensors = [transforms.ToTensor()(
            Image.fromarray(np.uint8(frame * 255)).resize((224, 224))
        ) for frame in video]

        # Apilar los tensores en una dimensión temporal: [window_size, C, H, W]
        source_video = torch.stack(video_tensors)


        label = self.y[index]
        return source_video, label

    def __len__(self):
        return len(self.samples)