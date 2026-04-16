"""
Drive and act Dataloader

"""
import logging
import random

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
        self.stride = config.stride
        self.camera = config.camera

        # Cargar los datos crudos
        conv = { 1: lambda x: int(x), 2: lambda x: int(x)}
        data = np.loadtxt(self.config.label_path + "dmd_vicomtech.csv", dtype=str, delimiter=",", skiprows=1,
                          converters=conv)

        self.samples = []
        raw_labels = []

        # Procesar los slices
        for row in data:
            file_path = '_'.join(row[0].strip().split('_')[:-2]).replace(';','_')
            file_id = f'{file_path}_{self.camera}_240'
            f_start = int(row[1])
            f_end = int(row[2])
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
                    stride = self.stride
                    for start in range(f_start, f_end - self.window_size + 1, stride):
                        self.samples.append(
                            {'file_id': file_id, 'frame_start': start, 'frame_end': start + self.window_size})
                        raw_labels.append(label)

        # Codificar las etiquetas resultantes
        le = LabelEncoder()
        self.y = le.fit_transform(raw_labels)

    def __getitem__(self, index: int):

        if self.config.setup == "multy_camera":
            return self.get_one(index, 'face'), self.get_one(index, 'body')
        else:
            return self.get_one(index)


    def __len__(self):
        return len(self.samples)

    def get_one(self, index: int, camera = None):
        while True:
            sample = self.samples[index]
            if camera is not None:
                file_path = str(sample['file_id']).split('_')
                file_path[-2] = camera
                sample['file_id'] = '_'.join(file_path)
            video = slice_frame(self.dataset, sample)

            # Verificamos si logramos extraer exactamente la cantidad requerida de frames reales
            if len(video) == self.window_size:
                break  # ¡Éxito! Salimos del bucle.

            # Si faltaron frames (video corrupto o demasiado corto), elegimos otra muestra al azar
            # Esto evita que el DataLoader reciba datos inconsistentes y no usamos padding.
            logging.warning(f"Muestra {index} incompleta. Reintentando con otra...")
            index = random.randint(0, len(self.samples) - 1)

            # Continúa el flujo normal, sabiendo que 'video' tiene exactamente el tamaño de window_size
        video_tensors = [transforms.ToTensor()(
            Image.fromarray(np.uint8(frame * 255)).resize((224, 224))
        ) for frame in video]

        source_video = torch.stack(video_tensors)
        label = self.y[index]

        return source_video, label