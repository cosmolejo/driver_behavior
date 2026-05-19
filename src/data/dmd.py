"""
Drive and Act Dataloader - lee frames pre-extraídos como JPEG.

Ya NO usa slice_frame (que decodifica .mp4); lee directamente los JPEGs
que extrajiste con extract_all_frames.py.

Estructura esperada:
    <data_path>/dmd/<grupo>/<sujeto>/<sesion>/<file_id>/<frame:06d>.jpg

donde <file_id> es algo como:
    gF_25_s3_2019-03-14T14_42_40+01_00_rgb_face_240

El preprocesamiento (resize a 224x128, BGR→RGB) ya está aplicado en los
JPEGs, así que aquí solo se hace: cargar → tensor → normalizar ImageNet.
"""
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


# Normalización estándar de torchvision (ImageNet)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DMD(Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.window_size = config.window_size
        self.frames_dir = Path(config.data_path)
        self.stride = config.stride
        self.camera = config.camera

        # Si extrajiste los JPEGs con --no-rgb (recomendado), están en BGR
        # y el Dataset hace BGR->RGB al leer. Configurable desde el config.
        self.frames_are_rgb = getattr(config, 'frames_are_rgb', False)

        # Cargar slices del CSV (lógica original sin cambios)
        conv = {1: lambda x: int(x), 2: lambda x: int(x)}
        data = np.loadtxt(
            self.config.label_path + "dmd_vicomtech.csv",
            dtype=str, delimiter=",", skiprows=1, converters=conv,
        )

        self.samples = []
        raw_labels = []

        for row in data:
            file_path = '_'.join(row[0].strip().split('_')[:-2]).replace(';', '_')
            file_id = f'{file_path}_{self.camera}_240'
            f_start = int(row[1])
            f_end = int(row[2])
            label = row[3].strip()
            delta = f_end - f_start

            if self.config.model.type == "static":
                self.samples.append({'file_id': file_id, 'frame_start': 0,
                                     'frame_end': self.window_size})
                raw_labels.append(label)
            elif self.config.model.type == "temporal":
                if delta < self.window_size:
                    start = max(0, f_end - self.window_size)
                    self.samples.append({'file_id': file_id, 'frame_start': start,
                                         'frame_end': start + self.window_size})
                    raw_labels.append(label)
                else:
                    for start in range(f_start, f_end - self.window_size + 1, self.stride):
                        self.samples.append({'file_id': file_id, 'frame_start': start,
                                             'frame_end': start + self.window_size})
                        raw_labels.append(label)

        # Filtrar samples cuyos JPEG no existen en disco
        self.samples, raw_labels = self._filter_existing_frames(self.samples, raw_labels)

        self.le = LabelEncoder()
        self.y = self.le.fit_transform(raw_labels)

    def _filter_existing_frames(self, samples, raw_labels):
        """Descarta samples cuyos JPEGs no están todos disponibles."""
        if getattr(self.config.setup, 'mode', None) == 'multicamera':
            cameras_to_check = ['face', 'body']
        else:
            cameras_to_check = [self.camera]

        available_cache: dict = {}

        def get_available(file_id_for_cam: str) -> set:
            if file_id_for_cam not in available_cache:
                folder = self.frames_dir / file_id_for_cam
                if folder.exists():
                    available_cache[file_id_for_cam] = {
                        int(p.stem) for p in folder.glob('*.jpg')
                    }
                else:
                    available_cache[file_id_for_cam] = set()
            return available_cache[file_id_for_cam]

        def file_id_for(base_file_id: str, camera: str) -> str:
            parts = base_file_id.split('_')
            parts[-2] = camera
            return '_'.join(parts)

        valid_samples = []
        valid_labels = []
        for sample, label in zip(samples, raw_labels):
            needed = range(sample['frame_start'], sample['frame_end'])
            ok = True
            for cam in cameras_to_check:
                check_id = (sample['file_id'] if cam == self.camera
                            else file_id_for(sample['file_id'], cam))
                available = get_available(check_id)
                if not all(f in available for f in needed):
                    ok = False
                    break
            if ok:
                valid_samples.append(sample)
                valid_labels.append(label)

        n_dropped = len(samples) - len(valid_samples)
        if n_dropped > 0:
            logging.warning(
                "Filtrados %d samples (%.2f%%) por frames faltantes. Restantes: %d",
                n_dropped, 100.0 * n_dropped / len(samples), len(valid_samples),
            )
        else:
            logging.info("Todos los samples tienen sus frames disponibles (%d)", len(samples))
        return valid_samples, valid_labels

    def _load_clip(self, sample, camera=None):
        """
        Carga window_size frames consecutivos como tensor (T, C, H, W) float32
        normalizado para ImageNet. Devuelve None si falta algún frame.
        """
        file_id = sample['file_id']
        if camera is not None:
            parts = file_id.split('_')
            parts[-2] = camera
            file_id = '_'.join(parts)

        clip_dir = self.frames_dir / file_id
        start = int(sample['frame_start'])
        end = int(sample['frame_end'])
        n = end - start

        # Leer primer frame para conocer dimensiones reales
        first_path = clip_dir / f'{start:06d}.jpg'
        first_img = cv2.imread(str(first_path))
        if first_img is None:
            return None
        H, W = first_img.shape[:2]

        frames = np.empty((n, H, W, 3), dtype=np.uint8)
        for i, fidx in enumerate(range(start, end)):
            if i == 0:
                img = first_img
            else:
                img = cv2.imread(str(clip_dir / f'{fidx:06d}.jpg'))
                if img is None:
                    return None
            if not self.frames_are_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames[i] = img

        # (T, H, W, C) uint8 -> (T, C, H, W) float32 [0,1] -> normalizado
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t

    def __getitem__(self, index: int):
        if self.config.setup.mode == "multicamera":
            face = self.get_one(index, 'face')[0]
            body, label = self.get_one(index, 'body')
            return face, body, label
        return self.get_one(index)

    def __len__(self):
        return len(self.samples)

    def get_one(self, index: int, camera=None):
        attempts = 0
        while True:
            sample = dict(self.samples[index])  # copia defensiva
            clip = self._load_clip(sample, camera=camera)
            if clip is not None:
                break
            logging.warning("Muestra %d falló al cargar. Reintentando.", index)
            index = random.randint(0, len(self.samples) - 1)
            attempts += 1
            if attempts > 5:
                raise RuntimeError(
                    "Demasiados reintentos cargando samples; revisa la extracción"
                )

        label = self.y[index]
        return clip, label