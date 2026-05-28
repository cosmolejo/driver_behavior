"""
DMD Dataloader - lee frames pre-extraidos como JPEG desde la estructura de
particiones generada por split_dataset.py.

Estructura esperada en disco
----------------------------
    <data_path>/<SPLIT>/<CLASE>/<video_folder>/seg_<ini>-<fin>_<clase>/<camera>/<frame:06d>.jpg

donde:
    <data_path>     : raiz que contiene TRAIN/ VALIDATION/ TEST/
                      (p.ej. /home/antares/Tesis_Data/dmd)
    <SPLIT>         : TRAIN | VALIDATION | TEST
    <CLASE>         : reaching | safe | unsafe   (== etiqueta del segmento)
    <video_folder>  : nombre del video original
    seg_*           : un segmento (una fila del CSV original) ya filtrado
    <camera>        : body | face

Cada Dataset carga UN solo split (se pasa por argumento). Las ventanas se
generan recorriendo los frames REALES de cada carpeta seg_*/<camera>:
  - se ordenan los frames existentes,
  - se parten en tramos CONTIGUOS (sin huecos numericos),
  - sobre cada tramo se sacan ventanas secuenciales de window_size frames
    deslizando con stride.
La etiqueta se toma del nombre de la carpeta de clase (no del CSV).

El preprocesamiento (resize, etc.) ya esta aplicado en los JPEG; aqui solo:
cargar -> tensor -> normalizar ImageNet. Si los JPEG estan en BGR (extraidos
con --no-rgb), se hace BGR->RGB segun config.frames_are_rgb.
"""
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors


# Normalizacion estandar de torchvision (ImageNet)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Orden FIJO de clases -> indice. Debe coincidir en numero con config.model.num_classes.
# Fijarlo explicitamente garantiza el mismo mapeo en TRAIN/VALIDATION/TEST,
# aunque a un split le faltara alguna clase.
CLASSES = ["reaching", "safe", "unsafe"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Nombres validos de split (subcarpetas directas de data_path)
VALID_SPLITS = {"TRAIN", "VALIDATION", "TEST"}


class DMD(Dataset):
    def __init__(self, config, split, transform=None):
        """
        Args:
            config: configuracion (Hydra/OmegaConf o dict-like).
            split : 'TRAIN' | 'VALIDATION' | 'TEST'. Determina la subcarpeta.
        """
        super().__init__()

        if split not in VALID_SPLITS:
            raise ValueError(
                "split debe ser uno de {}, recibido: {!r}".format(sorted(VALID_SPLITS), split)
            )

        if config.model.type == "static":
            raise NotImplementedError(
                "El modo 'static' aun no esta implementado para la estructura de "
                "particiones. Usa config.model.type == 'temporal'."
            )
        if config.model.type != "temporal":
            raise NotImplementedError(
                "config.model.type desconocido: {!r} (solo 'temporal').".format(config.model.type)
            )

        self.config = config
        self.split = split
        self.window_size = int(config.window_size)
        self.stride = int(config.stride)
        self.camera = config.camera
        self.transform = transform
        # Si los JPEG ya estan en RGB, no se hace conversion BGR->RGB.
        self.frames_are_rgb = getattr(config, "frames_are_rgb", False)

        # Raiz del split: <data_path>/<SPLIT>
        self.split_dir = Path(config.data_path) / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                "No existe la carpeta del split: {}".format(self.split_dir)
            )

        # Construir la lista de ventanas (solo metadatos, no imagenes)
        # Cada entrada: {'dir': Path a seg/<camera>, 'start': int, 'label': int}
        self.windows = self._build_windows()

        # Vector de etiquetas alineado con self.windows (util para class weights)
        self.y = np.array([w["label"] for w in self.windows], dtype=np.int64)

        # Reporte de distribucion del split
        self._log_distribution()

    # ------------------------------------------------------------------ #
    # Construccion de ventanas
    # ------------------------------------------------------------------ #
    def _build_windows(self):
        windows = []
        n_segments = 0
        n_segments_sin_ventana = 0

        # Recorremos solo las clases conocidas para no depender del orden del FS
        for clase in CLASSES:
            class_dir = self.split_dir / clase
            if not class_dir.is_dir():
                continue
            label = CLASS_TO_IDX[clase]

            # <clase>/<video>/seg_*/<camera>/
            for video_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                for seg_dir in sorted(p for p in video_dir.iterdir() if p.is_dir()):
                    cam_dir = seg_dir / self.camera
                    if not cam_dir.is_dir():
                        # ese segmento no tiene la camara pedida; se ignora
                        continue
                    n_segments += 1

                    frame_ids = self._existing_frame_ids(cam_dir)
                    seg_windows = self._windows_from_contiguous(frame_ids)
                    if not seg_windows:
                        n_segments_sin_ventana += 1
                    for start in seg_windows:
                        windows.append({"dir": cam_dir, "start": start, "label": label})

        logging.info(
            "[%s/%s] Segmentos: %d (sin ventana valida: %d) -> ventanas: %d",
            self.split, self.camera, n_segments, n_segments_sin_ventana, len(windows),
        )
        if not windows:
            logging.warning(
                "[%s/%s] No se genero ninguna ventana. Revisa data_path y camera.",
                self.split, self.camera,
            )
        return windows

    @staticmethod
    def _existing_frame_ids(cam_dir):
        """Lista ordenada de numeros de frame existentes en la carpeta."""
        ids = []
        for p in cam_dir.glob("*.jpg"):
            try:
                ids.append(int(p.stem))
            except ValueError:
                # nombre no numerico, se ignora
                continue
        ids.sort()
        return ids

    def _windows_from_contiguous(self, frame_ids):
        """
        Dada la lista ordenada de frames existentes, devuelve los 'start' (numero
        de frame inicial) de cada ventana. Solo se generan ventanas sobre tramos
        de numeracion CONTIGUA (sin huecos): un tramo es una secuencia
        n, n+1, n+2, ... Dentro de cada tramo se deslizan ventanas de
        window_size frames con paso stride.
        """
        starts = []
        if len(frame_ids) < self.window_size:
            return starts

        # Partir en tramos contiguos
        run_start = frame_ids[0]
        prev = frame_ids[0]
        runs = []  # lista de (primer_frame, ultimo_frame) de cada tramo contiguo
        for fid in frame_ids[1:]:
            if fid == prev + 1:
                prev = fid
                continue
            runs.append((run_start, prev))
            run_start = fid
            prev = fid
        runs.append((run_start, prev))

        # Ventanas secuenciales dentro de cada tramo contiguo
        for a, b in runs:
            run_len = b - a + 1
            if run_len < self.window_size:
                continue
            # last valid start: b - window_size + 1
            last_start = b - self.window_size + 1
            for s in range(a, last_start + 1, self.stride):
                starts.append(s)
        return starts

    def _log_distribution(self):
        counts = {c: 0 for c in CLASSES}
        for w in self.windows:
            counts[CLASSES[w["label"]]] += 1
        logging.info("[%s] Distribucion de ventanas por clase: %s", self.split, counts)

    # ------------------------------------------------------------------ #
    # Carga de un clip
    # ------------------------------------------------------------------ #
    def _load_clip(self, cam_dir, start):
        """
        Carga window_size frames consecutivos [start, start+window_size) como
        tensor (T, C, H, W) float32 normalizado ImageNet.
        Lanza FileNotFoundError si algun frame no se puede leer (no deberia
        pasar porque las ventanas se construyen sobre frames existentes y
        contiguos, pero protege ante borrados posteriores).
        """
        end = start + self.window_size  # exclusivo

        first_path = cam_dir / "{:06d}.jpg".format(start)
        first_img = cv2.imread(str(first_path))
        if first_img is None:
            raise FileNotFoundError("No se pudo leer {}".format(first_path))
        H, W = first_img.shape[:2]

        frames = np.empty((self.window_size, H, W, 3), dtype=np.uint8)
        for i, fidx in enumerate(range(start, end)):
            if i == 0:
                img = first_img
            else:
                img = cv2.imread(str(cam_dir / "{:06d}.jpg".format(fidx)))
                if img is None:
                    raise FileNotFoundError(
                        "No se pudo leer {}".format(cam_dir / "{:06d}.jpg".format(fidx))
                    )
            if not self.frames_are_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames[i] = img
            # frames: (T, H, W, C) uint8
            # -> tensor (T, C, H, W) uint8
            clip = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

            if self.transform:
                clip = tv_tensors.Video(clip)  # ahora v2 lo reconoce como video
                t = self.transform(clip)
            else:
                t = clip.float().div_(255.0)
                t = (t - IMAGENET_MEAN) / IMAGENET_STD
            return t

    # ------------------------------------------------------------------ #
    # API Dataset
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        w = self.windows[index]
        clip = self._load_clip(w["dir"], w["start"])
        return clip, int(w["label"])


if __name__ == "__main__":
    # Smoke test minimo (requiere un config-like). Se ejecuta solo a mano.
    logging.basicConfig(level=logging.INFO)