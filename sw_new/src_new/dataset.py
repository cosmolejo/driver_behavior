import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Optional, Tuple


class SegmentDataset(Dataset):
    """
    Cada __getitem__ devuelve UN segmento completo (ej. una carpeta
    seg_001248-001326_reaching), con TODAS sus windows apiladas en un solo
    tensor (num_windows, C, T, H, W) y un �nico label (todas las windows de
    un segmento comparten la misma actividad).

    Esto reemplaza el esquema anterior (una window = un item independiente
    mezclado con windows de otros segmentos en el mismo batch) por uno donde
    el "bag" de windows de un segmento se mantiene junto, para poder
    promediar su loss antes del backward (ver trainer.py).

    Reglas de ventaneo:
      - Segmento con >= sequence_length frames: sliding window normal con
        `stride` fijo; todas las windows de ESE segmento tienen
        T = sequence_length.
      - Segmento con < sequence_length frames: una �nica window que usa
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
        label_mode: str = "macro",
        partition_report: Optional[str] = None,
        augment=None,
        normalize=None,
        temporal_stride_jitter: Optional[list] = None,
        temporal_offset_jitter: bool = False,
        subject_subset: Optional[int] = None,
        subject_subset_seed: int = 42,
    ):
        """
        label_mode:
            "macro" (default) -> 3 clases, tomadas del nombre de la carpeta
                de primer nivel. Comportamiento historico, sin cambios.
            "fine"            -> 9 clases (las actividades del DMD). Requiere
                `partition_report`, porque la actividad fina NO esta en el
                arbol de directorios: el dataset esta organizado por
                macro-clase y hay que recuperarla del CSV con la clave
                (video_folder, segment_folder).

        augment / normalize:
            Solo para entrenamiento. Cuando se usa augmentation, `transform`
            NO debe incluir Normalize: el orden correcto es

                frames crudos -> transform (por frame: Resize + ToTensor)
                              -> stack a clip (C, T, H, W), rango [0, 1]
                              -> augment(clip, label)      <- nivel CLIP
                              -> normalize

            Las transformaciones fotometricas se samplean UNA vez por
            SEGMENTO (no por frame ni por ventana): un factor de brillo
            distinto en cada frame introduciria un parpadeo inexistente en
            el dominio real, que la LSTM puede confundir con se�al temporal.
            Un segmento es una grabacion continua, asi que su iluminacion
            es la misma en todas sus ventanas.

        temporal_stride_jitter:
            Lista de pasos candidatos, ej. [1, 2, 3]. Se elige UNO por
            segmento y se usa para todas sus ventanas. Con paso s se toman
            T frames abarcando T*s frames crudos, o sea que simula una
            grabacion a otro frame-rate (o el mismo gesto mas rapido/lento).

            El paso se fija por segmento y no por ventana porque todas las
            ventanas de un segmento se apilan en un tensor: si T variara
            entre ellas, el stack fallaria.

        temporal_offset_jitter:
            Desplaza el inicio de todas las ventanas del segmento en un
            delta aleatorio. Cambia que frames concretos componen cada
            ventana entre una epoca y otra.

        subject_subset:
            Cantidad de sujetos a conservar (curva de aprendizaje). None =
            todos. Requiere `partition_report`, porque la identidad del
            sujeto no esta en el arbol de directorios.

            Los subconjuntos son ANIDADOS y deterministas: los 5 sujetos
            son un subconjunto de los 10, y asi. Sirve para medir como
            escala el rendimiento con la cantidad de conductores, que es
            la restriccion sospechada del dataset.

        En modo "fine" quedan disponibles:
            self.class_to_idx  -> {actividad: indice}, 9 entradas
            self.label_groups  -> lista de 9 ints, indice fino -> indice macro
                                  (se pasa a train_pipeline para agregar en
                                  validacion)
        """
        if label_mode not in ("macro", "fine"):
            raise ValueError(f"label_mode debe ser 'macro' o 'fine', no {label_mode!r}")
        if label_mode == "fine" and partition_report is None:
            raise ValueError(
                "label_mode='fine' requiere `partition_report` "
                "(ruta a partition_report.csv)."
            )

        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.sample_one_each = sample_one_each
        self.stride = stride
        self.transform = transform
        self.augment = augment
        self.normalize = normalize
        self.temporal_stride_jitter = temporal_stride_jitter
        self.temporal_offset_jitter = temporal_offset_jitter
        # Cantidad de frames que devuelve cada ventana. Se mantiene
        # constante pase lo que pase con el jitter, para que el stack
        # de las ventanas de un segmento sea valido.
        self.frames_per_window = max(
            1, -(-sequence_length // max(1, sample_one_each))
        )
        self.label_mode = label_mode

        # Cada entrada: (video_path, label, starts, window_len)
        #   starts: lista de frames iniciales de cada window del segmento
        #   window_len: largo (en frames) de esas windows (igual para todas
        #               las windows de un mismo segmento)
        self.segments: List[Tuple[str, int, List[int], int]] = []

        # Carpetas de primer nivel = macro-clases, en orden alfabetico.
        # Con esta convencion: 0=reaching, 1=safe, 2=unsafe.
        self.macro_to_idx = {
            cls_name: i
            for i, cls_name in enumerate(sorted(os.listdir(root_dir)))
        }

        if label_mode == "macro":
            self.class_to_idx = dict(self.macro_to_idx)
            self.label_groups = None
            self._activity_lookup = None
        else:
            from fine_labels import (
                build_fine_class_to_idx,
                build_label_groups,
                load_activity_lookup,
            )
            self.class_to_idx = build_fine_class_to_idx(partition_report)
            self.label_groups = build_label_groups(partition_report)
            self._activity_lookup = load_activity_lookup(partition_report)

        # --- Submuestreo por sujeto ---
        self.subject_subset = subject_subset
        self._subject_lookup = None
        self.selected_subjects = None
        if subject_subset is not None:
            if partition_report is None:
                raise ValueError(
                    "subject_subset requiere `partition_report`: la identidad "
                    "del sujeto no esta en el arbol de directorios."
                )
            from fine_labels import load_subject_lookup, select_subjects
            split_name = Path(root_dir).name.upper()
            self._subject_lookup = load_subject_lookup(partition_report)
            self.selected_subjects = set(
                select_subjects(partition_report, subject_subset,
                                split=split_name, seed=subject_subset_seed)
            )
            print(f"[{split_name}] submuestreo: {len(self.selected_subjects)} sujetos "
                  f"-> {sorted(self.selected_subjects)}")

        self.unmatched = []   # segmentos sin entrada en el CSV (solo modo fine)

        self._build_index()

        if label_mode == "fine":
            if self.unmatched:
                print(
                    f"AVISO: {len(self.unmatched)} segmentos de {root_dir} no "
                    f"aparecen en el partition_report y quedaron FUERA del "
                    f"dataset. Ejemplos: {self.unmatched[:3]}"
                )
            conteo = {}
            for _, lab, _, _ in self.segments:
                conteo[lab] = conteo.get(lab, 0) + 1
            inv = {v: k for k, v in self.class_to_idx.items()}
            print(f"[{Path(root_dir).name}] {len(self.segments)} segmentos, "
                  f"{len(self.class_to_idx)} clases finas:")
            for i in sorted(conteo):
                print(f"    {i}  {inv[i]:<24}{conteo[i]:>6}")

    def _build_index(self):
        for cls_name, macro_label in self.macro_to_idx.items():
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

                    # Filtro por sujeto (curva de aprendizaje)
                    if self._subject_lookup is not None:
                        subj = self._subject_lookup.get((session_name, video_name))
                        if subj is None or subj not in self.selected_subjects:
                            continue

                    # En modo "fine" la etiqueta se resuelve por
                    # (session_name, video_name) == (video_folder,
                    # segment_folder) del partition_report.
                    if self.label_mode == "macro":
                        label = macro_label
                    else:
                        activity = self._activity_lookup.get((session_name, video_name))
                        if activity is None:
                            self.unmatched.append((session_name, video_name))
                            continue
                        label = self.class_to_idx[activity]

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

    def _window_indices(self, num_frames: int, start: int, stride: int) -> List[int]:
        """
        Indices de los `self.frames_per_window` frames de una ventana.

        Siempre devuelve la misma cantidad de indices, sin importar el paso
        ni la posicion: si la ventana se pasa del final del segmento, se
        repite el ultimo frame disponible (padding), que es la misma
        politica que ya se usaba para los segmentos cortos.
        """
        T = self.frames_per_window
        last = num_frames - 1
        return [min(start + i * stride, last) for i in range(T)]

    def _load_window(self, video_path: str, start: int, window_len: int,
                     stride: Optional[int] = None,
                     frame_files: Optional[List[str]] = None) -> torch.Tensor:
        if frame_files is None:
            frame_files = sorted(os.listdir(video_path))
        if stride is None:
            stride = self.sample_one_each

        idx = self._window_indices(len(frame_files), start, stride)
        sequence_files = [frame_files[i] for i in idx]
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

        # `sorted(os.listdir(...))` una sola vez por segmento en lugar de una
        # por ventana: con hasta 34 ventanas por segmento el ahorro de I/O
        # es notable.
        frame_files = sorted(os.listdir(video_path))
        num_frames = len(frame_files)

        # --- Jitter temporal, sampleado UNA vez por segmento ---
        # Por segmento y no por ventana: todas las ventanas se apilan en un
        # tensor, asi que necesitan el mismo T. Ademas es lo fisicamente
        # coherente: un segmento es una grabacion continua, su frame-rate
        # no cambia a la mitad.
        stride = self.sample_one_each
        if self.temporal_stride_jitter:
            # Solo pasos que dejen entrar la ventana entera en el segmento;
            # si ninguno entra, se cae al paso base (el padding se encarga).
            T = self.frames_per_window
            validos = [s_ for s_ in self.temporal_stride_jitter
                       if (T - 1) * s_ + 1 <= num_frames] or [self.sample_one_each]
            stride = validos[int(torch.randint(len(validos), (1,)).item())]

        offset = 0
        if self.temporal_offset_jitter:
            span = (self.frames_per_window - 1) * stride + 1
            margen = max(0, num_frames - (max(starts) + span))
            if margen > 0:
                offset = int(torch.randint(min(margen, stride * 2) + 1, (1,)).item())

        windows = [
            self._load_window(video_path, s_ + offset, window_len,
                              stride=stride, frame_files=frame_files)
            for s_ in starts
        ]
        windows_tensor = torch.stack(windows, dim=0)  # (num_windows, C, T, H, W)

        # --- Augmentation a nivel CLIP ---
        # Se aplica sobre las ventanas ya apiladas y ANTES de normalizar
        # (el modulo espera rango [0, 1]). Los parametros fotometricos se
        # samplean una vez y valen para todas las ventanas del segmento.
        if self.augment is not None:
            windows_tensor = torch.stack(
                [self.augment(w, label=label) for w in windows_tensor], dim=0
            )

        if self.normalize is not None:
            windows_tensor = torch.stack(
                [self.normalize(w) for w in windows_tensor], dim=0
            )

        # (session_name, segment_name)  �til para debug/logging
        name = Path(video_path).parts[-3:-1]
        return windows_tensor, torch.tensor(label, dtype=torch.long), name


class ClipNormalize:
    """
    Normalize consciente del layout de clip.

    `torchvision.transforms.Normalize` normaliza sobre dim=-3, que en un
    tensor (C, T, H, W) es la dimension TEMPORAL, no la de canales: usarlo
    directamente mezclaria las estadisticas de RGB con las de los frames.
    Esta clase hace el broadcast sobre el eje correcto.
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float).view(-1, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float).view(-1, 1, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (C, T, H, W)
        return (clip - self.mean) / self.std


def segment_collate_fn(batch):
    """
    El DataLoader se usa con batch_size=1 (un segmento por iteraci�n): cada
    __getitem__ ya devuelve el "batch" de windows de ese segmento. Este
    collate simplemente desempaqueta la lista de un solo elemento.
    """
    return batch[0]