import os
from pathlib import Path
import numpy as np
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
        balance_classes: bool = False,
        balance_unit: Optional[str] = None,
        windows_per_segment: Optional[int] = None,
        balance_seed: int = 42,
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

        balance_unit / windows_per_segment:
            Modo de balanceo de clases:

              None       sin balanceo
              "segment"  iguala SEGMENTOS por clase (submuestrea la
                         mayoritaria round-robin entre sujetos)
              "window"   iguala VENTANAS por clase conservando TODOS los
                         segmentos: recorta ventanas dentro de los
                         segmentos de la clase mayoritaria
              "both"     iguala ambas. Requiere windows_per_segment.

            AVISO sobre "window": con window bagging la perdida se promedia
            dentro de cada segmento, asi que cada segmento aporta una sola
            contribucion de gradiente sin importar cuantas ventanas tenga.
            Igualar ventanas NO cambia el peso efectivo de las clases en el
            entrenamiento; eso lo determina el conteo de SEGMENTOS. Afecta
            la metrica a nivel ventana, el computo y la varianza de la
            estimacion de la perdida por segmento.

        balance_classes (retrocompatible) / windows_per_segment:
            Balanceo de clases. Las dos unidades de conteo estan en tension
            y no se pueden igualar a la vez sin tocar las ventanas:

                un segmento de safe_drive da ~21 ventanas
                un segmento de phone      da ~70 ventanas

            Igualar SEGMENTOS deja las ventanas 0.30:1; igualar VENTANAS
            deja los segmentos 3.26:1.

            La solucion es `windows_per_segment=K`: se descartan los
            segmentos con menos de K ventanas y de los que quedan se toman
            EXACTAMENTE K, equiespaciadas a lo largo del segmento (no las K
            primeras, para no sesgar hacia el inicio). Con la misma cantidad
            de segmentos por clase, ambas unidades quedan balanceadas de
            forma exacta.

            `balance_classes=True` sin K solo iguala los segmentos.

            El submuestreo de la clase mayoritaria es ROUND-ROBIN entre
            sujetos: se toma un segmento de cada sujeto por turno hasta
            llegar al objetivo. Asi ningun sujeto desaparece y se acota el
            peso de los que aportan mas segmentos.

            Aplicar esto a VALIDACION o TEST cambia el piso trivial y hace
            que la metrica deje de reflejar la distribucion real de
            despliegue: se usa solo en entrenamiento.

        En modo "fine" quedan disponibles:
            self.class_to_idx  -> {actividad: indice}, 9 entradas
            self.label_groups  -> lista de 9 ints, indice fino -> indice macro
                                  (se pasa a train_pipeline para agregar en
                                  validacion)
        """
        if label_mode != "macro" and partition_report is None:
            raise ValueError(
                f"label_mode={label_mode!r} requiere `partition_report`: la "
                "actividad de grano fino no esta en el arbol de directorios."
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
            self.class_names = sorted(self.macro_to_idx, key=self.macro_to_idx.get)
            self.label_groups = None
            self._activity_lookup = None
        else:
            from fine_labels import (
                build_label_groups, get_scheme, load_activity_lookup,
            )
            amap, names = get_scheme(label_mode, partition_report)
            self.class_to_idx = amap
            self.class_names = names
            self._activity_lookup = load_activity_lookup(partition_report)
            # `label_groups` solo tiene sentido en el esquema "fine": las 9
            # actividades se agregan a las 3 macro-clases en inferencia. En
            # los esquemas con mapa propio (binario, ternario limpio) las
            # clases del modelo YA son las finales; no hay que agregar nada.
            self.label_groups = (
                build_label_groups(partition_report) if label_mode == "fine" else None
            )

        # --- Submuestreo por sujeto ---
        self.subject_subset = subject_subset
        self._subject_lookup = None
        self.selected_subjects = None
        # El lookup de sujetos tambien lo necesita el balanceo round-robin
        if partition_report is not None:
            from fine_labels import load_subject_lookup
            self._subject_lookup = load_subject_lookup(partition_report)

        if subject_subset is not None:
            if partition_report is None:
                raise ValueError(
                    "subject_subset requiere `partition_report`: la identidad "
                    "del sujeto no esta en el arbol de directorios."
                )
            from fine_labels import select_subjects
            split_name = Path(root_dir).name.upper()
            self.selected_subjects = set(
                select_subjects(partition_report, subject_subset,
                                split=split_name, seed=subject_subset_seed)
            )
            print(f"[{split_name}] submuestreo: {len(self.selected_subjects)} sujetos "
                  f"-> {sorted(self.selected_subjects)}")

        # Resolucion del modo de balanceo (retrocompatible con
        # balance_classes=True, que equivale a "segment" o a "both" segun
        # se haya fijado windows_per_segment).
        if balance_unit is None and balance_classes:
            balance_unit = "both" if windows_per_segment is not None else "segment"
        if balance_unit not in (None, "segment", "window", "both"):
            raise ValueError(
                f"balance_unit debe ser 'segment', 'window', 'both' o None, "
                f"no {balance_unit!r}"
            )
        if balance_unit == "both" and windows_per_segment is None:
            raise ValueError(
                "balance_unit='both' requiere windows_per_segment: es la unica "
                "forma de igualar segmentos y ventanas a la vez."
            )
        self.balance_unit = balance_unit
        self.balance_classes = balance_unit is not None
        self.windows_per_segment = windows_per_segment
        self.balance_seed = balance_seed

        self.unmatched = []   # segmentos sin entrada en el CSV (solo modo fine)

        self._build_index()

        if self.balance_unit is not None or windows_per_segment is not None:
            self._balance_index(root_dir)

        if label_mode != "macro":
            if self.unmatched:
                print(
                    f"AVISO: {len(self.unmatched)} segmentos de {root_dir} no "
                    f"aparecen en el partition_report y quedaron FUERA del "
                    f"dataset. Ejemplos: {self.unmatched[:3]}"
                )
            conteo = {}
            for _, lab, _, _ in self.segments:
                conteo[lab] = conteo.get(lab, 0) + 1
            print(f"[{Path(root_dir).name}] esquema {label_mode!r}: "
                  f"{len(self.segments)} segmentos, {len(self.class_names)} clases")
            for i in sorted(conteo):
                print(f"    {i}  {self.class_names[i]:<24}{conteo[i]:>6}")

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
                    video_path = os.path.join(session_path, video_name, "body")
                    if not os.path.isdir(video_path):
                        continue

                    # Filtro por sujeto (curva de aprendizaje)
                    if self.selected_subjects is not None:
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
                        # Actividad fuera del esquema -> el segmento se EXCLUYE
                        # (p. ej. `radio` en el esquema binario de telefono).
                        if activity not in self.class_to_idx:
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

    def _submuestrear_ventanas(self, segs, objetivo, rnd):
        """
        Reduce el total de ventanas de una clase hasta `objetivo`,
        CONSERVANDO todos los segmentos.

        El recorte se reparte proporcionalmente entre los segmentos y las
        ventanas que quedan se toman equiespaciadas, de modo que cada
        segmento conserve cobertura de principio a fin. Ningun segmento
        baja de 1 ventana.
        """
        total = sum(len(st) for _, _, st, _ in segs)
        if total <= objetivo:
            return segs

        f = objetivo / total
        recortados, cuotas = [], []
        for video_path, label, starts, wl in segs:
            k = max(1, int(round(len(starts) * f)))
            k = min(k, len(starts))
            cuotas.append(k)

        # Ajuste fino para llegar al objetivo exacto: se quita (o agrega) de
        # los segmentos mas largos, que son los que menos pierden en
        # cobertura relativa.
        orden = sorted(range(len(segs)), key=lambda i: -len(segs[i][2]))
        while sum(cuotas) > objetivo:
            avance = False
            for i in orden:
                if sum(cuotas) <= objetivo:
                    break
                if cuotas[i] > 1:
                    cuotas[i] -= 1; avance = True
            if not avance:
                break
        while sum(cuotas) < objetivo:
            avance = False
            for i in orden:
                if sum(cuotas) >= objetivo:
                    break
                if cuotas[i] < len(segs[i][2]):
                    cuotas[i] += 1; avance = True
            if not avance:
                break

        for (video_path, label, starts, wl), k in zip(segs, cuotas):
            if k >= len(starts):
                recortados.append((video_path, label, starts, wl))
            else:
                idx = np.linspace(0, len(starts) - 1, k).round().astype(int)
                nuevos = [starts[i] for i in sorted(set(idx.tolist()))]
                recortados.append((video_path, label, nuevos, wl))
        return recortados

    def _balance_index(self, root_dir: str):
        """
        Aplica el balanceo segun `balance_unit`. Ver __init__.

        NOTA IMPORTANTE sobre el modo "window": con window bagging la
        perdida se promedia DENTRO de cada segmento, asi que cada segmento
        aporta una contribucion de gradiente sin importar cuantas ventanas
        tenga. Igualar ventanas NO cambia el peso efectivo de las clases en
        el entrenamiento -eso lo determina el conteo de SEGMENTOS-. Afecta
        la metrica a nivel ventana, el tiempo de computo y la varianza de
        la estimacion de la perdida por segmento.
        """
        import random as _random

        rnd = _random.Random(self.balance_seed)
        K = self.windows_per_segment

        # --- 1. Tope de ventanas por segmento (SOLO modo "both") ---
        # En modo "window" se deben conservar todos los segmentos; por eso
        # windows_per_segment no puede recortar/descartar segmentos aqui.
        if self.balance_unit == "both":
            recortados, descartados = [], 0
            for video_path, label, starts, window_len in self.segments:
                if len(starts) < K:
                    descartados += 1
                    continue
                idx = np.linspace(0, len(starts) - 1, K).round().astype(int)
                nuevos = [starts[i] for i in sorted(set(idx.tolist()))]
                recortados.append((video_path, label, nuevos, window_len))
            self.segments = recortados
            print(f"  tope de {K} ventanas/segmento: {descartados} segmentos "
                  f"descartados por ser demasiado cortos")

        if self.balance_unit is None:
            return

        por_clase = {}
        for seg in self.segments:
            por_clase.setdefault(seg[1], []).append(seg)

        # --- 2a. Igualar SEGMENTOS (modos "segment" y "both") ---
        if self.balance_unit in ("segment", "both"):
            objetivo = min(len(v) for v in por_clase.values())
            nuevas = {}
            for label, segs in sorted(por_clase.items()):
                if len(segs) <= objetivo:
                    nuevas[label] = segs
                    continue

                # Round-robin entre sujetos: ninguno desaparece y se acota
                # el peso de los que aportan mas segmentos.
                por_sujeto = {}
                for seg in segs:
                    if self._subject_lookup is not None:
                        partes = Path(seg[0]).parts
                        subj = self._subject_lookup.get((partes[-3], partes[-2]), "?")
                    else:
                        subj = "?"
                    por_sujeto.setdefault(subj, []).append(seg)
                for k in por_sujeto:
                    rnd.shuffle(por_sujeto[k])

                sujetos = sorted(por_sujeto)
                elegidos, i = [], 0
                while len(elegidos) < objetivo:
                    avance = False
                    for sj in sujetos:
                        if len(elegidos) >= objetivo:
                            break
                        if i < len(por_sujeto[sj]):
                            elegidos.append(por_sujeto[sj][i]); avance = True
                    if not avance:
                        break
                    i += 1
                nuevas[label] = elegidos
            por_clase = nuevas

        # --- 2b. Igualar VENTANAS (modos "window" y "both") ---
        if self.balance_unit in ("window", "both"):
            objetivo_w = min(sum(len(st) for _, _, st, _ in v)
                             for v in por_clase.values())
            por_clase = {
                label: self._submuestrear_ventanas(segs, objetivo_w, rnd)
                for label, segs in por_clase.items()
            }

        self.segments = [seg for segs in por_clase.values() for seg in segs]

        # --- Resumen ---
        conteo, ventanas = {}, {}
        for _, label, starts, _ in self.segments:
            conteo[label] = conteo.get(label, 0) + 1
            ventanas[label] = ventanas.get(label, 0) + len(starts)
        nombres = getattr(self, "class_names", None)
        print(f"  balanceo '{self.balance_unit}' -> {len(self.segments)} segmentos")
        for i in sorted(conteo):
            n = nombres[i] if nombres else str(i)
            print(f"     {i}  {n:<20}{conteo[i]:>6} seg  {ventanas[i]:>7} ventanas")

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

        # (session_name, segment_name)  �til para debug/logging
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