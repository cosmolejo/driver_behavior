#!/usr/bin/env python3
"""
Busca un posible offset entre los JPEG extraídos de un segmento y el MP4 original.

Ejemplo:
    python compare_jpg_mp4_offset.py \
        --video /ruta/gC_11_s2_2019-03-04T09_25_33+01_00_rgb_body_240.mp4 \
        --jpg-dir /ruta/seg_008514-008546_unsafe/body \
        --expected-start 8514 \
        --radius 300 \
        --samples 16

La comparación usa NMAE (Normalized Mean Absolute Error) sobre imágenes
en escala de grises. Un valor menor significa mayor similitud.

El script:
  1. carga los JPEG del segmento en orden natural;
  2. toma varias posiciones distribuidas a lo largo del segmento;
  3. carga del MP4 un rango alrededor del frame esperado;
  4. prueba todos los posibles frames iniciales dentro de ±radius;
  5. reporta el mejor offset y los mejores candidatos.
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path: Path):
    """Orden natural: 00001, 00002, ..., 00010."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def preprocess(img, target_size=None, blur=True):
    if img is None:
        raise ValueError("Imagen vacía.")

    if target_size is not None:
        img = cv2.resize(
            img,
            target_size,
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray.astype(np.float32)


def nmae(a, b):
    """Mean Absolute Error normalizado a [0, 1]."""
    return float(np.mean(np.abs(a - b)) / 255.0)


def load_jpgs(jpg_dir: str):
    p = Path(jpg_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {jpg_dir}")

    files = sorted(
        [x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )

    if not files:
        raise RuntimeError(f"No se encontraron imágenes en: {jpg_dir}")

    images = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"No se pudo leer: {f}")
        images.append(img)

    return files, images


def choose_sample_indices(n, samples):
    if samples <= 0 or samples >= n:
        return list(range(n))

    idxs = np.linspace(0, n - 1, samples)
    idxs = np.unique(np.rint(idxs).astype(int))
    return idxs.tolist()


def load_video_range(video_path: str, start: int, end: int):
    """
    Carga secuencialmente frames [start, end] inclusive.

    Se hace un único seek al comienzo y luego lectura secuencial para
    reducir problemas de precisión y coste al buscar frame por frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if start < 0:
        start = 0
    if end >= total:
        end = total - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = {}
    current = start

    while current <= end:
        ok, frame = cap.read()
        if not ok:
            break
        frames[current] = frame
        current += 1

    cap.release()

    return frames, total, fps


def candidate_score(
    candidate_start,
    sample_indices,
    jpg_processed,
    video_processed,
):
    scores = []

    for local_idx in sample_indices:
        video_idx = candidate_start + local_idx

        if video_idx not in video_processed:
            return None

        score = nmae(
            jpg_processed[local_idx],
            video_processed[video_idx],
        )
        scores.append(score)

    return float(np.mean(scores)), scores


def main():
    parser = argparse.ArgumentParser(
        description="Busca offset JPEG ↔ MP4 para un segmento."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--jpg-dir", required=True)
    parser.add_argument(
        "--expected-start",
        type=int,
        required=True,
        help="Frame del MP4 donde debería comenzar el segmento.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=300,
        help="Buscar desde expected_start-radius hasta +radius.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help=(
            "Número de JPEG distribuidos a lo largo del segmento que "
            "se usan para calcular cada score. Usa 0 para usar todos."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Número de mejores candidatos a mostrar.",
    )
    parser.add_argument(
        "--no-blur",
        action="store_true",
        help="Desactiva GaussianBlur antes de comparar.",
    )

    args = parser.parse_args()

    jpg_files, jpg_images = load_jpgs(args.jpg_dir)
    n_jpg = len(jpg_images)

    sample_indices = choose_sample_indices(n_jpg, args.samples)

    # Todas las imágenes se comparan a la resolución del primer JPEG.
    h, w = jpg_images[0].shape[:2]
    target_size = (w, h)

    jpg_processed = [
        preprocess(
            img,
            target_size=target_size,
            blur=not args.no_blur,
        )
        for img in jpg_images
    ]

    search_start = max(0, args.expected_start - args.radius)

    # Hay que cargar suficiente video para probar también el último
    # candidato y todos los JPEG del segmento.
    search_end = (
        args.expected_start
        + args.radius
        + n_jpg
        - 1
    )

    video_frames, total_frames, fps = load_video_range(
        args.video,
        search_start,
        search_end,
    )

    if not video_frames:
        raise RuntimeError("No se pudieron cargar frames del MP4.")

    video_processed = {
        idx: preprocess(
            frame,
            target_size=target_size,
            blur=not args.no_blur,
        )
        for idx, frame in video_frames.items()
    }

    candidates = []

    candidate_min = max(0, args.expected_start - args.radius)
    candidate_max = min(
        args.expected_start + args.radius,
        total_frames - n_jpg,
    )

    for candidate_start in range(candidate_min, candidate_max + 1):
        result = candidate_score(
            candidate_start,
            sample_indices,
            jpg_processed,
            video_processed,
        )
        if result is None:
            continue

        mean_score, per_frame = result
        candidates.append(
            {
                "start": candidate_start,
                "offset": candidate_start - args.expected_start,
                "score": mean_score,
                "per_frame": per_frame,
            }
        )

    if not candidates:
        raise RuntimeError("No fue posible evaluar candidatos.")

    candidates.sort(key=lambda x: x["score"])
    best = candidates[0]

    expected = next(
        (x for x in candidates if x["start"] == args.expected_start),
        None,
    )

    print()
    print("=" * 72)
    print("COMPARACION JPEG ↔ MP4")
    print("=" * 72)
    print(f"Video                 : {args.video}")
    print(f"JPEG dir              : {args.jpg_dir}")
    print(f"JPEG encontrados      : {n_jpg}")
    print(f"Resolucion comparada  : {w}x{h}")
    print(f"FPS MP4               : {fps:.6f}")
    print(f"Frames MP4            : {total_frames}")
    print(f"Inicio esperado       : {args.expected_start}")
    print(f"Radio de busqueda     : ±{args.radius}")
    print(f"JPEG usados por score : {len(sample_indices)}")
    print(f"Indices locales       : {sample_indices}")
    print()

    if expected is not None:
        print(
            f"Score inicio esperado : {expected['score']:.8f} "
            f"(frame {args.expected_start})"
        )

    print(
        f"MEJOR inicio          : {best['start']}"
    )
    print(
        f"MEJOR offset          : {best['offset']:+d} frames"
    )
    print(
        f"MEJOR score NMAE      : {best['score']:.8f}"
    )

    if fps > 0:
        print(
            f"Offset temporal       : {best['offset'] / fps:+.6f} s"
        )

    print()
    print(f"Top {min(args.top, len(candidates))} candidatos:")
    print("-" * 55)
    print(f"{'start':>10} {'offset':>10} {'NMAE':>16}")
    for item in candidates[:args.top]:
        print(
            f"{item['start']:>10d} "
            f"{item['offset']:>+10d} "
            f"{item['score']:>16.8f}"
        )

    print()
    print("Scores individuales del mejor candidato:")
    print("-" * 72)
    print(f"{'JPEG':>8} {'archivo':>24} {'MP4 frame':>12} {'NMAE':>12}")

    for local_idx, score in zip(
        sample_indices,
        best["per_frame"],
    ):
        print(
            f"{local_idx:>8d} "
            f"{jpg_files[local_idx].name:>24} "
            f"{best['start'] + local_idx:>12d} "
            f"{score:>12.8f}"
        )

    print()
    if best["offset"] == 0:
        print(
            "RESULTADO: el mejor alineamiento coincide con el frame "
            "esperado. No se detecta offset dentro del rango buscado."
        )
    elif abs(best["offset"]) <= 1:
        print(
            "RESULTADO: se detecta un desplazamiento de solo 1 frame. "
            "Puede corresponder a una diferencia de indexacion 0-based/"
            "1-based."
        )
    else:
        print(
            "RESULTADO: se detecta un offset respecto al inicio esperado. "
            "Conviene revisar cómo se generaron los índices de segmento."
        )


if __name__ == "__main__":
    main()
