#!/usr/bin/env python3
"""
Comparación DEFINITIVA de las rutas de inferencia JPEG(TEST) vs MP4(DEMO).

Para una ventana concreta:
  A) reconstruye el tensor como lo hace SegmentDataset/eval_model desde JPEG;
  B) reconstruye el mismo tensor pasando esos JPEG por ClipPreprocessor;
  C) reconstruye el tensor desde los frames equivalentes del MP4;
  D) opcionalmente vuelve a codificar los frames MP4 a JPEG antes de preprocesar.

Todos los tensores se pasan por EL MISMO objeto de modelo cargado una sola vez
desde el checkpoint, por lo que cualquier diferencia de logits queda atribuida
a la entrada y no a otra carga del modelo.

Ejemplo para el segmento positivo de una sola ventana:

python compare_inference_paths.py \
  --config configs/config_binary_phone_balanced.yaml \
  --checkpoint ../models/binary_phone_lad13_augmentation/model_best_segment.pth \
  --video /home/agomez/Documents/demo_vids/gC_11_s2_2019-03-04T09_25_33+01_00_rgb_body_240.mp4 \
  --jpg-dir /home/agomez/Documents/dmd_user_split/TEST/unsafe/gC_11_s2_2019-03-04T09_25_33+01_00_rgb_ann_distraction/seg_008514-008546_unsafe/body \
  --start 8514 \
  --device cpu
"""

import argparse
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import transforms

from predictors.base_predictor import Predictor


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_jpeg_frames(jpg_dir: str):
    folder = Path(jpg_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"No existe el directorio JPEG: {folder}")

    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )
    if not files:
        raise RuntimeError(f"No se encontraron imágenes en: {folder}")

    frames = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"No se pudo leer: {p}")
        frames.append(img)

    return files, frames


def load_mp4_window(video_path: str, start: int, length: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if start < 0 or start + length > total:
        cap.release()
        raise ValueError(
            f"Ventana MP4 fuera de rango: start={start}, length={length}, "
            f"total_frames={total}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    for i in range(length):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(
                f"No se pudo leer el frame {start + i} del MP4."
            )
        frames.append(frame)

    cap.release()
    return frames, total, fps


def build_dataset_tensor(
    jpeg_frames,
    sequence_length: int,
    sample_one_each: int,
    input_size: int,
):
    """
    Reproduce la ventana start=0 de SegmentDataset:

      cv2.imread -> BGR->RGB -> ToPILImage -> Resize -> ToTensor -> Normalize

    Los índices temporales son:
      0, sample_one_each, 2*sample_one_each, ...
    """
    frames_per_window = max(
        1, math.ceil(sequence_length / max(1, sample_one_each))
    )
    last = len(jpeg_frames) - 1
    indices = [
        min(i * sample_one_each, last)
        for i in range(frames_per_window)
    ]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    processed = []
    for idx in indices:
        img_rgb = cv2.cvtColor(jpeg_frames[idx], cv2.COLOR_BGR2RGB)
        processed.append(transform(img_rgb))

    clip = torch.stack(processed, dim=1)  # C,T,H,W
    return clip.unsqueeze(0).contiguous(), indices


def reencode_jpeg(frames, quality: int):
    """
    Simula el paso MP4 frame -> JPEG -> cv2.imread sin escribir a disco.
    """
    out = []
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    for i, frame in enumerate(frames):
        ok, encoded = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise RuntimeError(f"No se pudo codificar a JPEG el frame local {i}")

        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError(f"No se pudo decodificar JPEG del frame local {i}")

        out.append(decoded)

    return out


def tensor_stats(name, t):
    x = t.detach().cpu().float()
    print(
        f"{name:<24} shape={tuple(x.shape)}  "
        f"min={x.min().item():+.6f}  "
        f"max={x.max().item():+.6f}  "
        f"mean={x.mean().item():+.6f}  "
        f"std={x.std().item():.6f}"
    )


def compare_tensors(name, a, b):
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()

    if a.shape != b.shape:
        print(f"{name}: SHAPE DISTINTA {tuple(a.shape)} vs {tuple(b.shape)}")
        return

    diff = a - b
    abs_diff = diff.abs()

    mae = abs_diff.mean().item()
    rmse = torch.sqrt((diff * diff).mean()).item()
    max_abs = abs_diff.max().item()

    # Eje C = 1 en B,C,T,H,W
    channel_mae = abs_diff.mean(dim=(0, 2, 3, 4)).tolist()

    af = a.flatten().double()
    bf = b.flatten().double()
    cosine = torch.nn.functional.cosine_similarity(
        af.unsqueeze(0), bf.unsqueeze(0), dim=1
    ).item()

    print(name)
    print(f"  MAE tensor       : {mae:.10f}")
    print(f"  RMSE tensor      : {rmse:.10f}")
    print(f"  max |diff|       : {max_abs:.10f}")
    print(
        "  MAE por canal   : "
        f"R={channel_mae[0]:.10f}, "
        f"G={channel_mae[1]:.10f}, "
        f"B={channel_mae[2]:.10f}"
    )
    print(f"  cosine similarity: {cosine:.12f}")


def compare_raw_selected(jpeg_frames, mp4_frames, indices):
    maes = []
    maxes = []

    for idx in indices:
        a = jpeg_frames[idx].astype(np.float32)
        b = mp4_frames[idx].astype(np.float32)
        d = np.abs(a - b)
        maes.append(float(d.mean()))
        maxes.append(float(d.max()))

    print("JPEG vs MP4 en pixels CRUDOS seleccionados")
    print(f"  MAE medio [0,255]: {np.mean(maes):.8f}")
    print(f"  MAE min/max      : {np.min(maes):.8f} / {np.max(maes):.8f}")
    print(f"  max diff pixel   : {np.max(maxes):.1f}")


@torch.no_grad()
def infer_tensor(predictor: Predictor, tensor: torch.Tensor):
    x = tensor.to(predictor.device, dtype=torch.float32)
    logits = predictor.model(x)
    probs = torch.softmax(logits, dim=1)

    return (
        logits[0].detach().cpu().numpy(),
        probs[0].detach().cpu().numpy(),
    )


def print_prediction(name, predictor, tensor):
    logits, probs = infer_tensor(predictor, tensor)
    names = list(predictor.class_names)

    pred = int(np.argmax(probs))
    label = names[pred] if pred < len(names) else f"class_{pred}"

    print(name)
    print("  logits : " + "  ".join(
        f"{n}={float(v):+.8f}"
        for n, v in zip(names, logits)
    ))
    print("  probs  : " + "  ".join(
        f"{n}={float(v):.8f}"
        for n, v in zip(names, probs)
    ))
    print(f"  pred   : {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--jpg-dir", required=True)
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Frame absoluto del MP4 correspondiente al primer JPEG del segmento.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu | cuda | cuda:N | auto",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="Calidad usada para la prueba MP4->JPEG->modelo. Default 95.",
    )
    parser.add_argument(
        "--save-tensors",
        default=None,
        help=(
            "Prefijo opcional para guardar los tensores .pt, por ejemplo "
            "'debug_8514'."
        ),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        config=cfg,
        device=args.device,
    )

    sequence_length = predictor.sequence_length
    sample_one_each = predictor.sample_one_each
    input_size = predictor.input_size

    jpeg_files, jpeg_frames = load_jpeg_frames(args.jpg_dir)

    if len(jpeg_frames) < sequence_length:
        raise RuntimeError(
            f"El segmento solo tiene {len(jpeg_frames)} JPEG, pero esta prueba "
            f"requiere al menos sequence_length={sequence_length}."
        )

    # Para la primera/única ventana del segmento, la ruta streaming recibe
    # exactamente los primeros sequence_length frames crudos.
    jpeg_raw_window = jpeg_frames[:sequence_length]

    mp4_frames, total_frames, fps = load_mp4_window(
        args.video,
        args.start,
        sequence_length,
    )

    # A) Ruta exacta de SegmentDataset/eval_model.
    t_dataset, temporal_indices = build_dataset_tensor(
        jpeg_frames,
        sequence_length=sequence_length,
        sample_one_each=sample_one_each,
        input_size=input_size,
    )

    # B) Mismos JPEG pero procesados por ClipPreprocessor usado por demo.
    t_jpeg_predictor = predictor.preprocess(
        np.stack(jpeg_raw_window, axis=0)
    )

    # C) MP4 directo por ClipPreprocessor.
    t_mp4 = predictor.preprocess(
        np.stack(mp4_frames, axis=0)
    )

    # D) MP4 re-codificado a JPEG para probar el efecto del codec.
    mp4_as_jpeg_frames = reencode_jpeg(
        mp4_frames,
        quality=args.jpeg_quality,
    )
    t_mp4_as_jpeg = predictor.preprocess(
        np.stack(mp4_as_jpeg_frames, axis=0)
    )

    print()
    print("=" * 80)
    print("VENTANA COMPARADA")
    print("=" * 80)
    print(f"Video              : {args.video}")
    print(f"JPEG dir           : {args.jpg_dir}")
    print(f"Primer JPEG        : {jpeg_files[0].name}")
    print(f"JPEG disponibles   : {len(jpeg_files)}")
    print(f"MP4 start          : {args.start}")
    print(f"MP4 frames usados  : {args.start} .. {args.start + sequence_length - 1}")
    print(f"MP4 total frames   : {total_frames}")
    print(f"FPS                 : {fps:.6f}")
    print(f"sequence_length     : {sequence_length}")
    print(f"sample_one_each     : {sample_one_each}")
    print(f"indices seleccionados locales: {temporal_indices}")
    print(
        "frames absolutos modelo    : "
        + str([args.start + i for i in temporal_indices])
    )

    print()
    print("=" * 80)
    print("ESTADISTICAS DE LOS TENSORES")
    print("=" * 80)
    tensor_stats("A JPEG / Dataset", t_dataset)
    tensor_stats("B JPEG / Predictor", t_jpeg_predictor)
    tensor_stats("C MP4 / Predictor", t_mp4)
    tensor_stats("D MP4->JPEG / Predict", t_mp4_as_jpeg)

    print()
    print("=" * 80)
    print("DIFERENCIAS ENTRE RUTAS")
    print("=" * 80)
    compare_tensors(
        "A Dataset JPEG vs B Predictor JPEG",
        t_dataset,
        t_jpeg_predictor,
    )
    print()
    compare_tensors(
        "B Predictor JPEG vs C Predictor MP4",
        t_jpeg_predictor,
        t_mp4,
    )
    print()
    compare_tensors(
        "B Predictor JPEG vs D MP4->JPEG",
        t_jpeg_predictor,
        t_mp4_as_jpeg,
    )
    print()
    compare_raw_selected(
        jpeg_frames,
        mp4_frames,
        temporal_indices,
    )

    print()
    print("=" * 80)
    print("MISMO MODELO, CUATRO ENTRADAS")
    print("=" * 80)
    print_prediction("A) JPEG como SegmentDataset", predictor, t_dataset)
    print()
    print_prediction("B) JPEG via ClipPreprocessor", predictor, t_jpeg_predictor)
    print()
    print_prediction("C) MP4 via ClipPreprocessor", predictor, t_mp4)
    print()
    print_prediction(
        f"D) MP4 re-JPEG q={args.jpeg_quality}",
        predictor,
        t_mp4_as_jpeg,
    )

    if args.save_tensors:
        prefix = Path(args.save_tensors)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        paths = {
            "dataset_jpeg": t_dataset,
            "predictor_jpeg": t_jpeg_predictor,
            "predictor_mp4": t_mp4,
            "predictor_mp4_rejpeg": t_mp4_as_jpeg,
        }
        for suffix, tensor in paths.items():
            path = Path(f"{prefix}_{suffix}.pt")
            torch.save(tensor.cpu(), path)
            print(f"Guardado: {path}")

    print()
    print("=" * 80)
    print("COMO INTERPRETAR EL RESULTADO")
    print("=" * 80)
    print(
        "1) A y B deben ser practicamente identicos. Si no lo son, "
        "SegmentDataset y ClipPreprocessor NO reproducen el mismo preprocessing."
    )
    print(
        "2) Si A/B dan phone (~0.83 esperado por el CSV) pero C da safe, "
        "la divergencia esta en JPEG vs MP4."
    )
    print(
        "3) Si D vuelve a parecerse a A/B, el modelo es sensible a la "
        "codificacion JPEG usada para construir TEST."
    )
    print(
        "4) Si A tambien da safe, entonces hay una discrepancia entre "
        "eval_model_segment_debug y Predictor/model loading, no en el video."
    )


if __name__ == "__main__":
    main()
