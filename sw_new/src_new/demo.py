"""
Demo de inferencia para la arquitectura actual del proyecto.

No usa Hydra. La configuración del modelo se carga con OmegaConf y las
rutas/modos de ejecución se pasan por argumentos de consola.

Ejemplos
--------
Video con checkpoint PyTorch:

    python demo.py \
        --config configs/config_binary_phone_balanced.yaml \
        --checkpoint ../models/binary_phone_lad13_augmentation/model_8.pth \
        --video /ruta/video.mp4 \
        --mode video

Latencia end-to-end PyTorch:

    python demo.py \
        --config configs/config_binary_phone_balanced.yaml \
        --checkpoint ../models/binary_phone_lad13_augmentation/model_8.pth \
        --video /ruta/video.mp4 \
        --mode latency \
        --samples 500

Latencia del .pte con ExecuTorch:

    python demo.py \
        --config configs/config_binary_phone_balanced.yaml \
        --pte model_binary_phone.pte \
        --mode executor_latency \
        --samples 500
"""

import argparse
import math
import time
from collections import deque
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from predictors.base_predictor import Predictor


DEFAULT_INPUT_SIZE = 112


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo de driver behavior con OmegaConf."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Ruta al YAML usado por el modelo.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pth para modos video/latency.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Ruta al video para modos video/latency.",
    )
    parser.add_argument(
        "--mode",
        default="video",
        choices=[
            "video",
            "latency",
            "executor_latency",
        ],
    )
    parser.add_argument(
        "--pte",
        default=None,
        help="Modelo ExecuTorch .pte para executor_latency.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Número de repeticiones para pruebas de latencia.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Número de inferencias de calentamiento.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda | cuda:N",
    )
    parser.add_argument(
        "--output",
        default="demo_salida.mp4",
        help="Video de salida para modo video.",
    )
    parser.add_argument(
        "--inference-every",
        type=int,
        default=1,
        help=(
            "Ejecuta una inferencia cada N frames una vez lleno "
            "el buffer. Por defecto, cada frame."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="No abrir ventana OpenCV (útil en Raspberry Pi headless).",
    )

    return parser.parse_args()


def validate_args(args):
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"No existe el config: {config_path}"
        )

    if args.samples <= 0:
        raise ValueError("--samples debe ser > 0.")

    if args.warmup < 0:
        raise ValueError("--warmup debe ser >= 0.")

    if args.inference_every <= 0:
        raise ValueError("--inference-every debe ser > 0.")

    if args.mode in ("video", "latency"):
        if args.checkpoint is None:
            raise ValueError(
                f"--checkpoint es obligatorio para mode={args.mode}."
            )
        if args.video is None:
            raise ValueError(
                f"--video es obligatorio para mode={args.mode}."
            )
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(
                f"No existe el checkpoint: {args.checkpoint}"
            )
        if not Path(args.video).exists():
            raise FileNotFoundError(
                f"No existe el video: {args.video}"
            )

    if args.mode == "executor_latency":
        if args.pte is None:
            raise ValueError(
                "--pte es obligatorio para mode=executor_latency."
            )
        if not Path(args.pte).exists():
            raise FileNotFoundError(
                f"No existe el modelo .pte: {args.pte}"
            )


def open_video(video_path: str):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"No fue posible abrir el video: {video_path}"
        )

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    return cap, fps, width, height


def draw_prediction(frame, label, confidence):
    if label is None:
        text = "loading buffer"
    else:
        text = f"{label}  {confidence:.1%}"

    cv.putText(
        frame,
        text,
        (50, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )


def run_video(
    cap,
    fps,
    width,
    height,
    predictor: Predictor,
    output_path: str,
    inference_every: int = 1,
    display: bool = True,
):
    """
    Inferencia online con ventana deslizante.

    Importante:
    el buffer conserva `sequence_length` frames CRUDOS. Predictor aplica
    después `sample_one_each`, igual que el dataset de entrenamiento.
    """
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not out.isOpened():
        raise RuntimeError(
            f"No fue posible crear el video de salida: {output_path}"
        )

    frame_buffer = deque(
        maxlen=predictor.sequence_length
    )

    last_label = None
    last_confidence = 0.0
    frames_since_inference = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append(frame)

            if len(frame_buffer) == predictor.sequence_length:
                if frames_since_inference == 0:
                    raw_window = np.stack(
                        frame_buffer,
                        axis=0,
                    )
                    (
                        last_label,
                        last_confidence,
                        _,
                    ) = predictor.predict_with_confidence(
                        raw_window
                    )

                frames_since_inference = (
                    frames_since_inference + 1
                ) % inference_every

            draw_prediction(
                frame,
                last_label,
                last_confidence,
            )
            out.write(frame)

            if display:
                cv.imshow("Driver Behavior Demo", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        out.release()

    print(f"Video guardado en: {output_path}")


def build_video_sample(
    cap,
    sequence_length: int,
) -> np.ndarray:
    """
    Obtiene una ventana cruda de `sequence_length` frames consecutivos.
    Predictor aplicará posteriormente el submuestreo temporal.
    """
    frames = []

    while cap.isOpened() and len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if len(frames) < sequence_length:
        raise RuntimeError(
            "El video no contiene suficientes frames para construir "
            f"una ventana de {sequence_length}. "
            f"Solo se obtuvieron {len(frames)}."
        )

    return np.stack(frames, axis=0)


def synchronize_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_latency_test(
    cap,
    num_samples: int,
    predictor: Predictor,
    warmup: int,
):
    """
    Latencia end-to-end PyTorch:
      preprocesamiento + transferencia al device + forward + postproceso.
    """
    print("Construyendo muestra de video...")
    raw_window = build_video_sample(
        cap,
        predictor.sequence_length,
    )

    print(f"Warm-up: {warmup} inferencias")
    for _ in range(warmup):
        predictor.predict(raw_window)

    synchronize_if_needed(predictor.device)

    print("Midiendo latencia end-to-end PyTorch...")
    start_time = time.perf_counter()

    with tqdm(total=num_samples) as pbar:
        for _ in range(num_samples):
            predictor.predict(raw_window)
            pbar.update(1)

    synchronize_if_needed(predictor.device)

    elapsed = time.perf_counter() - start_time
    return elapsed / num_samples


def executor_latency_test(
    cfg,
    pte_path: str,
    num_samples: int,
    warmup: int,
):
    """
    Latencia del método forward del modelo ExecuTorch.

    No incluye lectura de video ni preprocesamiento.
    """
    # Import lazy: la demo PyTorch no debe depender de ExecuTorch.
    from executorch.runtime import Runtime

    sequence_length = int(cfg.sequence_length)
    sample_one_each = max(1, int(cfg.sample_one_each))
    frames_per_window = max(
        1,
        math.ceil(sequence_length / sample_one_each),
    )
    input_size = int(
        cfg.get("input_size", DEFAULT_INPUT_SIZE)
    )

    input_tensor = torch.randn(
        1,
        3,
        frames_per_window,
        input_size,
        input_size,
        dtype=torch.float32,
    )

    print(
        "Input ExecuTorch: "
        f"{tuple(input_tensor.shape)}"
    )

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    for _ in range(warmup):
        method.execute([input_tensor])

    start_time = time.perf_counter()

    with tqdm(total=num_samples) as pbar:
        for _ in range(num_samples):
            output: List[torch.Tensor] = method.execute(
                [input_tensor]
            )
            pbar.update(1)

    elapsed = time.perf_counter() - start_time

    # Evita que la variable se considere innecesaria y permite detectar
    # errores de ejecución antes de reportar la latencia.
    if not output:
        raise RuntimeError(
            "ExecuTorch no devolvió ningún tensor de salida."
        )

    return elapsed / num_samples


def main():
    args = parse_args()
    validate_args(args)

    # OmegaConf solamente; no Hydra.
    cfg = OmegaConf.load(args.config)

    print("Configuración")
    print("-------------")
    print(f"Config : {args.config}")
    print(f"Mode   : {args.mode}")

    if args.mode == "executor_latency":
        avg_latency = executor_latency_test(
            cfg=cfg,
            pte_path=args.pte,
            num_samples=args.samples,
            warmup=args.warmup,
        )

        print(
            f"Latencia media ExecuTorch: "
            f"{avg_latency * 1000:.3f} ms"
        )
        print(
            f"Throughput aproximado: "
            f"{1.0 / avg_latency:.2f} inferencias/s"
        )
        return

    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        config=cfg,
        device=args.device,
    )

    cap, fps, width, height = open_video(args.video)

    try:
        if args.mode == "video":
            print("Running video demo")
            run_video(
                cap=cap,
                fps=fps,
                width=width,
                height=height,
                predictor=predictor,
                output_path=args.output,
                inference_every=args.inference_every,
                display=not args.no_display,
            )

        elif args.mode == "latency":
            avg_latency = run_latency_test(
                cap=cap,
                num_samples=args.samples,
                predictor=predictor,
                warmup=args.warmup,
            )

            print(
                f"Latencia media PyTorch end-to-end: "
                f"{avg_latency * 1000:.3f} ms"
            )
            print(
                f"Throughput aproximado: "
                f"{1.0 / avg_latency:.2f} inferencias/s"
            )

    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
