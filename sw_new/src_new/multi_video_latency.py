
from collections import deque
from pathlib import Path
import csv
import time

import cv2 as cv
import numpy as np


def summarize_latencies(latencies_s):
    arr = np.asarray(latencies_s, dtype=np.float64)

    if arr.size == 0:
        return {
            "n": 0,
            "mean_ms": float("nan"),
            "median_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "infer_fps": float("nan"),
        }

    mean_s = float(arr.mean())

    return {
        "n": int(arr.size),
        "mean_ms": mean_s * 1000.0,
        "median_ms": float(np.median(arr)) * 1000.0,
        "p95_ms": float(np.percentile(arr, 95)) * 1000.0,
        "p99_ms": float(np.percentile(arr, 99)) * 1000.0,
        "min_ms": float(arr.min()) * 1000.0,
        "max_ms": float(arr.max()) * 1000.0,
        "infer_fps": (1.0 / mean_s) if mean_s > 0 else float("inf"),
    }


def get_first_full_window(video_path, sequence_length):
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No fue posible abrir: {video_path}")

    frames = []
    try:
        while len(frames) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) < sequence_length:
        return None

    return np.stack(frames, axis=0)


def warmup_predictor(predictor, video_paths, warmup):
    if warmup <= 0:
        return

    sample = None
    for video_path in video_paths:
        sample = get_first_full_window(video_path, predictor.sequence_length)
        if sample is not None:
            break

    if sample is None:
        raise RuntimeError(
            "Ningún video contiene suficientes frames para "
            f"una ventana de {predictor.sequence_length}."
        )

    print(f"Warm-up: {warmup} inferencias")
    for _ in range(warmup):
        predictor.predict(sample)


def run_video_latency(video_path, predictor, inference_every=1):
    """
    Recorre un video completo y mide la latencia de cada inferencia real.
    La lectura/decodificación del video ocurre secuencialmente.
    """
    video_path = Path(video_path)
    cap = cv.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"No fue posible abrir: {video_path}")

    fps_source = float(cap.get(cv.CAP_PROP_FPS))
    total_frames_reported = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    frame_buffer = deque(maxlen=predictor.sequence_length)
    latencies = []
    decoded_frames = 0
    inference_counter = 0

    wall_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            decoded_frames += 1
            frame_buffer.append(frame)

            if len(frame_buffer) < predictor.sequence_length:
                continue

            if inference_counter % inference_every == 0:
                raw_window = np.stack(frame_buffer, axis=0)

                t0 = time.perf_counter()
                predictor.predict(raw_window)
                latencies.append(time.perf_counter() - t0)

            inference_counter += 1
    finally:
        cap.release()

    wall_time_s = time.perf_counter() - wall_start
    stats = summarize_latencies(latencies)

    source_duration_s = (
        decoded_frames / fps_source if fps_source > 0 else float("nan")
    )
    processing_fps = (
        decoded_frames / wall_time_s if wall_time_s > 0 else float("inf")
    )
    realtime_factor = (
        wall_time_s / source_duration_s
        if source_duration_s > 0
        else float("nan")
    )

    return {
        "video": str(video_path),
        "source_fps": fps_source,
        "frames_reported": total_frames_reported,
        "frames_decoded": decoded_frames,
        "source_duration_s": source_duration_s,
        "wall_time_s": wall_time_s,
        "processing_fps": processing_fps,
        "realtime_factor": realtime_factor,
        **stats,
        "_latencies_s": latencies,
    }


def run_multiple_video_latency(
    video_paths,
    predictor,
    inference_every=1,
    warmup=10,
    csv_path=None,
):
    """
    Evalúa videos completos y agrega estadísticas por video y globales.
    Funciona tanto con Predictor (PyTorch) como con ExecuTorchPredictor.
    """
    video_paths = [Path(p) for p in video_paths]

    missing = [str(p) for p in video_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "No existen estos videos:\n" + "\n".join(missing)
        )

    if inference_every <= 0:
        raise ValueError("inference_every debe ser > 0.")

    warmup_predictor(predictor, video_paths, warmup)

    rows = []
    global_latencies = []

    print("\nMidiendo videos completos")
    print("=========================")

    for i, video_path in enumerate(video_paths, start=1):
        print(f"[{i}/{len(video_paths)}] {video_path}")

        result = run_video_latency(
            video_path=video_path,
            predictor=predictor,
            inference_every=inference_every,
        )

        global_latencies.extend(result.pop("_latencies_s"))
        rows.append(result)

        print(
            f"  inferencias : {result['n']}\n"
            f"  media       : {result['mean_ms']:.3f} ms\n"
            f"  mediana     : {result['median_ms']:.3f} ms\n"
            f"  p95         : {result['p95_ms']:.3f} ms\n"
            f"  p99         : {result['p99_ms']:.3f} ms\n"
            f"  infer FPS   : {result['infer_fps']:.2f}\n"
            f"  proc. FPS   : {result['processing_fps']:.2f}\n"
            f"  RT factor   : {result['realtime_factor']:.3f}x"
        )

    global_stats = summarize_latencies(global_latencies)

    total_wall_s = sum(row["wall_time_s"] for row in rows)
    total_frames = sum(row["frames_decoded"] for row in rows)
    total_source_duration_s = sum(
        row["source_duration_s"]
        for row in rows
        if np.isfinite(row["source_duration_s"])
    )

    global_processing_fps = (
        total_frames / total_wall_s if total_wall_s > 0 else float("inf")
    )
    global_realtime_factor = (
        total_wall_s / total_source_duration_s
        if total_source_duration_s > 0
        else float("nan")
    )

    print("\nRESUMEN GLOBAL")
    print("==============")
    print(f"Videos              : {len(rows)}")
    print(f"Inferencias          : {global_stats['n']}")
    print(f"Latencia media       : {global_stats['mean_ms']:.3f} ms")
    print(f"Latencia mediana     : {global_stats['median_ms']:.3f} ms")
    print(f"Latencia p95         : {global_stats['p95_ms']:.3f} ms")
    print(f"Latencia p99         : {global_stats['p99_ms']:.3f} ms")
    print(f"Latencia mínima      : {global_stats['min_ms']:.3f} ms")
    print(f"Latencia máxima      : {global_stats['max_ms']:.3f} ms")
    print(f"Throughput inferencia: {global_stats['infer_fps']:.2f} inferencias/s")
    print(f"Procesamiento global : {global_processing_fps:.2f} frames/s")
    print(f"Real-time factor     : {global_realtime_factor:.3f}x")

    if csv_path is not None and rows:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nResultados por video guardados en: {csv_path}")

    return rows, global_stats
