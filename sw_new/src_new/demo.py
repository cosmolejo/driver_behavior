"""
Demo de inferencia a NIVEL SEGMENTO para la arquitectura actual.

La configuracion se carga con OmegaConf (sin Hydra) y se mantienen los
backends existentes:

  - checkpoint PyTorch (.pth)
  - modelo ExecuTorch (.pte)

La decision online reproduce la agregacion usada en validacion:

    N ventanas temporales
        -> softmax por ventana
        -> promedio de probabilidades
        -> argmax del segmento

La version base mantiene un unico buffer maestro de frames crudos. Para N
ventanas, sequence_length=L y stride=S, el buffer necesita:

    L + (N - 1) * S

frames. Una vez lleno, se genera una nueva decision cada
`--inference-every` frames. Para reproducir el stride usado por
SegmentDataset, el valor recomendado es 8.

Ejemplos
--------
PyTorch, video a nivel segmento:

    python demo.py \
        --config configs/config_binary_phone_balanced.yaml \
        --checkpoint ../models/binary_phone_lad13_augmentation/model_best_segment.pth \
        --video /ruta/video.mp4 \
        --mode video \
        --segment-windows 20 \
        --window-stride 8 \
        --inference-every 8

ExecuTorch, video a nivel segmento:

    python demo.py \
        --config configs/config_binary_phone_balanced.yaml \
        --pte model_binary_phone.pte \
        --video /ruta/video.mp4 \
        --mode pte_video \
        --segment-windows 20 \
        --window-stride 8 \
        --inference-every 8

Latencia end-to-end PyTorch de UNA ventana (benchmark historico):

    python demo.py \
        --config configs/config_binary_phone_balanced.yaml \
        --checkpoint ../models/binary_phone_lad13_augmentation/model_best_segment.pth \
        --video /ruta/video.mp4 \
        --mode latency \
        --samples 500

Latencia del forward ExecuTorch de UNA ventana (benchmark historico):

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
from typing import List, Sequence

import cv2 as cv
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from predictors.base_predictor import (
    ClipPreprocessor,
    Predictor,
    resolve_class_names,
)

# Se conserva soporte para los modos multi-video existentes.
try:
    from multi_video_latency import run_multiple_video_latency
except ImportError:
    run_multiple_video_latency = None


DEFAULT_INPUT_SIZE = 112
DEFAULT_SEGMENT_WINDOWS = 20
DEFAULT_WINDOW_STRIDE = 8
DEFAULT_INFERENCE_EVERY = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo segment-level de driver behavior con OmegaConf."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Ruta al YAML usado por el modelo.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pth para modos video/latency/video_latency.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Ruta al video para modos video/latency/pte_video.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help=(
            "Lista de videos completos para video_latency o "
            "pte_video_latency."
        ),
    )
    parser.add_argument(
        "--latency-csv",
        default="latency_videos.csv",
        help="CSV con resultados por video para los modos multi-video.",
    )
    parser.add_argument(
        "--mode",
        default="video",
        choices=[
            "video",
            "latency",
            "video_latency",
            "pte_video",
            "pte_video_latency",
            "executor_latency",
        ],
    )
    parser.add_argument(
        "--pte",
        default=None,
        help="Modelo ExecuTorch .pte para los modos pte_*.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Numero de repeticiones para pruebas de latencia.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Numero de inferencias de calentamiento.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda | cuda:N",
    )
    parser.add_argument(
        "--output",
        default="demo_salida.mp4",
        help="Video de salida para modos video/pte_video.",
    )
    parser.add_argument(
        "--segment-windows",
        type=int,
        default=DEFAULT_SEGMENT_WINDOWS,
        help=(
            "Numero N de ventanas que componen la decision de segmento. "
            f"Default: {DEFAULT_SEGMENT_WINDOWS}."
        ),
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=DEFAULT_WINDOW_STRIDE,
        help=(
            "Separacion, en frames crudos, entre el inicio de ventanas "
            f"consecutivas. Default: {DEFAULT_WINDOW_STRIDE}, igual al "
            "stride por defecto de SegmentDataset."
        ),
    )
    parser.add_argument(
        "--inference-every",
        type=int,
        default=DEFAULT_INFERENCE_EVERY,
        help=(
            "Ejecuta una nueva decision de segmento cada N frames una vez "
            "lleno el buffer. Default: 8 para avanzar un window_stride."
        ),
    )
    parser.add_argument(
        "--debug-probs",
        action="store_true",
        help=(
            "Muestra en consola y sobre el video las probabilidades promedio "
            "del segmento. Util para diagnosticar sesgo safe/phone."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="No abrir ventana OpenCV (util en Raspberry Pi headless).",
    )

    return parser.parse_args()


def validate_args(args):
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el config: {config_path}")

    if args.samples <= 0:
        raise ValueError("--samples debe ser > 0.")
    if args.warmup < 0:
        raise ValueError("--warmup debe ser >= 0.")
    if args.inference_every <= 0:
        raise ValueError("--inference-every debe ser > 0.")
    if args.segment_windows <= 0:
        raise ValueError("--segment-windows debe ser > 0.")
    if args.window_stride <= 0:
        raise ValueError("--window-stride debe ser > 0.")

    if args.mode in ("video", "latency", "video_latency"):
        if args.checkpoint is None:
            raise ValueError(
                f"--checkpoint es obligatorio para mode={args.mode}."
            )
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(
                f"No existe el checkpoint: {args.checkpoint}"
            )

    if args.mode in ("video", "latency", "pte_video"):
        if args.video is None:
            raise ValueError(
                f"--video es obligatorio para mode={args.mode}."
            )
        if not Path(args.video).exists():
            raise FileNotFoundError(f"No existe el video: {args.video}")

    if args.mode in ("pte_video", "pte_video_latency", "executor_latency"):
        if args.pte is None:
            raise ValueError(f"--pte es obligatorio para mode={args.mode}.")
        if not Path(args.pte).exists():
            raise FileNotFoundError(f"No existe el modelo .pte: {args.pte}")

    if args.mode in ("video_latency", "pte_video_latency"):
        if not args.videos:
            raise ValueError(f"--videos es obligatorio para mode={args.mode}.")
        missing = [video for video in args.videos if not Path(video).exists()]
        if missing:
            raise FileNotFoundError(
                "No existen estos videos:\n" + "\n".join(missing)
            )
        if run_multiple_video_latency is None:
            raise ImportError(
                "No fue posible importar multi_video_latency.py, requerido "
                f"para mode={args.mode}."
            )


class ExecuTorchPredictor:
    """
    Predictor que usa directamente un archivo .pte.

    El preprocesamiento es el mismo ClipPreprocessor usado por Predictor
    para el checkpoint PyTorch.
    """

    def __init__(self, pte_path: str, config):
        from executorch.runtime import Runtime

        self.cfg = config
        self.pte_path = Path(pte_path)
        self.preprocessor = ClipPreprocessor(config)

        self.sequence_length = self.preprocessor.sequence_length
        self.sample_one_each = self.preprocessor.sample_one_each
        self.frames_per_window = self.preprocessor.frames_per_window
        self.input_size = self.preprocessor.input_size

        runtime = Runtime.get()
        self.program = runtime.load_program(str(self.pte_path))
        self.method = self.program.load_method("forward")

        self.class_names = None

        print("ExecuTorch predictor inicializado")
        print("--------------------------------")
        print(f"Modelo .pte       : {self.pte_path}")
        print(f"sequence_length   : {self.sequence_length}")
        print(f"sample_one_each   : {self.sample_one_each}")
        print(f"frames al modelo  : {self.frames_per_window}")
        print(
            "Input esperado    : "
            f"(1, 3, {self.frames_per_window}, "
            f"{self.input_size}, {self.input_size})"
        )

    def preprocess(self, raw_frames: np.ndarray) -> torch.Tensor:
        return self.preprocessor(raw_frames).to(
            dtype=torch.float32,
            device="cpu",
        ).contiguous()

    def predict_with_confidence(self, raw_frames: np.ndarray):
        input_tensor = self.preprocess(raw_frames)
        outputs = self.method.execute([input_tensor])

        if not outputs:
            raise RuntimeError("ExecuTorch no devolvio ninguna salida.")

        logits = outputs[0]
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if self.class_names is None:
            num_classes = int(logits.shape[-1])
            self.class_names = resolve_class_names(self.cfg, num_classes)
            print(f"Clases            : {self.class_names}")

        probabilities = torch.softmax(logits, dim=-1)
        pred_idx = int(probabilities.argmax(dim=-1).item())
        confidence = float(probabilities[0, pred_idx].item())

        label = (
            self.class_names[pred_idx]
            if pred_idx < len(self.class_names)
            else f"class_{pred_idx}"
        )

        return label, confidence, probabilities[0].cpu().numpy()

    def predict(self, raw_frames: np.ndarray) -> str:
        label, _, _ = self.predict_with_confidence(raw_frames)
        return label


def open_video(video_path: str):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"No fue posible abrir el video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    return cap, fps, width, height


def segment_buffer_size(
    sequence_length: int,
    segment_windows: int,
    window_stride: int,
) -> int:
    """Numero de frames crudos necesarios para contener N ventanas."""
    return sequence_length + (segment_windows - 1) * window_stride


def build_segment_windows(
    frames: Sequence[np.ndarray],
    sequence_length: int,
    segment_windows: int,
    window_stride: int,
) -> list[np.ndarray]:
    """
    Construye las N ventanas del segmento exactamente por indices de inicio:

        0, stride, 2*stride, ..., (N-1)*stride

    Cada ventana contiene `sequence_length` frames CRUDOS. El submuestreo
    `sample_one_each` se aplica despues dentro de ClipPreprocessor.
    """
    expected = segment_buffer_size(
        sequence_length,
        segment_windows,
        window_stride,
    )

    if len(frames) != expected:
        raise ValueError(
            f"Buffer de segmento invalido: {len(frames)} frames; "
            f"se esperaban {expected}."
        )

    # Convertir una sola vez a lista indexable. No se hace np.stack del
    # buffer completo para evitar una copia grande innecesaria en cada ciclo.
    frame_list = list(frames)
    windows = []

    for i in range(segment_windows):
        start = i * window_stride
        end = start + sequence_length
        windows.append(np.stack(frame_list[start:end], axis=0))

    return windows


def predict_segment(predictor, raw_windows: Sequence[np.ndarray]):
    """
    Decision a nivel segmento.

    PyTorch:
        Preprocesa las N ventanas y las ejecuta en batches de hasta
        `max_windows_per_forward`, igual que eval_model.py. Esto reduce de
        forma importante el overhead frente a hacer N forwards separados.

    ExecuTorch:
        Se conserva inferencia ventana-a-ventana porque el .pte actual fue
        exportado con batch fijo 1.

    En ambos casos la decision final es exactamente:
        softmax por ventana -> promedio de probabilidades -> argmax.
    """
    if not raw_windows:
        raise RuntimeError("El segmento no contiene ventanas para inferir.")

    probabilities = []

    if isinstance(predictor, Predictor):
        # Cada preprocess devuelve (1, C, T, H, W).
        prepared = [predictor.preprocess(w) for w in raw_windows]
        batch = torch.cat(prepared, dim=0)
        max_forward = int(predictor.cfg.get("max_windows_per_forward", 8))
        max_forward = max(1, max_forward)

        with torch.no_grad():
            for start in range(0, batch.shape[0], max_forward):
                chunk = batch[start:start + max_forward].to(
                    predictor.device, dtype=torch.float32
                )
                logits = predictor.model(chunk)
                probs = torch.softmax(logits, dim=1)
                probabilities.append(probs.cpu().numpy())

        probs_matrix = np.concatenate(probabilities, axis=0)
    else:
        # ExecuTorch: batch=1 para m�xima compatibilidad con el .pte actual.
        for raw_window in raw_windows:
            _, _, probs = predictor.predict_with_confidence(raw_window)
            probabilities.append(np.asarray(probs, dtype=np.float32))
        probs_matrix = np.stack(probabilities, axis=0)

    mean_probs = probs_matrix.mean(axis=0)
    pred_idx = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_idx])

    class_names = getattr(predictor, "class_names", None)
    if class_names is not None and pred_idx < len(class_names):
        label = class_names[pred_idx]
    else:
        label = f"class_{pred_idx}"

    return label, confidence, mean_probs

def draw_prediction(
    frame,
    label,
    confidence,
    buffer_len: int | None = None,
    buffer_target: int | None = None,
    probabilities: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    show_probs: bool = False,
):
    """Dibuja la salida en la esquina inferior derecha."""
    if label is None:
        if buffer_len is not None and buffer_target is not None:
            lines = [f"loading segment {buffer_len}/{buffer_target}"]
        else:
            lines = ["loading segment"]
    else:
        lines = [f"{label}  {confidence:.1%}"]

        if show_probs and probabilities is not None:
            names = list(class_names or [])
            if len(names) != len(probabilities):
                names = [f"c{i}" for i in range(len(probabilities))]
            prob_text = "  ".join(
                f"{name}:{float(prob):.2f}"
                for name, prob in zip(names, probabilities)
            )
            lines.append(prob_text)

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness = 2
    margin = 18
    line_gap = 8
    pad = 8

    sizes = [cv.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    block_w = max(w for w, _ in sizes)
    block_h = sum(h for _, h in sizes) + line_gap * (len(lines) - 1)

    x0 = max(margin, frame.shape[1] - block_w - 2 * pad - margin)
    y_bottom = frame.shape[0] - margin
    y0 = max(margin, y_bottom - block_h - 2 * pad)

    # Fondo oscuro para que el texto siga siendo legible sobre la imagen.
    overlay = frame.copy()
    cv.rectangle(
        overlay,
        (x0, y0),
        (frame.shape[1] - margin, y_bottom),
        (0, 0, 0),
        -1,
    )
    cv.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = y0 + pad
    for line, (_, h) in zip(lines, sizes):
        y += h
        cv.putText(
            frame,
            line,
            (x0 + pad, y),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv.LINE_AA,
        )
        y += line_gap

def run_video(
    cap,
    fps,
    width,
    height,
    predictor,
    output_path: str,
    segment_windows: int,
    window_stride: int,
    inference_every: int = DEFAULT_INFERENCE_EVERY,
    display: bool = True,
    debug_probs: bool = False,
):
    """
    Inferencia online a NIVEL SEGMENTO.

    Ejemplo por defecto:
      sequence_length = 32
      N               = 20
      stride          = 8
      buffer           = 184 frames crudos

    Una vez lleno el buffer se conserva como deque deslizante. Cada frame
    nuevo desplaza automaticamente el mas antiguo. La decision se actualiza
    cada `inference_every` frames.
    """
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(
            f"No fue posible crear el video de salida: {output_path}"
        )

    target_size = segment_buffer_size(
        predictor.sequence_length,
        segment_windows,
        window_stride,
    )
    segment_buffer = deque(maxlen=target_size)

    last_label = None
    last_confidence = 0.0
    last_probs = None
    frames_since_inference = 0
    n_segment_predictions = 0

    print("Inferencia segment-level")
    print("------------------------")
    print(f"N ventanas         : {segment_windows}")
    print(f"window stride      : {window_stride} frames crudos")
    print(f"buffer maestro     : {target_size} frames crudos")
    print(f"inferencia cada    : {inference_every} frames")
    print(f"latencia inicial   : ~{target_size / fps:.3f} s a {fps:.2f} FPS")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            segment_buffer.append(frame)

            if len(segment_buffer) == target_size:
                if frames_since_inference == 0:
                    raw_windows = build_segment_windows(
                        segment_buffer,
                        predictor.sequence_length,
                        segment_windows,
                        window_stride,
                    )
                    (
                        last_label,
                        last_confidence,
                        last_probs,
                    ) = predict_segment(predictor, raw_windows)
                    n_segment_predictions += 1

                    if debug_probs:
                        names = list(getattr(predictor, "class_names", []) or [])
                        if len(names) != len(last_probs):
                            names = [f"c{i}" for i in range(len(last_probs))]
                        probs_txt = ", ".join(
                            f"{n}={float(p):.4f}"
                            for n, p in zip(names, last_probs)
                        )
                        print(
                            f"segment #{n_segment_predictions}: "
                            f"{last_label} ({last_confidence:.4f}) | {probs_txt}"
                        )

                frames_since_inference = (
                    frames_since_inference + 1
                ) % inference_every

            draw_prediction(
                frame,
                last_label,
                last_confidence,
                buffer_len=len(segment_buffer),
                buffer_target=target_size,
                probabilities=last_probs,
                class_names=getattr(predictor, "class_names", None),
                show_probs=debug_probs,
            )
            out.write(frame)

            if display:
                cv.imshow("Driver Behavior Demo", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        out.release()

    print(f"Video guardado en: {output_path}")
    print(f"Decisiones de segmento realizadas: {n_segment_predictions}")


def build_video_sample(cap, sequence_length: int) -> np.ndarray:
    """
    Obtiene una ventana cruda de `sequence_length` frames consecutivos.

    Se mantiene para conservar el benchmark historico de latencia de una
    sola ventana. Predictor aplicara despues el submuestreo temporal.
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
    Benchmark historico de latencia end-to-end PyTorch de UNA ventana:
      preprocesamiento + transferencia al device + forward + postproceso.

    Se conserva sin cambiar para mantener comparabilidad con las mediciones
    anteriores del proyecto.
    """
    print("Construyendo muestra de video...")
    raw_window = build_video_sample(cap, predictor.sequence_length)

    print(f"Warm-up: {warmup} inferencias")
    for _ in range(warmup):
        predictor.predict(raw_window)

    synchronize_if_needed(predictor.device)

    print("Midiendo latencia end-to-end PyTorch (1 ventana)...")
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
    Benchmark historico del metodo forward ExecuTorch de UNA ventana.

    No incluye lectura de video ni preprocesamiento. Se conserva para
    mantener comparabilidad con las mediciones anteriores.
    """
    from executorch.runtime import Runtime

    sequence_length = int(cfg.sequence_length)
    sample_one_each = max(1, int(cfg.sample_one_each))
    frames_per_window = max(
        1,
        math.ceil(sequence_length / sample_one_each),
    )
    input_size = int(cfg.get("input_size", DEFAULT_INPUT_SIZE))

    input_tensor = torch.randn(
        1,
        3,
        frames_per_window,
        input_size,
        input_size,
        dtype=torch.float32,
    )

    print(f"Input ExecuTorch: {tuple(input_tensor.shape)}")

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    output = None
    for _ in range(warmup):
        output = method.execute([input_tensor])

    start_time = time.perf_counter()

    with tqdm(total=num_samples) as pbar:
        for _ in range(num_samples):
            output: List[torch.Tensor] = method.execute([input_tensor])
            pbar.update(1)

    elapsed = time.perf_counter() - start_time

    if not output:
        raise RuntimeError("ExecuTorch no devolvio ningun tensor de salida.")

    return elapsed / num_samples


def main():
    args = parse_args()
    validate_args(args)

    cfg = OmegaConf.load(args.config)

    print("Configuracion")
    print("-------------")
    print(f"Config             : {args.config}")
    print(f"Mode               : {args.mode}")
    print(f"segment_windows    : {args.segment_windows}")
    print(f"window_stride      : {args.window_stride}")
    print(f"inference_every    : {args.inference_every}")

    # Modos multi-video conservados tal como estaban. Estos utilizan
    # multi_video_latency.py y mantienen su benchmark previo.
    if args.mode == "video_latency":
        predictor = Predictor(
            checkpoint_path=args.checkpoint,
            config=cfg,
            device=args.device,
        )

        run_multiple_video_latency(
            video_paths=args.videos,
            predictor=predictor,
            inference_every=args.inference_every,
            warmup=args.warmup,
            csv_path=args.latency_csv,
        )
        return

    if args.mode == "pte_video_latency":
        predictor = ExecuTorchPredictor(
            pte_path=args.pte,
            config=cfg,
        )

        run_multiple_video_latency(
            video_paths=args.videos,
            predictor=predictor,
            inference_every=args.inference_every,
            warmup=args.warmup,
            csv_path=args.latency_csv,
        )
        return

    if args.mode == "executor_latency":
        avg_latency = executor_latency_test(
            cfg=cfg,
            pte_path=args.pte,
            num_samples=args.samples,
            warmup=args.warmup,
        )

        print(
            "Latencia media ExecuTorch (1 ventana): "
            f"{avg_latency * 1000:.3f} ms"
        )
        print(
            "Throughput aproximado: "
            f"{1.0 / avg_latency:.2f} inferencias/s"
        )
        return

    if args.mode == "pte_video":
        predictor = ExecuTorchPredictor(
            pte_path=args.pte,
            config=cfg,
        )


        cap, fps, width, height = open_video(args.video)

        try:
            print("Running ExecuTorch segment-level video demo")
            run_video(
                cap=cap,
                fps=fps,
                width=width,
                height=height,
                predictor=predictor,
                output_path=args.output,
                segment_windows=args.segment_windows,
                window_stride=args.window_stride,
                inference_every=args.inference_every,
                display=not args.no_display,
                debug_probs=args.debug_probs,
            )
        finally:
            cap.release()
            cv.destroyAllWindows()

        return

    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        config=cfg,
        device=args.device,
    )

    cap, fps, width, height = open_video(args.video)

    try:
        if args.mode == "video":
            print("Running PyTorch segment-level video demo")
            run_video(
                cap=cap,
                fps=fps,
                width=width,
                height=height,
                predictor=predictor,
                output_path=args.output,
                segment_windows=args.segment_windows,
                window_stride=args.window_stride,
                inference_every=args.inference_every,
                display=not args.no_display,
                debug_probs=args.debug_probs,
            )

        elif args.mode == "latency":
            avg_latency = run_latency_test(
                cap=cap,
                num_samples=args.samples,
                predictor=predictor,
                warmup=args.warmup,
            )

            print(
                "Latencia media PyTorch end-to-end (1 ventana): "
                f"{avg_latency * 1000:.3f} ms"
            )
            print(
                "Throughput aproximado: "
                f"{1.0 / avg_latency:.2f} inferencias/s"
            )

    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()