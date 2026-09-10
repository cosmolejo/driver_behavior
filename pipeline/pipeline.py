#!/usr/bin/env python3
"""
pipeline.py — Full distraction-detection pipeline with CAN context.

Chains: frame capture (video file or USB camera) -> per-model inference
-> CAN context gate -> N-iteration moving average -> alert.

Modalities (inferred from which sources are passed)
    --body-* only       phone model        rules G1, G2, D
    --face-* only       reaching model      rules G1, G2, R1, R2, D
    both                both models         all rules, independently

Each feed takes either --{feed}-video (a file) or --{feed}-camera (a USB
device index). Cameras are captured in a background thread; files are read
synchronously and paced to real time when CAN is active.

SIZING. One iteration costs `--segment-windows` forward passes and must fit
in `--inference-every / fps` seconds. Measure the per-window cost on the
target device first:

    python demo.py --config <cfg> --pte <model.pte> \\
        --video-dir <dir> --mode pte_video_latency --latency-csv lat.csv

then pick K so that K x p99_latency stays under the budget. The pipeline
prints that budget at startup. On a Raspberry Pi at ~84 ms/window, K=20
(the validation setting) needs 1.7 s per decision and is not viable; K=2
with --inference-every 8, or K=4 with 16, are realistic. The moving average
recovers part of the lost bagging at no compute cost.

One iteration yields one decision per model. With --iteration-mode segment
(default) each decision aggregates N windows by averaging probabilities,
the same as in validation. With --iteration-mode window the decision uses a
single window, which reduces the initial latency at the cost of no bagging.

On the moving average: the distraction-class probability is averaged over
the last N iterations, and an alert fires once that average crosses the
threshold. An iteration suppressed by a CAN rule enters the average as safe
(p_distraction = 0), so it dilutes the accumulator instead of resetting it
abruptly.

Examples
--------
Both feeds, with virtual CAN:

    python3 pipeline.py \\
        --face-video face.mp4 \\
        --face-config configs/config_binary_reaching.yaml \\
        --face-checkpoint ../models/binary_reaching_lad13_augmentation/model_best_segment.pth \\
        --body-video body.mp4 \\
        --body-config configs/config_binary_phone_balanced.yaml \\
        --body-checkpoint ../models/binary_phone_lad13_augmentation/model_best_segment.pth \\
        --can-channel vcan0 \\
        --csv resultados.csv

Body only, no CAN (baseline for measuring the filter's effect):

    python3 pipeline.py \\
        --body-video body.mp4 \\
        --body-config configs/config_binary_phone_balanced.yaml \\
        --body-pte model_binary_phone.pte \\
        --no-can --csv baseline.csv

Injecting a can_injector.py scenario in parallel with the feeds, to
reproduce the CAN context in a controlled way while there is still no access
to the real bus (uses the same --can-channel as CanContextReader; the
injector does not need to run as a separate process):

    python3 pipeline.py \\
        --face-video face.mp4 --face-config configs/config_binary_reaching.yaml \\
        --face-checkpoint checkpoints/reaching.pth \\
        --can-channel vcan0 \\
        --inject-scenario aparcando \\
        --csv con_aparcando.csv

With a custom scenario, looped, starting 3 s after launch:

    python3 pipeline.py \\
        --body-video body.mp4 --body-config configs/config_binary_phone_balanced.yaml \\
        --body-checkpoint checkpoints/phone.pth \\
        --inject-scenario mi_escenario.yaml --inject-loop --inject-delay 3

List the built-in scenarios:

    python3 pipeline.py --list-scenarios
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np
from omegaconf import OmegaConf

# Reuses the already-validated logic from demo.py instead of duplicating it:
# the windowing and the probability-averaging aggregation must be bit-for-bit
# identical to offline evaluation.
from demo import (
    ExecuTorchPredictor,
    build_segment_windows,
    open_video,
    predict_segment,
    segment_buffer_size,
)

try:
    from predictors.base_predictor import Predictor
except ImportError:  # running from the same directory
    from base_predictor import Predictor

from can_context import CanContextReader, NullContextReader
import can_injector

DEFAULT_SEGMENT_WINDOWS = 20
DEFAULT_WINDOW_STRIDE = 8
DEFAULT_INFERENCE_EVERY = 8
DEFAULT_N_PERSIST = 3
DEFAULT_THRESHOLD = 0.5

# feed -> (model name, expected distraction class)
FEEDS = {
    "face": ("reaching", "reaching"),
    "body": ("phone", "phone"),
}


# --------------------------------------------------------------------------
# Class resolution
# --------------------------------------------------------------------------


def scheme_class_names(cfg, feed: str) -> list[str] | None:
    """Class names according to the scheme declared in the YAML.

    This is the authoritative source: LABEL_SCHEMES in fine_labels.py is what
    defined the index order during training. base_predictor's
    resolve_class_names() keeps a partial copy of that dict that doesn't know
    'binary_reaching' and that for 'ternary_clean' returns
    ['reaching','safe','unsafe'] instead of ['safe','reaching','phone'].
    """
    label_mode = str(cfg.get("label_mode", "")).strip()
    if not label_mode:
        return None
    try:
        from fine_labels import get_scheme

        _, names = get_scheme(label_mode, cfg.get("partition_report", None))
        return list(names)
    except Exception as exc:
        print(f"WARNING [{feed}]: could not read scheme {label_mode!r}: {exc}")
        return None


def resolve_indices(class_names, expected: str, feed: str, override: str | None):
    """Returns (idx_safe, idx_distraction).

    Works with binary schemes and with ternary_clean, because it locates the
    two classes by name instead of assuming a fixed length.
    """
    if override:
        class_names = [n.strip() for n in override.split(",")]

    names = list(class_names or [])
    if expected in names and "safe" in names:
        return names.index("safe"), names.index(expected)

    raise SystemExit(
        f"[{feed}] could not determine the class order from "
        f"{names}. Check 'label_mode' in the YAML, or pass the checkpoint's "
        f"real order with --{feed}-classes safe,{expected}"
    )


# --------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------


class FileSource:
    """Video file. Frames are pulled synchronously, one per loop iteration."""

    live = False

    def __init__(self, path: str):
        self.cap, self.fps, self.width, self.height = open_video(path)
        self.label = path

    def read_pending(self) -> list:
        ok, frame = self.cap.read()
        return [frame] if ok else []

    def release(self) -> None:
        self.cap.release()


class CameraSource:
    """USB camera, captured in a background thread.

    The thread exists so that V4L2 buffers are drained at the sensor's own
    rate regardless of how long inference takes. If capture shared the main
    loop, frames would pile up in the driver queue while a decision was being
    computed and every later read would return stale images.

    Frames are QUEUED, not overwritten, and read_pending() hands back every
    frame captured since the last call. This matters more than it looks:
    the model's temporal buffer has to stay evenly spaced, because
    sample_one_each assumes a constant frame interval. Keeping only the
    newest frame would punch a hole in the buffer for the whole duration of
    each inference and shift the input away from the training distribution.

    The queue is bounded. If inference runs slower than real time for long
    enough the oldest frames are discarded, which DOES break the spacing;
    that case is counted in `dropped` and reported at the end rather than
    passing silently.
    """

    live = True

    def __init__(self, device: int, width: int, height: int, fps: float):
        self.cap = cv.VideoCapture(device)
        if not self.cap.isOpened():
            raise SystemExit(
                f"Could not open camera device {device}. Check `ls /dev/video*` "
                "and that no other process is holding it."
            )

        if width:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv.CAP_PROP_FPS, fps)
        # Smallest possible driver queue: buffering belongs here, where it can
        # be observed, not inside the driver.
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        reported = float(self.cap.get(cv.CAP_PROP_FPS))
        # Some UVC drivers report 0 or a nonsense value; fall back to the
        # requested rate so the buffer-span arithmetic stays meaningful.
        self.fps = reported if 1.0 < reported < 240.0 else (fps or 30.0)
        self.reported_fps = reported
        self.label = f"camera {device}"

        # ~2 s of slack: enough to ride out a slow decision without losing
        # spacing, small enough that a persistent backlog is caught quickly.
        self._pending: deque = deque(maxlen=max(8, int(2 * self.fps)))
        self._lock = threading.Lock()
        self._running = True
        self.captured = 0
        self.dropped = 0

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                if len(self._pending) == self._pending.maxlen:
                    self.dropped += 1  # oldest is about to fall off the deque
                self._pending.append(frame)
                self.captured += 1

    def read_pending(self) -> list:
        """Every frame captured since the last call, oldest first."""
        with self._lock:
            frames = list(self._pending)
            self._pending.clear()
        return frames

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# --------------------------------------------------------------------------
# A video feed with its model
# --------------------------------------------------------------------------


class Feed:
    def __init__(self, feed: str, args, iteration_mode: str):
        self.feed = feed
        self.model_name, self.expected_class = FEEDS[feed]

        video = getattr(args, f"{feed}_video")
        camera = getattr(args, f"{feed}_camera")
        config = getattr(args, f"{feed}_config")
        checkpoint = getattr(args, f"{feed}_checkpoint")
        pte = getattr(args, f"{feed}_pte")
        classes = getattr(args, f"{feed}_classes")

        if config is None:
            raise SystemExit(f"--{feed}-config is required with --{feed}-video.")
        if (checkpoint is None) == (pte is None):
            raise SystemExit(
                f"Pass exactly one of --{feed}-checkpoint or --{feed}-pte."
            )

        self.cfg = OmegaConf.load(config)

        if checkpoint is not None:
            self.predictor = Predictor(
                checkpoint_path=checkpoint, config=self.cfg, device=args.device
            )
            self.backend = "pytorch"
        else:
            self.predictor = ExecuTorchPredictor(pte_path=pte, config=self.cfg)
            self.backend = "executorch"

        if camera is not None:
            self.source = CameraSource(
                camera, args.camera_width, args.camera_height, args.camera_fps
            )
        else:
            self.source = FileSource(video)
        self.fps = self.source.fps
        self.live = self.source.live

        # --segment-windows is authoritative in every mode. 'window' is now a
        # shorthand for K=1 and only applies when K was left at its default:
        # on constrained hardware the useful setting is an intermediate K
        # (2, 4, 8), not the 1-or-20 dichotomy.
        if iteration_mode == "window" and args.segment_windows == DEFAULT_SEGMENT_WINDOWS:
            self.segment_windows = 1
        else:
            self.segment_windows = args.segment_windows
        self.window_stride = args.window_stride

        self.buffer_size = segment_buffer_size(
            self.predictor.sequence_length, self.segment_windows, self.window_stride
        )
        self.buffer: deque = deque(maxlen=self.buffer_size)

        # The YAML scheme takes precedence over whatever the predictor returns;
        # this way binary_reaching resolves without flags and ternary_clean
        # doesn't end up with swapped names. Resolved here so it fails at
        # startup, not halfway through the video.
        self.classes_override = classes
        self.class_names = scheme_class_names(self.cfg, feed) or getattr(
            self.predictor, "class_names", None
        )
        self.idx_safe = None
        self.idx_distraction = None
        if self.class_names is not None:
            self.idx_safe, self.idx_distraction = resolve_indices(
                self.class_names, self.expected_class, self.feed, classes
            )

        self.last_probs: np.ndarray | None = None
        self.last_label: str | None = None
        self.last_frame = None

    def _ensure_indices(self) -> None:
        if self.idx_distraction is not None:
            return
        names = getattr(self.predictor, "class_names", None) or self.class_names
        self.class_names = names
        self.idx_safe, self.idx_distraction = resolve_indices(
            names, self.expected_class, self.feed, self.classes_override
        )

    def read_pending(self) -> list:
        return self.source.read_pending()

    def push(self, frame) -> None:
        self.buffer.append(frame)

    @property
    def ready(self) -> bool:
        return len(self.buffer) == self.buffer_size

    def infer(self) -> tuple[str, float, np.ndarray]:
        windows = build_segment_windows(
            self.buffer,
            self.predictor.sequence_length,
            self.segment_windows,
            self.window_stride,
        )
        label, confidence, probs = predict_segment(self.predictor, windows)
        self._ensure_indices()
        self.last_probs = probs
        self.last_label = label
        return label, confidence, probs

    def p_distraction(self) -> float:
        if self.last_probs is None or self.idx_distraction is None:
            return 0.0
        return float(self.last_probs[self.idx_distraction])

    def describe(self) -> str:
        span = self.buffer_size / self.fps
        kind = "camera" if self.live else "file"
        return (
            f"  {self.feed:<5s} -> {self.model_name:<9s} {self.backend:<11s} "
            f"{kind:<7s} {self.fps:5.2f} fps  buffer {self.buffer_size:4d} fr "
            f"({span:5.2f} s)  windows {self.segment_windows}"
        )

    def release(self) -> None:
        self.source.release()


# --------------------------------------------------------------------------
# Temporal persistence (rule D)
# --------------------------------------------------------------------------


class PersistenceFilter:
    """Moving average of the distraction probability over N iterations.

    An iteration suppressed by the CAN context is injected as safe
    (p = 0.0). The accumulator is neither dropped nor reset: it dilutes the
    evidence proportionally, consistent with averaging probabilities.
    """

    def __init__(self, n: int, threshold: float):
        self.n = n
        self.threshold = threshold
        self.history: deque = deque(maxlen=n)

    def push(self, p_distraction: float, gate_open: bool,
             sensitivity: float = 1.0) -> tuple[float, bool, float]:
        self.history.append(p_distraction if gate_open else 0.0)
        mean = float(np.mean(self.history)) if self.history else 0.0
        # M3 / M4 modulate the threshold rather than the probability: the
        # model's output is left untouched, only how much evidence is demanded
        # of it. With the factors neutral this is exactly the previous
        # behaviour.
        effective = self.threshold * sensitivity
        alert = len(self.history) == self.n and mean >= effective
        return mean, alert, effective


# --------------------------------------------------------------------------
# CAN scenario injection
# --------------------------------------------------------------------------


def resolve_scenario(spec: str) -> tuple[dict, str]:
    """Name of a built-in scenario, or a path to a custom .yaml/.json."""
    if spec in can_injector.BUILTIN_SCENARIOS:
        return can_injector.BUILTIN_SCENARIOS[spec], spec
    path = Path(spec)
    if not path.exists():
        options = ", ".join(can_injector.BUILTIN_SCENARIOS)
        raise SystemExit(
            f"--inject-scenario {spec!r} is neither a built-in scenario "
            f"({options}) nor an existing path."
        )
    return can_injector.load_scenario_file(str(path)), path.stem


class ScenarioRunner:
    """Launches can_injector.Injector on the same channel CanContextReader uses.

    The injector transmits the three frames cyclically from the moment it is
    created, using the idle values declared in can_injector.BASELINE. When
    --inject-scenario is given, it additionally replays that sequence of
    changes on those same frames; the rest of the time the bus stays idle,
    indistinguishable from not having an injector at all.

    Shares process and channel with CanContextReader.evaluate(): being on the
    same channel, frames accumulate in the gate exactly as they would from a
    real can0, with no special test-only code path.
    """

    def __init__(self, dbc_path: str, channel: str, interface: str):
        self.injector = can_injector.Injector(dbc_path, channel, interface)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, scenario: dict, name: str, delay: float, loop: bool) -> None:
        self._thread = threading.Thread(
            target=self._run, args=(scenario, name, delay, loop), daemon=True
        )
        self._thread.start()

    def _run(self, scenario: dict, name: str, delay: float, loop: bool) -> None:
        if delay > 0 and self._stop.wait(delay):
            return
        while not self._stop.is_set():
            self.injector.run_scenario(scenario, name)
            # run_scenario() is async: wait for it to finish before repeating
            # or returning control.
            while self.injector.scenario_thread and self.injector.scenario_thread.is_alive():
                if self._stop.wait(0.05):
                    return
            if not loop:
                return

    def shutdown(self) -> None:
        self._stop.set()
        self.injector.scenario_stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.injector.shutdown()


# --------------------------------------------------------------------------
# Diagnostic overlay
# --------------------------------------------------------------------------

FONT = cv.FONT_HERSHEY_SIMPLEX
COL_OK = (120, 220, 120)         # BGR: gate open
COL_SUPPRESSED = (60, 190, 240)  # amber: suppressed by a CAN rule
COL_ALERT = (60, 60, 235)        # red: alerting
COL_TEXT = (240, 240, 240)
COL_DIM = (160, 160, 160)
COL_PANEL_BG = (26, 26, 26)
COL_BAR_BG = (62, 62, 62)

ALERT_BORDER_PX = 6  # peripheral cue: a thin frame, not a band over the image


def _text_w(text: str, scale: float, thick: int = 1) -> int:
    return cv.getTextSize(text, FONT, scale, thick)[0][0]


def _fit_scale(text: str, max_w: int, start: float = 0.5,
               thick: int = 1, floor: float = 0.30) -> float:
    """Largest scale at which `text` fits in `max_w`.

    Every panel line goes through this, so a long list of rule names shrinks
    instead of running off the edge. The previous version drew at a fixed
    scale and simply lost whatever did not fit, which is how the iteration
    counter ended up truncated.
    """
    scale = start
    while scale > floor and _text_w(text, scale, thick) > max_w:
        scale -= 0.02
    return max(scale, floor)


def _put(img, text: str, x: int, y: int, scale: float = 0.5,
         colour=COL_TEXT, thick: int = 1) -> None:
    cv.putText(img, text, (x, y), FONT, scale, colour, thick, cv.LINE_AA)


def draw_overlay(frame, row, gate, feeds, args, alert_class, drift_s):
    """Video on top, telemetry in a panel underneath.

    Nothing is drawn over the image except a thin border while alerting: the
    point of the view is to judge camera framing and driver posture, which a
    panel sitting on top of the frame defeats.
    """
    video = frame
    if args.video_scale != 1.0:
        video = cv.resize(video, None, fx=args.video_scale, fy=args.video_scale,
                          interpolation=cv.INTER_LINEAR)
    else:
        video = video.copy()
    vh, vw = video.shape[:2]

    if alert_class:
        cv.rectangle(video, (0, 0), (vw - 1, vh - 1), COL_ALERT, ALERT_BORDER_PX)

    pad = 12
    line_h = 22
    feed_block = 40                      # model line + progress bar
    n_rows = 3 + (1 if alert_class else 0)   # CAN, rules, timing (+ alert)
    panel_h = pad + len(feeds) * feed_block + n_rows * line_h + pad
    panel = np.full((panel_h, vw, 3), COL_PANEL_BG, np.uint8)

    inner_w = vw - 2 * pad
    y = pad + 14

    # --- one block per feed: name, probability, moving average, gate state ---
    for name, feed in feeds.items():
        model = feed.model_name
        p = row.get(f"{name}_p_{model}", 0.0) or 0.0
        ma = row.get(f"ma_{model}", 0.0) or 0.0
        thr = row.get("threshold_eff", 0.0) or 0.0
        open_gate = bool(gate[model])
        colour = COL_OK if open_gate else COL_SUPPRESSED
        state = "GATE OPEN" if open_gate else "SUPPRESSED"

        # The gate label is right-aligned from its measured width, so it can
        # never collide with the text to its left regardless of video width.
        state_w = _text_w(state, 0.52, 2)
        left = f"{model}   p={p:.3f}   ma={ma:.3f} / {thr:.2f}"
        left_scale = _fit_scale(left, inner_w - state_w - 16, start=0.52)

        _put(panel, left, pad, y, left_scale)
        _put(panel, state, vw - pad - state_w, y, 0.52, colour, 2)
        y += 12

        bx, bw, bh = pad, inner_w, 8
        cv.rectangle(panel, (bx, y), (bx + bw, y + bh), COL_BAR_BG, -1)
        cv.rectangle(panel, (bx, y), (bx + int(bw * min(ma, 1.0)), y + bh),
                     colour, -1)
        thr_x = bx + int(bw * min(thr, 1.0))
        cv.line(panel, (thr_x, y - 2), (thr_x, y + bh + 2), (255, 255, 255), 1)
        y += feed_block - 12

    # --- CAN signals ---
    speed = gate["speed"]
    speed_txt = "n/a" if speed != speed else f"{speed:.1f} km/h"  # NaN check
    can_line = (
        f"CAN   V={speed_txt}   rev={gate['reverse']}   turn={gate['turn']}   "
        f"haz={gate['hazard']}   stop={gate['traffic_stop']}   "
        f"rain={gate['raining']}({gate['wiper_speed']})   beam={gate['low_beam']}"
    )
    _put(panel, can_line, pad, y, _fit_scale(can_line, inner_w, start=0.48))
    y += line_h

    # --- rules that fired this iteration ---
    reasons = " ".join(gate["reasons"]) if gate["reasons"] else "none active"
    rules_line = f"rules: {reasons}"
    _put(panel, rules_line, pad, y, _fit_scale(rules_line, inner_w, start=0.48),
         COL_DIM)
    y += line_h

    # --- clocks. Drift is the tell for CAN/video misalignment. ---
    timing = (
        f"video t={row['t_video_s']:.2f}s   wall t={row['t_wall_s']:.2f}s   "
        f"drift={drift_s:+.2f}s   iter {row['iteration']}"
    )
    drift_col = COL_TEXT if abs(drift_s) < 0.5 else COL_SUPPRESSED
    _put(panel, timing, pad, y, _fit_scale(timing, inner_w, start=0.46), drift_col)
    y += line_h

    if alert_class:
        _put(panel, f"ALERT: {alert_class.upper()}", pad, y + 2, 0.72,
             COL_ALERT, 2)

    canvas = np.vstack([video, panel])
    if args.display_scale != 1.0:
        canvas = cv.resize(canvas, None, fx=args.display_scale,
                           fy=args.display_scale)
    return canvas


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    for feed in FEEDS:
        p.add_argument(f"--{feed}-video", default=None)
        p.add_argument(
            f"--{feed}-camera",
            type=int,
            default=None,
            metavar="DEV",
            help=f"USB camera index for the {feed} feed (0, 1, ...); see "
                 f"`ls /dev/video*`. Alternative to --{feed}-video.",
        )
        p.add_argument(f"--{feed}-config", default=None)
        p.add_argument(f"--{feed}-checkpoint", default=None)
        p.add_argument(f"--{feed}-pte", default=None)
        p.add_argument(
            f"--{feed}-classes",
            default=None,
            help=f"Real class order of the checkpoint, e.g. 'safe,{FEEDS[feed][1]}'.",
        )

    p.add_argument("--camera-width", type=int, default=0, help="0 = driver default.")
    p.add_argument("--camera-height", type=int, default=0, help="0 = driver default.")
    p.add_argument(
        "--camera-fps",
        type=float,
        default=30.0,
        help="Requested capture rate, and fallback when the driver reports none.",
    )

    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")

    p.add_argument(
        "--iteration-mode",
        default="segment",
        choices=["segment", "window"],
        help="segment: N windows with bagging. window: a single window.",
    )
    p.add_argument("--segment-windows", type=int, default=DEFAULT_SEGMENT_WINDOWS)
    p.add_argument("--window-stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    p.add_argument(
        "--inference-every",
        type=int,
        default=DEFAULT_INFERENCE_EVERY,
        help="Raw frames between consecutive iterations.",
    )

    p.add_argument(
        "--n-persist",
        type=int,
        default=DEFAULT_N_PERSIST,
        help="Moving-average iterations required before an alert can fire.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold on the moving average of the distraction probability.",
    )

    p.add_argument("--dbc", default="BODY_CAN.dbc")
    p.add_argument("--can-channel", default="vcan0")
    p.add_argument("--can-interface", default="socketcan")
    p.add_argument("--no-can", action="store_true", help="Disables the CAN filter.")

    p.add_argument(
        "--inject-scenario",
        default=None,
        metavar="NAME_OR_PATH",
        help="Name of a can_injector.py built-in scenario, or a path to a custom "
             ".yaml/.json. Runs in parallel with the feeds, over "
             "--can-channel/--can-interface. Requires not using --no-can.",
    )
    p.add_argument(
        "--inject-loop",
        action="store_true",
        help="Loops the scenario for as long as the video runs.",
    )
    p.add_argument(
        "--inject-delay",
        type=float,
        default=0.0,
        help="Seconds to wait before launching the scenario, to let some "
             "baseline iterations run before injecting.",
    )
    p.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Lists can_injector.py's built-in scenarios and exits.",
    )

    p.add_argument("--csv", default=None, help="Per-iteration trace.")
    p.add_argument("--verbose", action="store_true", help="Prints every iteration.")
    p.add_argument(
        "--display",
        action="store_true",
        help="Opens a window with the feed, the model probability, the CAN "
             "signals and the rule that fired. Press 'q' to quit. Needs a "
             "display (or X forwarding over SSH: ssh -X).",
    )
    p.add_argument(
        "--video-scale",
        type=float,
        default=2.0,
        help="Magnifies the video before the panel is appended. DMD clips are "
             "426x240, so the default 2.0 makes framing judgeable.",
    )
    p.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scales the whole composed view (video + panel) at the very end.",
    )
    p.add_argument(
        "--display-video",
        default=None,
        metavar="PATH",
        help="Also writes the annotated view to a video file.",
    )
    p.add_argument(
        "--realtime",
        dest="realtime",
        action="store_true",
        default=None,
        help="Plays the video at real-time speed. Default: on when CAN is active.",
    )
    p.add_argument("--no-realtime", dest="realtime", action="store_false")

    return p.parse_args()


def validate(args) -> list[str]:
    if args.list_scenarios:
        print("can_injector.py built-in scenarios:\n")
        for name, sc in can_injector.BUILTIN_SCENARIOS.items():
            print(f"  {name:<16s} {sc['description']}")
        raise SystemExit(0)

    if args.inject_scenario and args.no_can:
        raise SystemExit("--inject-scenario requires not using --no-can.")

    active = [
        f for f in FEEDS
        if getattr(args, f"{f}_video") or getattr(args, f"{f}_camera") is not None
    ]
    if not active:
        raise SystemExit(
            "Pass at least one source: --face-video / --body-video, "
            "or --face-camera / --body-camera."
        )

    for feed in active:
        video = getattr(args, f"{feed}_video")
        camera = getattr(args, f"{feed}_camera")
        if video and camera is not None:
            raise SystemExit(
                f"Pass --{feed}-video or --{feed}-camera, not both."
            )
        if video and not Path(video).exists():
            raise SystemExit(f"Video not found: {video}")

    for name in ("segment_windows", "window_stride", "inference_every", "n_persist"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be > 0.")
    if not 0.0 < args.threshold <= 1.0:
        raise SystemExit("--threshold must be in (0, 1].")

    if args.realtime is None:
        args.realtime = not args.no_can

    return active


CSV_COLUMNS = [
    "iteration", "t_video_s", "t_wall_s",
    "face_label", "face_p_reaching", "gate_reaching", "ma_reaching",
    "body_label", "body_p_phone", "gate_phone", "ma_phone",
    "can_speed", "can_reverse", "can_turn", "can_hazard",
    "can_traffic_stop", "can_autostop_enable",
    "can_raining", "can_wiper_speed", "can_low_beam",
    "can_sensitivity", "threshold_eff", "can_reasons",
    "alert", "alert_class",
]


def main() -> None:
    args = parse_args()
    active = validate(args)

    print("Distraction detection pipeline")
    print("-------------------------------")
    modality = "+".join(active)
    print(f"Modality           : {modality}")

    feeds = {f: Feed(f, args, args.iteration_mode) for f in active}
    for feed in feeds.values():
        print(feed.describe())

    if args.no_can:
        context = NullContextReader()
        print("CAN context        : disabled (vision-only baseline)")
    else:
        context = CanContextReader(args.dbc, args.can_channel, args.can_interface)
        print(f"CAN context        : {args.can_channel} ({args.can_interface})")

    scenario_runner = None
    scenario = scenario_name = None
    if args.inject_scenario:
        scenario, scenario_name = resolve_scenario(args.inject_scenario)
        scenario_runner = ScenarioRunner(args.dbc, args.can_channel, args.can_interface)
        loop_note = ", looped" if args.inject_loop else ""
        delay_note = f", after {args.inject_delay:.1f} s" if args.inject_delay else ""
        print(f"CAN injection      : {scenario_name}{delay_note}{loop_note}")
        # The Injector itself already transmits the idle frames (BASELINE)
        # from the moment it's created; the scenario sequence is deliberately
        # delayed until right before the frame loop (see start_wall below),
        # so that t=0 of the scenario lines up with t=0 of the video.

    # The R rules only affect reaching; without a front feed the
    # subsystem they suppress doesn't exist.
    rules = ["G1", "G2", "D"] + (["R1", "R2"] if "face" in feeds else [])
    print(f"Active rules       : {', '.join(sorted(rules))}")

    fps = min(f.fps for f in feeds.values())
    if len({round(f.fps, 2) for f in feeds.values()}) > 1:
        print(
            f"WARNING: feeds have different fps "
            f"{ {k: round(v.fps, 2) for k, v in feeds.items()} }; "
            f"advancing frame-locked using {fps:.2f} fps as reference."
        )

    print(f"Iteration          : every {args.inference_every} frames "
          f"({args.inference_every / fps:.3f} s at {fps:.2f} fps)")
    windows_total = sum(f.segment_windows for f in feeds.values())
    budget_ms = 1000.0 * args.inference_every / fps
    print(f"Budget             : {windows_total} window(s) per iteration in "
          f"{budget_ms:.0f} ms  ->  {budget_ms / windows_total:.0f} ms/window max")
    print(f"Persistence        : moving average of {args.n_persist} iterations, "
          f"threshold {args.threshold:.2f}")
    print()

    persistence = {
        f: PersistenceFilter(args.n_persist, args.threshold) for f in feeds
    }

    writer = None
    csv_file = None
    writer_video = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    frame_idx = 0
    iteration = 0
    skipped_iterations = 0
    prev_alert: str | None = None
    lag_warned = False

    live = any(f.live for f in feeds.values())
    reference_feed = active[0]
    iteration_period = args.inference_every / fps
    next_inference_frame = args.inference_every

    start_wall = time.perf_counter()
    next_deadline = start_wall + iteration_period

    if scenario_runner:
        scenario_runner.start(scenario, scenario_name, args.inject_delay, args.inject_loop)

    try:
        while True:
            pending = {name: feed.read_pending() for name, feed in feeds.items()}

            if live:
                # No new frames yet is normal for a camera; wait, don't stop.
                if not any(pending.values()):
                    time.sleep(0.002)
                    continue
            elif any(len(p) == 0 for p in pending.values()):
                break  # stops with whichever file feed is shorter

            # Every pending frame goes into the buffer, not just the newest:
            # the buffer has to stay evenly spaced for sample_one_each to mean
            # what it meant during training.
            for name, feed in feeds.items():
                for frame in pending[name]:
                    feed.push(frame)
                if pending[name]:
                    feed.last_frame = pending[name][-1]
            frame_idx += len(pending[reference_feed])

            if args.realtime and not live:
                # File playback only. CAN arrives in real time; without this
                # brake the video would be consumed much faster and the
                # timestamps of both sources would no longer be comparable.
                # A camera needs no brake: it already runs at wall-clock rate.
                target = start_wall + frame_idx / fps
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0 and not lag_warned:
                    lag_warned = True
                    print(
                        "WARNING: inference is not keeping up with real time; video and "
                        "CAN timestamps are no longer aligned.",
                        file=sys.stderr,
                    )

            if not all(feed.ready for feed in feeds.values()):
                continue
            # Frame-index scheduling rather than a per-pass counter: with a
            # camera several frames can arrive in one pass, so a counter that
            # advanced by one would drift. Equivalent to demo.py's modulo when
            # frames arrive one at a time, as they do from a file.
            if frame_idx < next_inference_frame:
                continue
            # Snap forward rather than accumulate: the buffer only becomes
            # ready at frame `buffer_size`, so a `+=` here would fire once for
            # every scheduled slot that elapsed while the buffer was filling
            # (8, 16, 24, ... up to buffer_size) in one burst.
            next_inference_frame = frame_idx + args.inference_every

            # Skip WHOLE iterations when running behind, never individual
            # frames: the buffer stays evenly spaced either way, we simply
            # make fewer decisions from it. This applies to files as well as
            # cameras: when inference is slower than real time the CAN
            # scenario runs ahead of the video, and without skipping the two
            # clocks drift apart permanently.
            if time.perf_counter() > next_deadline:
                skipped_iterations += 1
                next_deadline = time.perf_counter() + iteration_period
                if not lag_warned:
                    lag_warned = True
                    print(
                        f"WARNING: inference is slower than the "
                        f"{iteration_period * 1000:.0f} ms iteration period; "
                        "skipping whole iterations to stay aligned with CAN. "
                        "Lower --segment-windows or raise --inference-every.",
                        file=sys.stderr,
                    )
                continue
            next_deadline += iteration_period

            # Closes the CAN accumulation interval right here, so that the
            # OR over the interval coincides with one iteration.
            gate = context.evaluate()
            iteration += 1

            row = {c: "" for c in CSV_COLUMNS}
            row["iteration"] = iteration
            row["t_video_s"] = round(frame_idx / fps, 3)
            row["t_wall_s"] = round(time.perf_counter() - start_wall, 3)
            row["can_speed"] = round(gate["speed"], 2)
            row["can_reverse"] = gate["reverse"]
            row["can_turn"] = gate["turn"]
            row["can_hazard"] = gate["hazard"]
            row["can_traffic_stop"] = gate["traffic_stop"]
            row["can_autostop_enable"] = gate["autostop_enable"]
            row["can_raining"] = gate["raining"]
            row["can_wiper_speed"] = gate["wiper_speed"]
            row["can_low_beam"] = gate["low_beam"]
            row["can_sensitivity"] = round(gate["sensitivity"], 4)
            row["can_reasons"] = " ".join(gate["reasons"])

            alert_class = None
            for name, feed in feeds.items():
                model = feed.model_name  # 'reaching' or 'phone'
                label, _, _ = feed.infer()
                p = feed.p_distraction()
                gate_open = gate[model]
                mean, alert, effective = persistence[name].push(
                    p, gate_open, gate["sensitivity"]
                )

                row[f"{name}_label"] = label
                row[f"{name}_p_{model}"] = round(p, 4)
                row[f"gate_{model}"] = int(gate_open)
                row[f"ma_{model}"] = round(mean, 4)
                row["threshold_eff"] = round(effective, 4)

                if alert and alert_class is None:
                    alert_class = model

            row["alert"] = int(alert_class is not None)
            row["alert_class"] = alert_class or ""

            if writer:
                writer.writerow(row)

            if args.verbose:
                parts = [f"[{iteration:5d}] t={row['t_video_s']:7.2f}s"]
                for name, feed in feeds.items():
                    model = feed.model_name
                    parts.append(
                        f"{model}: p={row[f'{name}_p_{model}']:.3f} "
                        f"ma={row[f'ma_{model}']:.3f} "
                        f"gate={'ok ' if gate[model] else 'SUP'}"
                    )
                parts.append(f"V={gate['speed']:6.1f}")
                if gate["reasons"]:
                    parts.append(" ".join(gate["reasons"]))
                print("  ".join(parts))

            if alert_class != prev_alert:
                if alert_class:
                    print(
                        f">> ALERT   t={row['t_video_s']:7.2f}s  class={alert_class}  "
                        f"moving_avg={row[f'ma_{alert_class}']:.3f}"
                    )
                else:
                    print(f"   alert ended    t={row['t_video_s']:7.2f}s")
                prev_alert = alert_class

            if args.display or writer_video is not None:
                base = feeds[reference_feed].last_frame
                if base is not None:
                    drift = row["t_video_s"] - row["t_wall_s"]
                    annotated = draw_overlay(
                        base, row, gate, feeds, args, alert_class, drift
                    )
                    if writer_video is None and args.display_video:
                        h, w = annotated.shape[:2]
                        writer_video = cv.VideoWriter(
                            args.display_video,
                            cv.VideoWriter_fourcc(*"mp4v"),
                            fps / args.inference_every,  # one frame per iteration
                            (w, h),
                        )
                    if writer_video is not None:
                        writer_video.write(annotated)
                    if args.display:
                        cv.imshow("pipeline", annotated)
                        if cv.waitKey(1) & 0xFF == ord("q"):
                            print("\nClosed by user.")
                            break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for feed in feeds.values():
            feed.release()
        if scenario_runner:
            scenario_runner.shutdown()  # stops the sequence and the bus itself
        context.shutdown()
        if csv_file:
            csv_file.close()
        if writer_video is not None:
            writer_video.release()
        cv.destroyAllWindows()

    print()
    print(f"Frames processed   : {frame_idx}")
    print(f"Iterations         : {iteration}")
    if skipped_iterations:
        total = iteration + skipped_iterations
        print(
            f"Iterations skipped : {skipped_iterations} of {total} "
            f"({100.0 * skipped_iterations / total:.1f}%) — inference too slow"
        )
    for name, feed in feeds.items():
        src = feed.source
        if getattr(src, "live", False):
            print(
                f"Capture [{name}]     : {src.captured} frames, "
                f"{src.dropped} dropped before the buffer"
            )
            if src.reported_fps and not (1.0 < src.reported_fps < 240.0):
                print(
                    f"  NOTE: the driver reported {src.reported_fps} fps; "
                    f"{src.fps:.2f} was assumed instead. Timestamps in the CSV "
                    "depend on this value."
                )
    if args.csv:
        print(f"Trace              : {args.csv}")
    if scenario_runner:
        print(f"Injected scenario  : {scenario_name}")


if __name__ == "__main__":
    main()