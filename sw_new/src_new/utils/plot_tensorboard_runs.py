#!/usr/bin/env python3
"""
plot_tensorboard_runs.py

Generate thesis-ready training figures and CSV summaries directly from
TensorBoard event files.

Examples
--------
1) Final phone classifier: train vs validation, window vs segment, loss and LR

python plot_tensorboard_runs.py \
  --run "Final phone=/path/to/events.out.tfevents...." \
  --title "Final phone-use classifier" \
  --output-dir figures/final_phone \
  --reference-line 0.434 \
  --reference-label "Trivial macro-F1 floor"

2) Compare several runs

python plot_tensorboard_runs.py \
  --run "Frozen=/path/lad00/events.out.tfevents..." \
  --run "2 blocks=/path/lad02/events.out.tfevents..." \
  --run "4 blocks=/path/lad04/events.out.tfevents..." \
  --run "8 blocks=/path/lad08/events.out.tfevents..." \
  --run "All blocks=/path/lad17/events.out.tfevents..." \
  --title "Backbone unfreezing experiment" \
  --output-dir figures/unfreezing \
  --plateau 5:11

3) Learning curve by number of subjects

python plot_tensorboard_runs.py \
  --run "5 subjects=/path/lc_s05/events.out.tfevents..." \
  --run "8 subjects=/path/lc_s08/events.out.tfevents..." \
  --run "10 subjects=/path/lc_s10/events.out.tfevents..." \
  --run "15 subjects=/path/lc_s15/events.out.tfevents..." \
  --run "18 subjects=/path/lc_s18/events.out.tfevents..." \
  --run "21 subjects=/path/lc_s21/events.out.tfevents..." \
  --x-values 5 8 10 15 18 21 \
  --x-label "Training subjects" \
  --summary-stat plateau_mean \
  --plateau 5:11 \
  --title "Effect of training-subject diversity" \
  --output-dir figures/subjects

Input
-----
Each --run argument must have the form:

    "Display label=/path/to/event_file_or_run_directory"

A directory may contain one or more events.out.tfevents.* files. Scalar events
with the same TensorBoard step are deduplicated by keeping the latest event.

Expected tags from the current trainer
--------------------------------------
train/epoch_loss
train/epoch_accuracy
train/epoch_macro_f1
val/loss
val/accuracy
val/macro_f1
val/macro_f1_segment
val/accuracy_segment
train/lr

The script tolerates missing tags and prints a warning instead of failing.

Outputs
-------
For a single run, when the corresponding tags exist:
  - train_val_macro_f1.{png,pdf}
  - validation_window_vs_segment_macro_f1.{png,pdf}
  - train_val_loss.{png,pdf}
  - learning_rate.{png,pdf}

For multiple runs:
  - comparison_val_segment_macro_f1.{png,pdf}
  - summary_val_segment_macro_f1.{png,pdf}

When --x-values is supplied:
  - summary_vs_x.{png,pdf}

Always:
  - tensorboard_summary.csv
  - one extracted scalar CSV per run

Notes
-----
- Epoch numbers are reconstructed from the order of epoch-level scalar events
  (0, 1, 2, ...). This is intentional because the trainer logs epoch metrics at
  optimizer global_step rather than using the epoch number as TensorBoard step.
- `--plateau A:B` is inclusive, so `--plateau 5:11` uses epochs 5 through 11.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:
    raise SystemExit(
        "TensorBoard is required. Install it with:\n"
        "  pip install tensorboard\n"
    ) from exc


TAGS = [
    "train/epoch_loss",
    "train/epoch_accuracy",
    "train/epoch_macro_f1",
    "val/loss",
    "val/accuracy",
    "val/macro_f1",
    "val/macro_f1_segment",
    "val/accuracy_segment",
    "train/lr",
]

TAG_SHORT = {
    "train/epoch_loss": "train_loss",
    "train/epoch_accuracy": "train_accuracy",
    "train/epoch_macro_f1": "train_macro_f1",
    "val/loss": "val_loss",
    "val/accuracy": "val_accuracy",
    "val/macro_f1": "val_macro_f1_window",
    "val/macro_f1_segment": "val_macro_f1_segment",
    "val/accuracy_segment": "val_accuracy_segment",
    "train/lr": "learning_rate",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate thesis figures from TensorBoard event files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Run label and TensorBoard event file/directory. "
            "Repeat this option for multiple runs."
        ),
    )
    p.add_argument(
        "--title",
        default="Training experiment",
        help="Base title used in all generated figures.",
    )
    p.add_argument(
        "--output-dir",
        default="tensorboard_figures",
        help="Directory where figures and CSV files are written.",
    )
    p.add_argument(
        "--plateau",
        default="5:11",
        metavar="START:END",
        help="Inclusive epoch range used for plateau_mean (default: 5:11).",
    )
    p.add_argument(
        "--summary-stat",
        choices=["max", "plateau_mean", "last"],
        default="max",
        help="Statistic used in the run-summary figure (default: max).",
    )
    p.add_argument(
        "--x-values",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional numeric x values, one per --run, for an ordered summary "
            "curve (e.g. number of training subjects)."
        ),
    )
    p.add_argument(
        "--x-label",
        default="Experimental condition",
        help="X-axis label used with --x-values.",
    )
    p.add_argument(
        "--reference-line",
        type=float,
        default=None,
        help="Optional horizontal reference value, e.g. trivial macro-F1 floor.",
    )
    p.add_argument(
        "--reference-label",
        default="Reference",
        help="Label for --reference-line.",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png", "pdf"],
        help="Figure formats to save (default: png pdf).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster output (default: 300).",
    )
    return p.parse_args()


def parse_plateau(spec: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", spec)
    if not m:
        raise SystemExit("--plateau must have format START:END, e.g. 5:11")
    a, b = map(int, m.groups())
    if b < a:
        raise SystemExit("--plateau END must be >= START")
    return a, b


def parse_run_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(
            f"Invalid --run {spec!r}. Expected LABEL=/path/to/events_or_directory"
        )
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise SystemExit(f"Empty run label in {spec!r}")
    if not path.exists():
        raise SystemExit(f"TensorBoard path does not exist: {path}")
    return label, path


def event_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.rglob("events.out.tfevents.*"))
    if not files:
        raise SystemExit(f"No TensorBoard event files found under: {path}")
    return files


def load_run(path: Path) -> Dict[str, List[float]]:
    """
    Load selected scalar tags.

    Multiple event files are supported. Duplicate TensorBoard global steps are
    resolved by keeping the event with the latest wall_time.
    """
    by_tag: Dict[str, Dict[int, Tuple[float, float]]] = {
        tag: {} for tag in TAGS
    }

    for file in event_files(path):
        ea = EventAccumulator(
            str(file),
            size_guidance={"scalars": 0},
        )
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))

        for tag in TAGS:
            if tag not in available:
                continue
            for ev in ea.Scalars(tag):
                old = by_tag[tag].get(ev.step)
                if old is None or ev.wall_time >= old[0]:
                    by_tag[tag][ev.step] = (ev.wall_time, float(ev.value))

    data: Dict[str, List[float]] = {}
    for tag, events in by_tag.items():
        ordered = [events[step][1] for step in sorted(events)]
        if ordered:
            data[tag] = ordered
    return data


def align(a: List[float], b: List[float]) -> Tuple[List[int], List[float], List[float]]:
    n = min(len(a), len(b))
    return list(range(n)), a[:n], b[:n]


def finite(values: Iterable[float]) -> List[float]:
    return [v for v in values if math.isfinite(v)]


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else math.nan


def summary_stat(values: List[float], stat: str, plateau: Tuple[int, int]) -> float:
    vals = finite(values)
    if not vals:
        return math.nan

    if stat == "max":
        return max(vals)
    if stat == "last":
        return vals[-1]

    start, end = plateau
    selected = [
        values[i]
        for i in range(start, min(end + 1, len(values)))
        if math.isfinite(values[i])
    ]
    return mean(selected)


def summary_stat_label(stat: str, plateau: Tuple[int, int]) -> str:
    if stat == "max":
        return "Maximum validation segment macro-F1"
    if stat == "last":
        return "Final validation segment macro-F1"
    return f"Mean validation segment macro-F1 (epochs {plateau[0]}–{plateau[1]})"


def safe_slug(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return slug.strip("_") or "run"


def save_figure(fig, outdir: Path, stem: str, formats: List[str], dpi: int) -> None:
    fig.tight_layout()
    for fmt in formats:
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(outdir / f"{stem}.{fmt}", **kwargs)
    plt.close(fig)


def add_reference(ax, value: float | None, label: str) -> None:
    if value is not None:
        ax.axhline(value, label=label)


def legend_if_needed(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()


def single_run_figures(
    label: str,
    data: Dict[str, List[float]],
    title: str,
    outdir: Path,
    formats: List[str],
    dpi: int,
    reference_line: float | None,
    reference_label: str,
) -> None:
    # Train vs validation segment macro-F1
    t = data.get("train/epoch_macro_f1")
    vseg = data.get("val/macro_f1_segment")
    if t and vseg:
        epochs, t2, v2 = align(t, vseg)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(epochs, t2, marker="o", label="Train macro-F1")
        ax.plot(epochs, v2, marker="o", label="Validation segment macro-F1")
        add_reference(ax, reference_line, reference_label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{title}\nTraining and validation performance")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        legend_if_needed(ax)
        save_figure(fig, outdir, "train_val_macro_f1", formats, dpi)

    # Window vs segment validation macro-F1
    vwin = data.get("val/macro_f1")
    if vwin and vseg:
        epochs, w2, s2 = align(vwin, vseg)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(epochs, w2, marker="o", label="Window macro-F1")
        ax.plot(epochs, s2, marker="o", label="Segment macro-F1")
        add_reference(ax, reference_line, reference_label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{title}\nValidation granularity")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        legend_if_needed(ax)
        save_figure(
            fig, outdir, "validation_window_vs_segment_macro_f1", formats, dpi
        )

    # Train vs validation loss
    tloss = data.get("train/epoch_loss")
    vloss = data.get("val/loss")
    if tloss and vloss:
        epochs, t2, v2 = align(tloss, vloss)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(epochs, t2, marker="o", label="Train loss")
        ax.plot(epochs, v2, marker="o", label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{title}\nTraining and validation loss")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        legend_if_needed(ax)
        save_figure(fig, outdir, "train_val_loss", formats, dpi)

    # Learning rate
    lr = data.get("train/lr")
    if lr:
        epochs = list(range(len(lr)))
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(epochs, lr, marker="o", label="Learning rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title(f"{title}\nLearning-rate schedule")
        ax.grid(True, alpha=0.25)
        legend_if_needed(ax)
        save_figure(fig, outdir, "learning_rate", formats, dpi)


def multi_run_figures(
    runs: List[Tuple[str, Dict[str, List[float]]]],
    title: str,
    outdir: Path,
    formats: List[str],
    dpi: int,
    stat: str,
    plateau: Tuple[int, int],
    x_values: List[float] | None,
    x_label: str,
    reference_line: float | None,
    reference_label: str,
) -> None:
    # Overlay validation segment macro-F1 curves.
    valid_runs = [
        (label, data["val/macro_f1_segment"])
        for label, data in runs
        if data.get("val/macro_f1_segment")
    ]

    if valid_runs:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for label, values in valid_runs:
            ax.plot(
                list(range(len(values))),
                values,
                marker="o",
                label=label,
            )
        add_reference(ax, reference_line, reference_label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation segment macro-F1")
        ax.set_title(f"{title}\nValidation segment macro-F1 by run")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        legend_if_needed(ax)
        save_figure(
            fig,
            outdir,
            "comparison_val_segment_macro_f1",
            formats,
            dpi,
        )

    # Summary statistic by run as a categorical bar chart.
    labels = []
    values = []
    for label, data in runs:
        vals = data.get("val/macro_f1_segment", [])
        if vals:
            labels.append(label)
            values.append(summary_stat(vals, stat, plateau))

    if labels:
        fig, ax = plt.subplots(figsize=(max(7.2, 1.05 * len(labels)), 4.8))
        positions = list(range(len(labels)))
        ax.bar(positions, values)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        add_reference(ax, reference_line, reference_label)
        ax.set_ylabel(summary_stat_label(stat, plateau))
        ax.set_title(f"{title}\nRun summary")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.25)
        legend_if_needed(ax)
        save_figure(
            fig,
            outdir,
            "summary_val_segment_macro_f1",
            formats,
            dpi,
        )

    # Ordered numeric x-axis summary, useful for subject learning curve.
    if x_values is not None:
        if len(x_values) != len(runs):
            raise SystemExit(
                f"--x-values has {len(x_values)} values but {len(runs)} runs "
                "were supplied."
            )

        points = []
        for x, (label, data) in zip(x_values, runs):
            vals = data.get("val/macro_f1_segment", [])
            if vals:
                points.append((x, summary_stat(vals, stat, plateau), label))

        if points:
            points.sort(key=lambda row: row[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            ax.plot(xs, ys, marker="o")
            add_reference(ax, reference_line, reference_label)
            ax.set_xlabel(x_label)
            ax.set_ylabel(summary_stat_label(stat, plateau))
            ax.set_title(title)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)
            legend_if_needed(ax)
            save_figure(fig, outdir, "summary_vs_x", formats, dpi)


def write_run_scalars_csv(
    outdir: Path,
    label: str,
    data: Dict[str, List[float]],
) -> None:
    if not data:
        return

    max_len = max(len(values) for values in data.values())
    columns = ["epoch"] + [TAG_SHORT[tag] for tag in TAGS if tag in data]

    path = outdir / f"{safe_slug(label)}_scalars.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for epoch in range(max_len):
            row = {"epoch": epoch}
            for tag in TAGS:
                if tag not in data:
                    continue
                values = data[tag]
                row[TAG_SHORT[tag]] = values[epoch] if epoch < len(values) else ""
            writer.writerow(row)


def write_summary_csv(
    outdir: Path,
    runs: List[Tuple[str, Path, Dict[str, List[float]]]],
    plateau: Tuple[int, int],
) -> None:
    path = outdir / "tensorboard_summary.csv"
    fields = [
        "run",
        "source",
        "epochs",
        "max_val_segment_macro_f1",
        "max_val_segment_epoch",
        f"plateau_mean_epochs_{plateau[0]}_{plateau[1]}",
        "last_val_segment_macro_f1",
        "max_val_window_macro_f1",
        "last_train_macro_f1",
        "min_val_loss",
        "last_val_loss",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for label, source, data in runs:
            seg = data.get("val/macro_f1_segment", [])
            win = data.get("val/macro_f1", [])
            train = data.get("train/epoch_macro_f1", [])
            vloss = data.get("val/loss", [])

            if seg:
                max_seg = max(seg)
                max_epoch = seg.index(max_seg)
                plateau_value = summary_stat(seg, "plateau_mean", plateau)
                last_seg = seg[-1]
            else:
                max_seg = max_epoch = plateau_value = last_seg = ""

            row = {
                "run": label,
                "source": str(source),
                "epochs": max((len(v) for v in data.values()), default=0),
                "max_val_segment_macro_f1": max_seg,
                "max_val_segment_epoch": max_epoch,
                f"plateau_mean_epochs_{plateau[0]}_{plateau[1]}": plateau_value,
                "last_val_segment_macro_f1": last_seg,
                "max_val_window_macro_f1": max(win) if win else "",
                "last_train_macro_f1": train[-1] if train else "",
                "min_val_loss": min(vloss) if vloss else "",
                "last_val_loss": vloss[-1] if vloss else "",
            }
            writer.writerow(row)


def warn_missing(label: str, data: Dict[str, List[float]]) -> None:
    missing = [tag for tag in TAGS if tag not in data]
    if missing:
        print(
            f"WARNING [{label}]: missing tags: {', '.join(missing)}",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    plateau = parse_plateau(args.plateau)

    specs = [parse_run_spec(spec) for spec in args.run]
    if args.x_values is not None and len(args.x_values) != len(specs):
        raise SystemExit(
            "--x-values must contain exactly one value for each --run."
        )

    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    loaded: List[Tuple[str, Path, Dict[str, List[float]]]] = []

    for label, path in specs:
        print(f"Loading {label}: {path}")
        data = load_run(path)
        if not data:
            print(
                f"WARNING [{label}]: none of the expected scalar tags were found.",
                file=sys.stderr,
            )
        warn_missing(label, data)
        write_run_scalars_csv(outdir, label, data)
        loaded.append((label, path, data))

    write_summary_csv(outdir, loaded, plateau)

    if len(loaded) == 1:
        label, _, data = loaded[0]
        single_run_figures(
            label=label,
            data=data,
            title=args.title,
            outdir=outdir,
            formats=args.formats,
            dpi=args.dpi,
            reference_line=args.reference_line,
            reference_label=args.reference_label,
        )
    else:
        multi_run_figures(
            runs=[(label, data) for label, _, data in loaded],
            title=args.title,
            outdir=outdir,
            formats=args.formats,
            dpi=args.dpi,
            stat=args.summary_stat,
            plateau=plateau,
            x_values=args.x_values,
            x_label=args.x_label,
            reference_line=args.reference_line,
            reference_label=args.reference_label,
        )

    print(f"\nDone. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
