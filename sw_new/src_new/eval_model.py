"""
Evaluacion detallada de un checkpoint: matriz de confusion, precision /
recall / F1 por clase, y macro-F1, a nivel WINDOW y a nivel SEGMENTO.

Ademas genera un CSV con una fila por segmento para inspeccionar exactamente
que segmentos del TEST se clasifican bien o mal y con que probabilidades.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset import SegmentDataset, segment_collate_fn
from model import get_model
from torch.utils.data import DataLoader


CLASS_NAMES = ["reaching", "safe", "unsafe"]
MACRO_NAMES_FINE = ["reaching", "safe", "unsafe"]


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def per_class_metrics(cm: np.ndarray):
    out = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        support = cm[c, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out.append({
            "class": c,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        })
    return out


def trivial_macro_f1(cm: np.ndarray) -> tuple:
    support = cm.sum(axis=1)
    total = support.sum()
    majority = int(np.argmax(support))
    p = support[majority] / total
    f1_majority = 2 * p / (1 + p) if p > 0 else 0.0
    return float(f1_majority / cm.shape[0]), majority


def format_report(cm: np.ndarray, title: str, class_names=None) -> str:
    num_classes = cm.shape[0]
    base = list(class_names) if class_names else CLASS_NAMES
    names = (base + [f"class_{i}" for i in range(num_classes)])[:num_classes]
    metrics = per_class_metrics(cm)
    macro_f1 = sum(m["f1"] for m in metrics) / num_classes
    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    floor, majority = trivial_macro_f1(cm)

    lines = ["", "=" * 72, f" {title}", "=" * 72, ""]
    lines.append("Matriz de confusion (filas = REAL, columnas = PREDICHO):")
    lines.append("")
    header = " " * 14 + "".join(f"{n:>12}" for n in names)
    lines.append(header)
    for i, n in enumerate(names):
        lines.append(f"{n:>12}  " + "".join(f"{cm[i, j]:>12,}" for j in range(num_classes)))

    lines.extend(["", "Normalizada por fila (que fraccion de cada clase real fue a cada prediccion):", "", header])
    for i, n in enumerate(names):
        total_row = cm[i, :].sum()
        row_vals = ["-"] * num_classes if total_row == 0 else [f"{cm[i, j] / total_row:.3f}" for j in range(num_classes)]
        lines.append(f"{n:>12}  " + "".join(f"{v:>12}" for v in row_vals))

    lines.extend(["", f"{'clase':>12}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>12}", "-" * 60])
    for m, n in zip(metrics, names):
        lines.append(f"{n:>12}{m['precision']:>12.4f}{m['recall']:>12.4f}{m['f1']:>12.4f}{m['support']:>12,}")

    lines.extend([
        "",
        f"  accuracy        : {accuracy:.4f}",
        f"  macro-F1        : {macro_f1:.4f}",
        f"  piso trivial    : {floor:.4f}  (predecir siempre '{names[majority]}')",
        f"  margen vs piso  : {macro_f1 - floor:+.4f}",
    ])

    minority = [i for i in range(num_classes) if i != majority]
    recalls_min = [metrics[i]["recall"] for i in minority]
    pred_counts = cm.sum(axis=0)
    frac_majority_pred = pred_counts[majority] / cm.sum() if cm.sum() > 0 else 0.0
    lines.extend(["", "Diagnostico:"])
    lines.append(
        f"  - {frac_majority_pred:.1%} de las predicciones fueron '{names[majority]}' "
        f"(esa clase es el {cm[majority, :].sum() / cm.sum():.1%} del split)"
    )
    if max(recalls_min) < 0.05:
        lines.append("  - COLAPSO: recall < 5% en TODAS las clases minoritarias.")
    elif macro_f1 - floor <= 0.01:
        lines.append("  - El macro-F1 no supera el piso trivial de forma significativa.")
    else:
        lines.append("  - NO hay colapso: el modelo supera el piso trivial y tiene recall no trivial en al menos una clase minoritaria.")
    return "\n".join(lines)


def infer_model_kwargs(state_dict: dict) -> dict:
    hidden_dim = state_dict["lstm.weight_ih_l0"].shape[0] // 4
    layer_ids = set()
    for k in state_dict:
        if k.startswith("lstm.weight_ih_l"):
            suffix = k[len("lstm.weight_ih_l"):]
            layer_ids.add(int(suffix.replace("_reverse", "")))
    lstm_layers = max(layer_ids) + 1
    classifier_keys = sorted(k for k in state_dict if k.startswith("classifier.") and k.endswith(".weight"))
    num_classes = state_dict[classifier_keys[-1]].shape[0]
    return {"hidden_dim": hidden_dim, "lstm_layers": lstm_layers, "num_classes": num_classes}


def _segment_name_parts(name):
    if isinstance(name, (tuple, list)):
        if len(name) >= 2:
            return str(name[0]), str(name[1])
        if len(name) == 1:
            return "", str(name[0])
    return "", str(name)


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes, max_windows_per_forward,
             label_groups=None, class_names=None):
    if label_groups is not None:
        from fine_labels import aggregate_probs, map_fine_to_macro, MACRO_CLASSES
        num_out = len(MACRO_CLASSES)
    else:
        num_out = num_classes

    names = list(class_names or [f"class_{i}" for i in range(num_out)])
    names = (names + [f"class_{i}" for i in range(num_out)])[:num_out]

    model.eval()
    win_preds, win_labels = [], []
    seg_preds, seg_labels = [], []
    segment_records = []

    for windows, label, name in tqdm(dataloader, desc="Evaluando"):
        num_windows = windows.shape[0]
        labels_expanded = label.repeat(num_windows).to(device)
        probs_acc = []

        for start in range(0, num_windows, max_windows_per_forward):
            chunk = windows[start:start + max_windows_per_forward].to(device, dtype=torch.float)
            outputs = model(chunk)
            if label_groups is None:
                probs = torch.softmax(outputs, dim=1)
            else:
                probs = aggregate_probs(outputs, label_groups)

            probs_acc.append(probs.cpu())
            win_preds.append(probs.argmax(1).cpu())

            chunk_lab = labels_expanded[start:start + max_windows_per_forward]
            if label_groups is not None:
                chunk_lab = map_fine_to_macro(chunk_lab, label_groups)
            win_labels.append(chunk_lab.cpu())

        probs_all = torch.cat(probs_acc, dim=0)
        window_pred = probs_all.argmax(dim=1)

        seg_prob = probs_all.mean(dim=0)
        pred_idx = int(seg_prob.argmax().item())
        lab = int(label.item())
        true_idx = label_groups[lab] if label_groups is not None else lab

        seg_preds.append(pred_idx)
        seg_labels.append(true_idx)

        session_name, segment_name = _segment_name_parts(name)
        record = {
            "session_name": session_name,
            "segment_name": segment_name,
            "true_idx": int(true_idx),
            "true_class": names[int(true_idx)],
            "pred_idx": int(pred_idx),
            "pred_class": names[int(pred_idx)],
            "correct": int(pred_idx == true_idx),
            "num_windows": int(num_windows),
            "confidence": float(seg_prob[pred_idx].item()),
        }

        for c, class_name in enumerate(names):
            col_name = class_name.replace(" ", "_")
            count_c = int((window_pred == c).sum().item())
            record[f"p_{col_name}"] = float(seg_prob[c].item())
            record[f"windows_pred_{col_name}"] = count_c
            record[f"frac_windows_pred_{col_name}"] = count_c / num_windows if num_windows > 0 else 0.0

        segment_records.append(record)

    win_preds_np = torch.cat(win_preds).numpy()
    win_labels_np = torch.cat(win_labels).numpy()
    cm_window = confusion_matrix(win_preds_np, win_labels_np, num_out)
    cm_segment = confusion_matrix(np.array(seg_preds), np.array(seg_labels), num_out)
    return cm_window, cm_segment, segment_records


def write_segment_csv(path, records):
    if not records:
        print("AVISO: no hay registros de segmento para guardar.")
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nPredicciones por segmento guardadas en: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint .pth")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="VALIDATION", choices=["TRAIN", "VALIDATION", "TEST"])
    parser.add_argument("--csv-out", default=None, help="Opcional: CSV con metricas por clase")
    parser.add_argument(
        "--segment-csv",
        default="segment_predictions.csv",
        help="CSV con una fila por segmento y probabilidades medias",
    )
    parser.add_argument(
        "--label-mode",
        default=None,
        help="macro | fine | binary_phone | ternary_clean",
    )
    parser.add_argument(
        "--partition-report",
        default=None,
        help="Ruta a partition_report.csv. Necesario para esquemas distintos de macro",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    inferred = infer_model_kwargs(state_dict)

    print(f"Checkpoint      : {args.checkpoint}")
    if "epoch" in ckpt:
        print(f"Epoca guardada  : {ckpt['epoch']}")
    for k in ("best_val_macro_f1", "best_val_acc"):
        if k in ckpt:
            print(f"{k:<16}: {ckpt[k]}")
    print(f"Arquitectura    : {inferred}  (inferida del state_dict)")
    print(f"Split evaluado  : {args.split}")

    num_classes = inferred.pop("num_classes")
    if num_classes != conf.num_classes:
        print(
            f"AVISO: el checkpoint tiene {num_classes} clases pero config.yaml "
            f"dice {conf.num_classes}. Se usa el del checkpoint."
        )

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    modo = args.label_mode or ("fine" if num_classes > 3 else "macro")
    if modo != "macro" and args.partition_report is None:
        raise SystemExit(
            f"El esquema {modo!r} necesita --partition-report para resolver las actividades del CSV."
        )

    dataset = SegmentDataset(
        os.path.join(conf.data_dir, args.split),
        sequence_length=conf.sequence_length,
        sample_one_each=conf.sample_one_each,
        transform=transform,
        label_mode=modo,
        partition_report=args.partition_report if modo != "macro" else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=segment_collate_fn,
    )

    model = get_model(num_classes, **inferred).to(args.device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"AVISO al cargar pesos -> faltantes: {missing} | inesperados: {unexpected}")

    grupos = dataset.label_groups if modo == "fine" else None
    class_names = list(getattr(dataset, "class_names", CLASS_NAMES))
    eval_class_names = list(MACRO_NAMES_FINE) if modo == "fine" else class_names

    cm_window, cm_segment, segment_records = evaluate(
        model,
        loader,
        args.device,
        num_classes,
        conf.max_windows_per_forward,
        label_groups=grupos,
        class_names=eval_class_names,
    )

    if modo == "fine":
        num_classes = len(MACRO_NAMES_FINE)
        class_names = list(MACRO_NAMES_FINE)

    print(format_report(cm_window, f"NIVEL WINDOW  ({args.split})", class_names))
    print(format_report(cm_segment, f"NIVEL SEGMENTO  ({args.split})", class_names))

    if args.segment_csv:
        write_segment_csv(args.segment_csv, segment_records)

    if args.csv_out:
        names = (class_names + [f"class_{i}" for i in range(num_classes)])[:num_classes]
        rows = ["nivel,clase,precision,recall,f1,support"]
        for nivel, cm in (("window", cm_window), ("segmento", cm_segment)):
            for m, n in zip(per_class_metrics(cm), names):
                rows.append(
                    f"{nivel},{n},{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['support']}"
                )
        Path(args.csv_out).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"\nMetricas por clase guardadas en: {args.csv_out}")


if __name__ == "__main__":
    main()