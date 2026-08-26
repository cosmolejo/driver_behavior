"""
Evaluaci�n detallada de un checkpoint: matriz de confusi�n, precision /
recall / F1 por clase, y macro-F1, a nivel WINDOW y a nivel SEGMENTO.

Objetivo principal
------------------
Distinguir entre dos escenarios que producen un macro-F1 parecido pero
significan cosas muy distintas:

  (a) COLAPSO: el modelo predice casi todo como la clase mayoritaria.
      El macro-F1 se pega al "piso trivial" y las minoritarias tienen
      recall ~0. El modelo no aprendi� nada �til sobre ellas.

  (b) Performance pobre pero REAL: el modelo distingue las tres clases,
      solo que con muchos errores. Recall > 0 en las minoritarias.

El script imprime el piso trivial calculado sobre la distribuci�n real
del split para que la comparaci�n sea directa.

Uso
---
    python eval_model.py --checkpoint models/<run>/model_best.pth
    python eval_model.py --checkpoint models/<run>/model_best.pth --split TEST
    python eval_model.py --checkpoint ... --csv-out resultados.csv

La arquitectura (hidden_dim, lstm_layers, num_classes) se infiere del
state_dict del checkpoint, as� que no hace falta pasar los hiperpar�metros
del trial a mano ni arriesgar un mismatch silencioso.
"""
import argparse
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


# Orden de las macro-clases VERIFICADO contra partition_report.csv.
#
# Los indices siguen el orden ALFABETICO de las etiquetas, no el orden en
# que suelen enunciarse (safe / reaching / unsafe):
#
#     0 = reaching     1 = safe     2 = unsafe
#
# Verificacion (VALIDATION): la cantidad de ventanas por segmento tiene que
# crecer con la duracion del segmento, y solo cierra con este orden:
#     clase 0: 195 seg, 76.9 fr de media  ->  6.21 win/seg
#     clase 1: 250 seg, 171.0 fr          -> 18.18 win/seg
#     clase 2: 169 seg, 294.6 fr          -> 34.00 win/seg
CLASS_NAMES = ["reaching", "safe", "unsafe"]

# Nombres de las macro-clases a las que agrega el esquema "fine".
MACRO_NAMES_FINE = ["reaching", "safe", "unsafe"]


# ---------------------------------------------------------------------
# M�tricas (implementadas a mano, sin sklearn, igual que en trainer.py)
# ---------------------------------------------------------------------
def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Devuelve una matriz (num_classes, num_classes) donde
    cm[i, j] = cantidad de muestras con label real i predichas como j.
    Filas = verdad, columnas = predicci�n.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def per_class_metrics(cm: np.ndarray):
    """
    Deriva precision / recall / F1 / support por clase desde la matriz de
    confusi�n. Divisiones por cero -> 0.0 (equivalente a zero_division=0).
    """
    num_classes = cm.shape[0]
    out = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        support = cm[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        out.append(
            {
                "class": c,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )
    return out


def trivial_macro_f1(cm: np.ndarray) -> tuple:
    """
    Macro-F1 de un clasificador degenerado que predice SIEMPRE la clase
    mayoritaria del split. Es el piso contra el cual hay que comparar:
    un macro-F1 igual o menor a este valor significa que el modelo no
    aporta nada sobre la estrategia trivial.
    """
    support = cm.sum(axis=1)
    total = support.sum()
    majority = int(np.argmax(support))

    # Para la clase mayoritaria: recall=1, precision=proporci�n de esa clase
    p = support[majority] / total
    f1_majority = 2 * p / (1 + p) if p > 0 else 0.0
    # El resto de las clases: F1 = 0
    return float(f1_majority / cm.shape[0]), majority


def format_report(cm: np.ndarray, title: str, class_names=None) -> str:
    num_classes = cm.shape[0]
    base = list(class_names) if class_names else CLASS_NAMES
    names = (base + [f"class_{i}" for i in range(num_classes)])[:num_classes]
    metrics = per_class_metrics(cm)
    macro_f1 = sum(m["f1"] for m in metrics) / num_classes
    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    floor, majority = trivial_macro_f1(cm)

    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f" {title}")
    lines.append("=" * 72)

    # --- Matriz de confusi�n (conteos) ---
    lines.append("")
    lines.append("Matriz de confusion (filas = REAL, columnas = PREDICHO):")
    lines.append("")
    header = " " * 14 + "".join(f"{n:>12}" for n in names)
    lines.append(header)
    for i, n in enumerate(names):
        row = f"{n:>12}  " + "".join(f"{cm[i, j]:>12,}" for j in range(num_classes))
        lines.append(row)

    # --- Matriz normalizada por fila (= recall por clase en la diagonal) ---
    lines.append("")
    lines.append("Normalizada por fila (que fraccion de cada clase real fue a cada prediccion):")
    lines.append("")
    lines.append(header)
    for i, n in enumerate(names):
        total_row = cm[i, :].sum()
        if total_row == 0:
            row_vals = ["-"] * num_classes
        else:
            row_vals = [f"{cm[i, j] / total_row:.3f}" for j in range(num_classes)]
        lines.append(f"{n:>12}  " + "".join(f"{v:>12}" for v in row_vals))

    # --- M�tricas por clase ---
    lines.append("")
    lines.append(f"{'clase':>12}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>12}")
    lines.append("-" * 60)
    for m, n in zip(metrics, names):
        lines.append(
            f"{n:>12}{m['precision']:>12.4f}{m['recall']:>12.4f}"
            f"{m['f1']:>12.4f}{m['support']:>12,}"
        )

    lines.append("")
    lines.append(f"  accuracy        : {accuracy:.4f}")
    lines.append(f"  macro-F1        : {macro_f1:.4f}")
    lines.append(f"  piso trivial    : {floor:.4f}  (predecir siempre '{names[majority]}')")

    delta = macro_f1 - floor
    lines.append(f"  margen vs piso  : {delta:+.4f}")

    # --- Diagn�stico autom�tico de colapso ---
    minority = [i for i in range(num_classes) if i != majority]
    recalls_min = [metrics[i]["recall"] for i in minority]
    pred_counts = cm.sum(axis=0)
    frac_majority_pred = pred_counts[majority] / cm.sum() if cm.sum() > 0 else 0.0

    lines.append("")
    lines.append("Diagnostico:")
    lines.append(
        f"  - {frac_majority_pred:.1%} de las predicciones fueron '{names[majority]}' "
        f"(esa clase es el {cm[majority, :].sum() / cm.sum():.1%} del split)"
    )
    if max(recalls_min) < 0.05:
        lines.append("  - COLAPSO: recall < 5% en TODAS las clases minoritarias.")
        lines.append("    El modelo no esta aprendiendo las minoritarias.")
    elif delta <= 0.01:
        lines.append("  - El macro-F1 no supera el piso trivial de forma significativa.")
        lines.append("    Hay algo de se�al en las minoritarias, pero no alcanza.")
    else:
        lines.append("  - NO hay colapso: el modelo supera el piso trivial y tiene")
        lines.append("    recall no trivial en al menos una clase minoritaria.")
        lines.append("    La performance es baja pero real -> tiene sentido atacar")
        lines.append("    el overfitting (augmentation / regularizacion).")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Inferencia de la arquitectura desde el checkpoint
# ---------------------------------------------------------------------
def infer_model_kwargs(state_dict: dict) -> dict:
    """
    Deduce hidden_dim, lstm_layers y num_classes del state_dict, para no
    depender de que el usuario recuerde los hiperpar�metros del trial.
    (dropout y freeze_backbone no afectan la evaluacion en model.eval()).
    """
    # LSTM bidireccional: weight_ih_l{i} tiene shape (4*hidden, input)
    hidden_dim = state_dict["lstm.weight_ih_l0"].shape[0] // 4

    layer_ids = set()
    for k in state_dict:
        if k.startswith("lstm.weight_ih_l"):
            suffix = k[len("lstm.weight_ih_l"):]
            layer_ids.add(int(suffix.replace("_reverse", "")))
    lstm_layers = max(layer_ids) + 1

    # Ultimo Linear del classifier -> num_classes
    classifier_keys = sorted(
        k for k in state_dict if k.startswith("classifier.") and k.endswith(".weight")
    )
    num_classes = state_dict[classifier_keys[-1]].shape[0]

    return {
        "hidden_dim": hidden_dim,
        "lstm_layers": lstm_layers,
        "num_classes": num_classes,
    }


# ---------------------------------------------------------------------
# Evaluaci�n
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device, num_classes, max_windows_per_forward,
             label_groups=None):
    """
    Devuelve (cm_window, cm_segment).

    - Nivel WINDOW: cada ventana es una muestra independiente. Es la
      granularidad que usan val/accuracy y val/macro_f1 en trainer.py,
      asi que estos numeros deberian coincidir con TensorBoard.
    - Nivel SEGMENTO: se promedian las probabilidades softmax de todas las
      ventanas del segmento y se toma el argmax. Es la unidad real de
      etiquetado y la que importa para la aplicacion final.

    Si `label_groups` no es None, el modelo predice CLASES FINAS y las
    matrices se devuelven ya agregadas a macro-clases: las probabilidades
    de las componentes de cada grupo se suman (P(unsafe) = suma de las 6
    actividades que la componen) y las etiquetas finas se traducen a macro.
    Asi el resultado es comparable con el de un modelo entrenado a 3 clases.
    """
    if label_groups is not None:
        from fine_labels import aggregate_probs, map_fine_to_macro, MACRO_CLASSES
        num_out = len(MACRO_CLASSES)
    else:
        num_out = num_classes
    model.eval()

    win_preds, win_labels = [], []
    seg_preds, seg_labels = [], []

    for windows, label, _ in tqdm(dataloader, desc="Evaluando"):
        num_windows = windows.shape[0]
        labels_expanded = label.repeat(num_windows).to(device)

        probs_acc = []
        for start in range(0, num_windows, max_windows_per_forward):
            chunk = windows[start:start + max_windows_per_forward].to(device, dtype=torch.float)
            outputs = model(chunk)
            if label_groups is None:
                probs = torch.softmax(outputs, dim=1)
            else:
                # Suma de probabilidades por grupo: la probabilidad de la
                # union de eventos excluyentes es la suma de las suyas.
                probs = aggregate_probs(outputs, label_groups)
            probs_acc.append(probs.cpu())

            win_preds.append(probs.argmax(1).cpu())
            chunk_lab = labels_expanded[start:start + max_windows_per_forward]
            if label_groups is not None:
                chunk_lab = map_fine_to_macro(chunk_lab, label_groups)
            win_labels.append(chunk_lab.cpu())

        # Agregacion a nivel segmento: promedio de probabilidades
        seg_prob = torch.cat(probs_acc).mean(dim=0)
        seg_preds.append(int(seg_prob.argmax().item()))
        lab = int(label.item())
        seg_labels.append(label_groups[lab] if label_groups is not None else lab)

    win_preds = torch.cat(win_preds).numpy()
    win_labels = torch.cat(win_labels).numpy()

    cm_window = confusion_matrix(win_preds, win_labels, num_out)
    cm_segment = confusion_matrix(
        np.array(seg_preds), np.array(seg_labels), num_out
    )
    return cm_window, cm_segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="ruta a model_best.pth")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--split", default="VALIDATION", choices=["TRAIN", "VALIDATION", "TEST"]
    )
    parser.add_argument("--csv-out", default=None, help="opcional: guarda metricas por clase")
    parser.add_argument("--label-mode", default=None,
                        help="esquema de etiquetado usado al entrenar: "
                             "macro | fine | binary_phone | ternary_clean. "
                             "Si se omite se infiere del numero de salidas.")
    parser.add_argument("--partition-report", default=None,
                        help="ruta a partition_report.csv. Necesario si el "
                             "checkpoint se entreno con 9 clases finas.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
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

    # MISMO transform que en entrenamiento (sin augmentation)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # El esquema explicito manda; si no se pasa, se infiere del numero de
    # salidas (solo distingue macro de fine, no los esquemas con mapa propio).
    modo = args.label_mode or ("fine" if num_classes > 3 else "macro")
    if modo != "macro" and args.partition_report is None:
        raise SystemExit(
        f"El esquema {modo!r} requiere --partition-report para "
        "recuperar la actividad de cada segmento."
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
    # Nombres de clase del esquema, para que las matrices salgan rotuladas
    # correctamente (en el esquema binario son ['safe', 'phone']).
    class_names = list(getattr(dataset, "class_names", CLASS_NAMES))
    cm_window, cm_segment = evaluate(
        model, loader, args.device, num_classes, conf.max_windows_per_forward,
        label_groups=grupos,
    )
    if modo == "fine":
        # Las matrices ya vienen agregadas a las macro-clases
        num_classes = len(MACRO_NAMES_FINE)
        class_names = list(MACRO_NAMES_FINE)

    print(format_report(cm_window, f"NIVEL WINDOW  ({args.split})", class_names))
    print(format_report(cm_segment, f"NIVEL SEGMENTO  ({args.split})", class_names))

    if args.csv_out:
        names = (class_names + [f"class_{i}" for i in range(num_classes)])[:num_classes]
        rows = ["nivel,clase,precision,recall,f1,support"]
        for nivel, cm in (("window", cm_window), ("segmento", cm_segment)):
            for m, n in zip(per_class_metrics(cm), names):
                rows.append(
                    f"{nivel},{n},{m['precision']:.6f},{m['recall']:.6f},"
                    f"{m['f1']:.6f},{m['support']}"
                )
        Path(args.csv_out).write_text("\n".join(rows) + "\n")
        print(f"\nMetricas por clase guardadas en: {args.csv_out}")


if __name__ == "__main__":
    main()