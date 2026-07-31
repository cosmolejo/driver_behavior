"""
Barre TODOS los checkpoints por epoca de una o mas corridas y evalua cada
uno a nivel window y a nivel segmento.

Para que sirve
--------------
El checkpoint `model_best.pth` se elige por maximo de macro-F1 a nivel
WINDOW, que es la metrica desalineada con el objetivo de entrenamiento
(window bagging pondera cada segmento por igual, no cada ventana). Por eso
la epoca guardada no es necesariamente la mejor a nivel SEGMENTO, y comparar
dos corridas por sus `model_best.pth` mezcla el efecto real con el criterio
de seleccion.

Este script recorre los `model_<epoca>.pth` que quedaron en disco y calcula
las dos metricas para cada epoca, sin reentrenar nada. Con eso se puede
comparar el maximo REAL de cada corrida en la metrica que importa.

Uso
---
    python sweep_checkpoints.py --runs ../models/exp00_frozen ../models/exp02_last4
    python sweep_checkpoints.py --runs ../models/exp00_frozen --split TEST
    python sweep_checkpoints.py --runs ... --csv-out barrido.csv
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset import SegmentDataset, segment_collate_fn
from model import get_model
from eval_model import (
    infer_model_kwargs,
    per_class_metrics,
    trivial_macro_f1,
    format_report,
    evaluate,
    CLASS_NAMES,
)


def find_checkpoints(run_dir: Path):
    """
    Devuelve [(epoca, ruta)] ordenado por epoca, solo de los model_<n>.pth.
    Excluye model_best.pth (es una copia de una de esas epocas).
    """
    out = []
    for p in run_dir.glob("model_*.pth"):
        m = re.fullmatch(r"model_(\d+)\.pth", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def macro_f1_from_cm(cm: np.ndarray) -> float:
    return sum(m["f1"] for m in per_class_metrics(cm)) / cm.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="carpetas de checkpoints, ej: ../models/exp00_frozen")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="VALIDATION",
                        choices=["TRAIN", "VALIDATION", "TEST"])
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--full-report", action="store_true",
                        help="imprime el reporte completo del mejor checkpoint por segmento")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    # El dataset y el DataLoader se construyen UNA sola vez y se reusan para
    # todos los checkpoints. persistent_workers evita respawnear los workers
    # en cada evaluacion.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = SegmentDataset(
        os.path.join(conf.data_dir, args.split),
        sequence_length=conf.sequence_length,
        sample_one_each=conf.sample_one_each,
        transform=transform,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=segment_collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Split: {args.split}  ({len(dataset)} segmentos)\n")

    resultados = {}   # run -> [(epoca, f1_win, f1_seg, cm_win, cm_seg)]

    for run in args.runs:
        run_dir = Path(run)
        nombre = run_dir.name
        ckpts = find_checkpoints(run_dir)

        if not ckpts:
            print(f"[{nombre}] sin checkpoints model_<n>.pth. Se omite.")
            continue

        # La arquitectura es la misma en toda la corrida: se infiere una vez
        primer_sd = torch.load(ckpts[0][1], map_location="cpu")
        primer_sd = primer_sd.get("model_state_dict", primer_sd)
        inferido = infer_model_kwargs(primer_sd)
        num_classes = inferido.pop("num_classes")
        model = get_model(num_classes, **inferido).to(args.device)

        # Que epoca corresponde a model_best.pth
        best_path = run_dir / "model_best.pth"
        epoca_best = None
        if best_path.exists():
            b = torch.load(best_path, map_location="cpu")
            epoca_best = b.get("epoch")

        print(f"[{nombre}] {len(ckpts)} checkpoints  |  arquitectura {inferido} "
              f"+ {num_classes} clases  |  model_best = epoca {epoca_best}")

        filas = []
        for epoca, path in ckpts:
            sd = torch.load(path, map_location="cpu")
            sd = sd.get("model_state_dict", sd)
            model.load_state_dict(sd, strict=False)

            cm_w, cm_s = evaluate(
                model, loader, args.device, num_classes,
                conf.max_windows_per_forward,
            )
            f1_w = macro_f1_from_cm(cm_w)
            f1_s = macro_f1_from_cm(cm_s)
            filas.append((epoca, f1_w, f1_s, cm_w, cm_s))
            print(f"    epoca {epoca:>2}:  window {f1_w:.4f}   segmento {f1_s:.4f}")

        resultados[nombre] = filas
        print()

    if not resultados:
        print("No se evaluo ningun checkpoint.")
        return

    # -----------------------------------------------------------------
    # Tabla comparativa
    # -----------------------------------------------------------------
    print("=" * 72)
    print(f" BARRIDO DE CHECKPOINTS  ({args.split})")
    print("=" * 72)

    epocas = sorted({e for filas in resultados.values() for e, *_ in filas})
    nombres = list(resultados)

    encabezado = f"{'epoca':>6}"
    for n in nombres:
        encabezado += f"{n[:14] + ' win':>20}{n[:14] + ' seg':>20}"
    print(encabezado)
    print("-" * len(encabezado))

    # maximos por corrida, para marcarlos
    max_w = {n: max(f[1] for f in filas) for n, filas in resultados.items()}
    max_s = {n: max(f[2] for f in filas) for n, filas in resultados.items()}

    for ep in epocas:
        linea = f"{ep:>6}"
        for n in nombres:
            fila = next((f for f in resultados[n] if f[0] == ep), None)
            if fila is None:
                linea += f"{'-':>20}{'-':>20}"
            else:
                mw = " *" if abs(fila[1] - max_w[n]) < 1e-12 else "  "
                ms = " *" if abs(fila[2] - max_s[n]) < 1e-12 else "  "
                linea += f"{fila[1]:>18.4f}{mw}{fila[2]:>18.4f}{ms}"
        print(linea)

    print()
    print("(* = maximo de esa columna)")
    print()
    print("-" * 72)
    print(f"{'corrida':<24}{'max window':>14}{'max segmento':>16}{'ep. seg':>10}")
    print("-" * 72)
    for n, filas in resultados.items():
        mejor_seg = max(filas, key=lambda f: f[2])
        print(f"{n:<24}{max_w[n]:>14.4f}{max_s[n]:>16.4f}{mejor_seg[0]:>10}")

    # -----------------------------------------------------------------
    # Comparacion directa si hay exactamente dos corridas
    # -----------------------------------------------------------------
    if len(nombres) == 2:
        a, b = nombres
        d_w = max_w[b] - max_w[a]
        d_s = max_s[b] - max_s[a]
        print()
        print(f"Diferencia ({b} - {a}):")
        print(f"  window   : {d_w:+.4f}")
        print(f"  segmento : {d_s:+.4f}")
        print()
        if abs(d_s) < 0.02:
            print("  El delta a nivel segmento esta dentro del ruido tipico entre")
            print("  epocas (~0.03). Las dos corridas son equivalentes en la metrica")
            print("  alineada con el objetivo de entrenamiento.")
        elif d_s > 0:
            print(f"  {b} supera a {a} a nivel segmento.")
        else:
            print(f"  {a} supera a {b} a nivel segmento.")

    # -----------------------------------------------------------------
    # Reporte completo del mejor checkpoint por segmento
    # -----------------------------------------------------------------
    if args.full_report:
        for n, filas in resultados.items():
            mejor = max(filas, key=lambda f: f[2])
            print(format_report(mejor[4], f"{n} - epoca {mejor[0]} - NIVEL SEGMENTO"))

    # -----------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------
    if args.csv_out:
        lineas = ["corrida,epoca,macro_f1_window,macro_f1_segmento"]
        for n, filas in resultados.items():
            for ep, f1w, f1s, _, _ in filas:
                lineas.append(f"{n},{ep},{f1w:.6f},{f1s:.6f}")
        Path(args.csv_out).write_text("\n".join(lineas) + "\n")
        print(f"\nCSV guardado en: {args.csv_out}")


if __name__ == "__main__":
    main()