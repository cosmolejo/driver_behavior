"""
Verifica automaticamente los criterios del sanity check leyendo los scalars
que dejo TensorBoard. Pensado para encadenar corridas sin supervision:
devuelve codigo de salida 0 si paso, 1 si fallo.

Criterios
---------
1. train/epoch_macro_f1 >= val/macro_f1 en la ultima epoca.
   Es EL criterio: si el modelo no rinde mejor sobre los datos que vio,
   sigue habiendo una discrepancia train/eval.

2. min(train/epoch_loss) > umbral.
   Si la perdida de entrenamiento se desploma a cero, el modelo volvio a
   apoyarse en alguna muleta (BatchNorm u otra) en lugar de aprender.

3. max(val/macro_f1) > piso trivial.
   Advertencia, no fallo: el modelo podria estar sano mecanicamente pero
   sin superar al clasificador degenerado.

Uso
---
    python check_sanity.py --run runs/sanity_freeze_bn
    python check_sanity.py --run runs/sanity_freeze_bn --floor 0.2221
"""
import argparse
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(run_dir: str) -> dict:
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="carpeta del run de TensorBoard")
    parser.add_argument("--floor", type=float, default=0.2221,
                        help="piso trivial de macro-F1 a nivel window")
    parser.add_argument("--loss-min", type=float, default=0.005,
                        help="train/epoch_loss por debajo de esto = colapso")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"ERROR: no existe {run_dir}")
        return 1

    scalars = load_scalars(str(run_dir))

    requeridos = ["train/epoch_macro_f1", "val/macro_f1", "train/epoch_loss"]
    faltantes = [t for t in requeridos if t not in scalars]
    if faltantes:
        print(f"ERROR: faltan scalars {faltantes}")
        print(f"       disponibles: {sorted(scalars)}")
        print("       (¿el trainer.py en uso es el que loguea train/epoch_*?)")
        return 1

    train_f1 = scalars["train/epoch_macro_f1"]
    val_f1 = scalars["val/macro_f1"]
    train_loss = scalars["train/epoch_loss"]

    n_epocas = len(val_f1)
    train_f1_final = train_f1[-1][1]
    val_f1_final = val_f1[-1][1]
    val_f1_max = max(v for _, v in val_f1)
    train_loss_min = min(v for _, v in train_loss)
    brecha = train_f1_final - val_f1_final

    print("=" * 62)
    print(f" VERIFICACION DEL SANITY CHECK  ({run_dir.name})")
    print("=" * 62)
    print(f"  epocas completadas       : {n_epocas}")
    print(f"  train/epoch_macro_f1 fin : {train_f1_final:.4f}")
    print(f"  val/macro_f1 fin         : {val_f1_final:.4f}")
    print(f"  val/macro_f1 max         : {val_f1_max:.4f}   (piso trivial {args.floor:.4f})")
    print(f"  train/epoch_loss min     : {train_loss_min:.6f}")
    print(f"  brecha train - val       : {brecha:+.4f}")
    print()

    fallos = []

    if brecha < 0:
        fallos.append(
            f"train ({train_f1_final:.4f}) por DEBAJO de val ({val_f1_final:.4f}). "
            "Sigue habiendo discrepancia train/eval."
        )
    elif brecha < 0.02:
        print(f"  AVISO: brecha muy chica ({brecha:+.4f}). Puede ser normal con el")
        print("         backbone congelado y pocas epocas, pero conviene mirar las curvas.")

    if train_loss_min < args.loss_min:
        fallos.append(
            f"train/epoch_loss bajo a {train_loss_min:.6f}. "
            "El modelo volvio a apoyarse en alguna muleta."
        )

    if val_f1_max <= args.floor:
        print(f"  AVISO: val/macro_f1 max ({val_f1_max:.4f}) no supera el piso")
        print(f"         trivial ({args.floor:.4f}). Mecanica sana pero sin señal util.")

    if fallos:
        print("  RESULTADO: FALLO")
        for f in fallos:
            print(f"    - {f}")
        print()
        print("  No conviene encadenar los experimentos de unfreeze.")
        return 1

    print("  RESULTADO: PASA")
    print("  La mecanica train/eval quedo sana. Se puede continuar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())