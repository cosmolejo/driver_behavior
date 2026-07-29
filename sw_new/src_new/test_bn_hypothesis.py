"""
Test decisivo: ¿la brecha entre `train/loss ~ 0` en TensorBoard y el ~43%
de accuracy al evaluar el checkpoint sobre TRAIN se explica por BatchNorm?

Hipotesis
---------
Cada forward procesa 8 windows x 32 frames = 256 frames del MISMO segmento
(mismo sujeto, misma actividad, misma iluminacion). En `model.train()`,
BatchNorm normaliza con las estadisticas de ese batch, es decir hace
normalizacion POR SEGMENTO: le filtra al modelo informacion del propio
segmento que esta clasificando. En `model.eval()` usa las running_stats
globales y esa muleta desaparece.

Si la hipotesis es correcta, sobre los MISMOS datos de train se deberia ver:

    modo train  -> accuracy alta (consistente con train/loss ~ 0)
    modo eval   -> accuracy baja (~0.43, lo que reporto eval_model.py)

Se evalua un tercer modo intermedio (BN en eval, resto en train) para
aislar que el causante es BatchNorm y no Dropout.

IMPORTANTE: correr un forward en modo train MODIFICA las running_stats de
BatchNorm. El script recarga el state_dict antes de cada modo para que las
mediciones sean independientes y no se contaminen entre si.

Uso
---
    python test_bn_hypothesis.py --checkpoint ../models/best_full_retrain/model_best.pth
    python test_bn_hypothesis.py --checkpoint ... --num-segments 150
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SegmentDataset, segment_collate_fn
from model import get_model
from eval_model import infer_model_kwargs, confusion_matrix, per_class_metrics


def set_bn_eval(module: nn.Module):
    """Fuerza modo eval SOLO en las capas BatchNorm, dejando el resto como este."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def count_bn_layers(module: nn.Module) -> int:
    return sum(
        1 for m in module.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    )


@torch.no_grad()
def run_mode(model, state_dict, loader, device, num_classes,
             max_windows_per_forward, mode: str, limit: int):
    """
    mode:
      "eval"     -> model.eval()                (lo que hace validate())
      "train"    -> model.train()               (lo que ve train/loss)
      "bn_eval"  -> model.train() + BN en eval  (aisla el efecto de BN)
    """
    # Recarga limpia: un forward en modo train muta las running_stats
    model.load_state_dict(state_dict, strict=False)

    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    elif mode == "bn_eval":
        model.train()
        set_bn_eval(model)
    else:
        raise ValueError(mode)

    preds, labels = [], []
    seen = 0

    for windows, label, _ in tqdm(loader, total=min(limit, len(loader)),
                                  desc=f"modo={mode:<8}", leave=False):
        if seen >= limit:
            break
        seen += 1

        num_windows = windows.shape[0]
        labels_expanded = label.repeat(num_windows).to(device)

        for start in range(0, num_windows, max_windows_per_forward):
            chunk = windows[start:start + max_windows_per_forward].to(device, dtype=torch.float)
            outputs = model(chunk)
            preds.append(outputs.argmax(1).cpu())
            labels.append(labels_expanded[start:start + max_windows_per_forward].cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    cm = confusion_matrix(preds, labels, num_classes)
    acc = np.trace(cm) / cm.sum()
    macro_f1 = sum(m["f1"] for m in per_class_metrics(cm)) / num_classes
    return acc, macro_f1, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="TRAIN")
    parser.add_argument("--num-segments", type=int, default=200,
                        help="cuantos segmentos evaluar (subconjunto, por velocidad)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    inferred = infer_model_kwargs(state_dict)
    num_classes = inferred.pop("num_classes")

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
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=4, collate_fn=segment_collate_fn)

    model = get_model(num_classes, **inferred).to(args.device)
    frozen_sd = copy.deepcopy(state_dict)

    n_bn = count_bn_layers(model)
    limit = min(args.num_segments, len(dataset))

    print(f"Checkpoint     : {args.checkpoint}  (epoca {ckpt.get('epoch', '?')})")
    print(f"Split          : {args.split}  ({limit} de {len(dataset)} segmentos)")
    print(f"Capas BatchNorm: {n_bn}")
    print(f"Frames por forward: {conf.max_windows_per_forward} windows "
          f"x {conf.sequence_length} frames = "
          f"{conf.max_windows_per_forward * conf.sequence_length}")
    print()

    resultados = {}
    for mode in ("eval", "train", "bn_eval"):
        acc, f1, cm = run_mode(
            model, frozen_sd, loader, args.device, num_classes,
            conf.max_windows_per_forward, mode, limit,
        )
        resultados[mode] = (acc, f1, cm)

    print("=" * 62)
    print(f"{'modo':<28}{'accuracy':>12}{'macro-F1':>12}")
    print("-" * 62)
    etiquetas = {
        "eval": "eval  (= validate())",
        "train": "train (= lo que ve train/loss)",
        "bn_eval": "train pero BN en eval",
    }
    for mode in ("eval", "train", "bn_eval"):
        acc, f1, _ = resultados[mode]
        print(f"{etiquetas[mode]:<28}{acc:>12.4f}{f1:>12.4f}")
    print("=" * 62)

    acc_eval = resultados["eval"][0]
    acc_train = resultados["train"][0]
    acc_bn = resultados["bn_eval"][0]
    gap = acc_train - acc_eval

    print()
    print("Conclusion:")
    if gap > 0.15:
        print(f"  CONFIRMADO: brecha de {gap:+.4f} en accuracy entre modo train y")
        print("  modo eval sobre LOS MISMOS DATOS. No es overfitting: es una")
        print("  discrepancia train/eval.")
        if abs(acc_bn - acc_eval) < abs(acc_bn - acc_train):
            print()
            print("  Al forzar SOLO BatchNorm a modo eval, el rendimiento cae al")
            print("  nivel de modo eval -> el causante es BatchNorm, no Dropout.")
        else:
            print()
            print("  AVISO: con BN en eval el rendimiento sigue alto -> BatchNorm")
            print("  no seria el unico causante. Revisar Dropout u otra fuente.")
    elif gap > 0.05:
        print(f"  Brecha moderada ({gap:+.4f}). BatchNorm aporta, pero puede haber")
        print("  otras causas en juego.")
    else:
        print(f"  Brecha chica ({gap:+.4f}). La hipotesis de BatchNorm NO se sostiene;")
        print("  hay que buscar otra explicacion para train/loss ~ 0.")


if __name__ == "__main__":
    main()