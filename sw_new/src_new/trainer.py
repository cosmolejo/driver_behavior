import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

try:
    import optuna
except ImportError:
    optuna = None

from dataset import SegmentDataset, segment_collate_fn
from model import get_model
from loss_factory import LossFactory


def compute_class_weights(dataset: SegmentDataset, num_classes: int, power: float = 1.0) -> torch.Tensor:
    """
    Pesos inversamente proporcionales a la frecuencia de cada clase, contada
    en SEGMENTOS (no en windows) para ser consistente con el esquema de
    pesar cada segmento por igual sin importar cuántas windows genere.
    Elevados a `power` (tuneable), normalizados para que promedien 1.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label, _, _ in dataset.segments:
        counts[label] += 1
    counts = np.clip(counts, 1, None)
    weights = (1.0 / counts) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float)


def compute_macro_f1(all_preds: torch.Tensor, all_labels: torch.Tensor, num_classes: int) -> float:
    """
    Macro-F1 a nivel window (misma granularidad que val/accuracy actual).
    Clases sin TP+FP o sin TP+FN se tratan como F1=0 para esa clase
    (equivalente a zero_division=0 de sklearn), en vez de NaN.
    """
    f1s = []
    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum().item()
        fp = ((all_preds == c) & (all_labels != c)).sum().item()
        fn = ((all_preds != c) & (all_labels == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / num_classes


def segment_forward_backward(
    model,
    criterion,
    windows: torch.Tensor,
    label: torch.Tensor,
    device,
    max_windows_per_forward: int,
    loss_scale: float = 1.0,
):
    """
    Procesa TODAS las windows de un segmento, en chunks de a lo sumo
    `max_windows_per_forward` (para no reventar memoria en segmentos con
    muchas windows), y hace backward() incremental por chunk.

    La normalización (dividir por num_windows del segmento, y por
    loss_scale = 1/accumulation_steps) asegura que el gradiente acumulado
    sea matemáticamente idéntico a haber promediado las windows del
    segmento en un solo forward gigante. El tamaño de chunk es solo una
    perilla de memoria/velocidad, no cambia el resultado.

    Devuelve: (loss_promedio_del_segmento (float), correctos, num_windows)
    """
    num_windows = windows.shape[0]
    labels_expanded = label.repeat(num_windows).to(device)

    total_loss = 0.0
    total_correct = 0

    for start in range(0, num_windows, max_windows_per_forward):
        chunk = windows[start:start + max_windows_per_forward].to(device, dtype=torch.float)
        chunk_labels = labels_expanded[start:start + max_windows_per_forward]

        outputs = model(chunk)
        per_window_loss = criterion(outputs, chunk_labels)  # (chunk_size,), reduction='none'

        # Promedio sobre el segmento completo * escala de acumulación,
        # partido en este chunk (ver docstring)
        chunk_loss = per_window_loss.sum() / num_windows * loss_scale
        chunk_loss.backward()

        total_loss += per_window_loss.sum().item()
        _, predicted = outputs.max(1)
        total_correct += (predicted == chunk_labels).sum().item()

    return total_loss / num_windows, total_correct, num_windows


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    writer,
    global_step,
    epoch,
    num_epochs,
    accumulation_steps: int,
    max_windows_per_forward: int,
    log_every_n_steps=50,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_windows = 0

    optimizer.zero_grad()
    segments_since_step = 0

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Training: Epoch [{epoch+1}/{num_epochs}]",
        leave=True,
    )

    for _, (windows, label, _) in pbar:
        seg_loss, seg_correct, seg_windows = segment_forward_backward(
            model, criterion, windows, label, device,
            max_windows_per_forward=max_windows_per_forward,
            loss_scale=1.0 / accumulation_steps,
        )

        running_loss += seg_loss * seg_windows
        running_correct += seg_correct
        running_windows += seg_windows

        segments_since_step += 1
        if segments_since_step == accumulation_steps:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            segments_since_step = 0
            global_step += 1

            if global_step % log_every_n_steps == 0:
                avg_loss = running_loss / running_windows
                avg_acc = running_correct / running_windows

                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/accuracy", avg_acc, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                running_loss = 0
                running_correct = 0
                running_windows = 0

    # Si sobran segmentos acumulados sin completar un accumulation_steps
    # (último batch de la época), se descartan esos gradientes parciales
    # llamando zero_grad implícitamente en la próxima época; no se hace
    # optimizer.step() con un accumulation incompleto para no sesgar la
    # escala del gradiente.

    return global_step


def validate(model, dataloader, criterion, device, epoch, num_epochs, max_windows_per_forward: int, num_classes: int):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_windows = 0
    segments_with_a_correct_window = 0
    total_segments = 0

    all_preds = []
    all_labels = []

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Validating: Epoch [{epoch+1}/{num_epochs}]",
        leave=True,
    )

    with torch.no_grad():
        for _, (windows, label, _) in pbar:
            num_windows = windows.shape[0]
            labels_expanded = label.repeat(num_windows).to(device)

            seg_loss_sum = 0.0
            seg_correct_mask = []

            for start in range(0, num_windows, max_windows_per_forward):
                chunk = windows[start:start + max_windows_per_forward].to(device, dtype=torch.float)
                chunk_labels = labels_expanded[start:start + max_windows_per_forward]

                outputs = model(chunk)
                per_window_loss = criterion(outputs, chunk_labels)
                seg_loss_sum += per_window_loss.sum().item()

                _, predicted = outputs.max(1)
                seg_correct_mask.append((predicted == chunk_labels).cpu())
                all_preds.append(predicted.cpu())
                all_labels.append(chunk_labels.cpu())

            seg_correct_mask = torch.cat(seg_correct_mask)

            total_loss += seg_loss_sum
            total_correct += seg_correct_mask.sum().item()
            total_windows += num_windows

            total_segments += 1
            if seg_correct_mask.any():
                segments_with_a_correct_window += 1

    epoch_loss = total_loss / total_windows
    epoch_acc = total_correct / total_windows
    at_least_one_correct_tot = segments_with_a_correct_window / total_segments

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    macro_f1 = compute_macro_f1(all_preds, all_labels, num_classes)

    return epoch_loss, epoch_acc, at_least_one_correct_tot, macro_f1


# ==========================
# Full Training Pipeline
# ==========================
def train_pipeline(
    data_dir: str,
    num_classes: int,
    sequence_length: int = 32,
    sample_one_each: int = 1,
    accumulation_steps: int = 4,
    max_windows_per_forward: int = 8,
    num_epochs: int = 10,
    loss_fn: str = "cross_entropy",
    loss_kwargs: Optional[dict] = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    pct_start: float = 0.1,
    div_factor: float = 25,
    final_div_factor: float = 1000,
    model_kwargs: Optional[dict] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    trial: Optional["optuna.Trial"] = None,
    run_name: str = "video_classifier",
    save_checkpoints: bool = True,
):
    """
    Entrenamiento con loss promediada por segmento: cada segmento aporta UNA
    contribución de gradiente (promedio de la CE de todas sus windows), sin
    importar cuántas windows haya generado. `accumulation_steps` segmentos
    se acumulan antes de cada optimizer.step() (batch efectivo fijo).

    Si se pasa `trial` (optuna.Trial), se reporta val_macro_f1 al final de
    cada época para permitir pruning (dirección "maximize" en el study).
    Devuelve el mejor val_macro_f1 observado. val_loss se sigue trackeando
    y logueando en TensorBoard, pero ya no es la métrica de selección.
    """
    loss_kwargs = dict(loss_kwargs or {})
    model_kwargs = dict(model_kwargs or {})

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SegmentDataset(os.path.join(data_dir, "TRAIN"),
                                    sequence_length=sequence_length,
                                    sample_one_each=sample_one_each,
                                    transform=transform)
    val_dataset = SegmentDataset(os.path.join(data_dir, "VALIDATION"),
                                  sequence_length=sequence_length,
                                  sample_one_each=sample_one_each,
                                  transform=transform)

    # batch_size=1: cada __getitem__ ya es "todas las windows de un
    # segmento"; el agrupamiento real ocurre vía accumulation_steps.
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=segment_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=segment_collate_fn,
    )

    model = get_model(num_classes, **model_kwargs).to(device)

    weight_tensor = None
    if loss_fn == "CE_weight":
        power = loss_kwargs.pop("class_weight_power", 1.0)
        weight_tensor = compute_class_weights(train_dataset, num_classes, power=power).to(device)

    criterion = LossFactory.get_loss(
        loss_fn, num_classes=num_classes, weight=weight_tensor, **loss_kwargs
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    steps_per_epoch = max(1, len(train_loader) // accumulation_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )

    base_dir = Path(__file__).parent.parent.resolve()
    writer = SummaryWriter(base_dir / "runs" / run_name)
    models_dir = base_dir / "models" / run_name
    if save_checkpoints:
        models_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_val_macro_f1 = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, writer, global_step, epoch, num_epochs,
            accumulation_steps=accumulation_steps,
            max_windows_per_forward=max_windows_per_forward,
            log_every_n_steps=50,
        )
        val_loss, val_acc, at_least_one_correct_tot, val_macro_f1 = validate(
            model, val_loader, criterion, device, epoch, num_epochs,
            max_windows_per_forward=max_windows_per_forward,
            num_classes=num_classes,
        )

        writer.add_scalar("val/loss", val_loss, global_step)
        writer.add_scalar("val/accuracy", val_acc, global_step)
        writer.add_scalar("val/at_least_one_correct", at_least_one_correct_tot, global_step)
        writer.add_scalar("val/macro_f1", val_macro_f1, global_step)

        best_val_loss = min(best_val_loss, val_loss)
        best_val_macro_f1 = max(best_val_macro_f1, val_macro_f1)

        if save_checkpoints:
            if val_acc > best_val_acc:
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_acc": val_acc,
                    },
                    models_dir / "model_best.pth"
                )
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                models_dir / f"model_{epoch}.pth"
            )
        else:
            best_val_acc = max(best_val_acc, val_acc)

        if trial is not None:
            trial.report(val_macro_f1, epoch)
            if trial.should_prune():
                writer.close()
                raise optuna.TrialPruned()

    writer.close()
    return best_val_macro_f1