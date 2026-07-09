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

from dataset import VideoDataset
from model import get_model
from loss_factory import LossFactory


def video_collate_fn(batch):
    """
    batch: list of tuples (video_tensor, label)
    video_tensor: (C, T, H, W)
    """
    videos, labels, names = zip(*batch)

    # Pad sequences to the max length in the batch
    max_len = max(v.shape[1] for v in videos)
    padded_videos = []

    for v in videos:
        C, T, H, W = v.shape
        if T < max_len:
            # Pad along time dimension with zeros
            pad = v[:, -1, ...].repeat(1, max_len - T, 1, 1)
            v = torch.cat([v, pad], dim=1)
        padded_videos.append(v)

    videos_batch = torch.stack(padded_videos)  # (B, C, T, H, W)
    labels_batch = torch.tensor(labels, dtype=torch.long)

    return videos_batch, labels_batch, names


def compute_class_weights(dataset: VideoDataset, num_classes: int, power: float = 1.0) -> torch.Tensor:
    """
    Pesos inversamente proporcionales a la frecuencia de cada clase
    (elevada a `power`, tuneable), normalizados para que promedien 1.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label, _ in dataset.samples:
        counts[label] += 1
    counts = np.clip(counts, 1, None)  # evitar división por cero
    weights = (1.0 / counts) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float)


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
    log_every_n_steps=50,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Training: Epoch [{epoch+1}/{num_epochs}]",
        leave=True,
    )

    for _, (videos, labels, _) in pbar:
        videos = videos.to(device, dtype=torch.float)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()

        global_step += 1

        running_loss += loss.item() * videos.size(0)
        _, predicted = outputs.max(1)
        running_samples += labels.size(0)
        running_correct += (predicted == labels).sum().item()

        if global_step % log_every_n_steps == 0:
            avg_loss = running_loss / running_samples
            avg_acc = running_correct / running_samples

            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/accuracy", avg_acc, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            running_loss = 0
            running_correct = 0
            running_samples = 0

    return global_step


def validate(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Validating: Epoch [{epoch+1}/{num_epochs}]",
        leave=True,
    )

    at_least_one_correct = {}
    with torch.no_grad():
        for _, (videos, labels, names) in pbar:
            for name in names:
                if name not in at_least_one_correct:
                    at_least_one_correct[name] = False

            videos = videos.to(device, dtype=torch.float)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * videos.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct_tensor = (predicted == labels)
            correct += correct_tensor.sum().item()

            for c, n in zip(correct_tensor, names):
                at_least_one_correct[n] = at_least_one_correct[n] or c

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    at_least_one_correct_tot = sum(1 for v in at_least_one_correct.values() if v) / len(at_least_one_correct)
    return epoch_loss, epoch_acc, at_least_one_correct_tot


# ==========================
# Full Training Pipeline
# ==========================
def train_pipeline(
    data_dir: str,
    num_classes: int,
    sequence_length: int = 32,
    sample_one_each: int = 1,
    batch_size: int = 4,
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
    Si se pasa `trial` (optuna.Trial), se reporta val_loss al final de cada
    época para permitir pruning (trial.should_prune()). En ese caso la
    función levanta optuna.TrialPruned() cuando corresponde, y devuelve el
    mejor val_loss observado para que se use como valor objetivo de Optuna.
    """
    loss_kwargs = dict(loss_kwargs or {})
    model_kwargs = dict(model_kwargs or {})

    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    # Datasets and Dataloaders
    train_dataset = VideoDataset(os.path.join(data_dir, "TRAIN"),
                                  sequence_length=sequence_length,
                                  sample_one_each=sample_one_each,
                                  transform=transform)
    val_dataset = VideoDataset(os.path.join(data_dir, "VALIDATION"),
                                sequence_length=sequence_length,
                                sample_one_each=sample_one_each,
                                transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=video_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=video_collate_fn
    )

    # Model
    model = get_model(num_classes, **model_kwargs).to(device)

    # Loss
    # CE_weight necesita un tensor de pesos por clase calculado a partir
    # del split de train; class_weight_power es el único hiperparámetro
    # tuneable de este esquema (viene dentro de loss_kwargs).
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

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
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
    global_step = 0

    for epoch in range(num_epochs):
        global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, writer, global_step, epoch, num_epochs, 100
        )
        val_loss, val_acc, at_least_one_correct_tot = validate(
            model, val_loader, criterion, device, epoch, num_epochs
        )

        writer.add_scalar("val/loss", val_loss, global_step)
        writer.add_scalar("val/accuracy", val_acc, global_step)
        writer.add_scalar("val/at_least_one_correct", at_least_one_correct_tot, global_step)

        best_val_loss = min(best_val_loss, val_loss)

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

        # --- Optuna: reporte + pruning ---
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                writer.close()
                raise optuna.TrialPruned()

    writer.close()
    return best_val_loss
