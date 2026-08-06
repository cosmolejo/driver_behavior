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
    pesar cada segmento por igual sin importar cu�ntas windows genere.
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

    La normalizaci�n (dividir por num_windows del segmento, y por
    loss_scale = 1/accumulation_steps) asegura que el gradiente acumulado
    sea matem�ticamente id�ntico a haber promediado las windows del
    segmento en un solo forward gigante. El tama�o de chunk es solo una
    perilla de memoria/velocidad, no cambia el resultado.

    Devuelve: (loss_promedio_del_segmento (float), correctos, num_windows,
               preds_cpu, labels_cpu)
    """
    num_windows = windows.shape[0]
    labels_expanded = label.repeat(num_windows).to(device)

    total_loss = 0.0
    total_correct = 0
    seg_preds = []
    seg_labels = []

    for start in range(0, num_windows, max_windows_per_forward):
        chunk = windows[start:start + max_windows_per_forward].to(device, dtype=torch.float)
        chunk_labels = labels_expanded[start:start + max_windows_per_forward]

        outputs = model(chunk)
        per_window_loss = criterion(outputs, chunk_labels)  # (chunk_size,), reduction='none'

        # Promedio sobre el segmento completo * escala de acumulaci�n,
        # partido en este chunk (ver docstring)
        chunk_loss = per_window_loss.sum() / num_windows * loss_scale
        chunk_loss.backward()

        total_loss += per_window_loss.sum().item()
        _, predicted = outputs.max(1)
        total_correct += (predicted == chunk_labels).sum().item()
        seg_preds.append(predicted.detach().cpu())
        seg_labels.append(chunk_labels.detach().cpu())

    seg_preds = torch.cat(seg_preds)
    seg_labels = torch.cat(seg_labels)

    return total_loss / num_windows, total_correct, num_windows, seg_preds, seg_labels


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
    num_classes: int,
    label_groups: Optional[list],
    accumulation_steps: int,
    max_windows_per_forward: int,
    grad_clip: Optional[float] = None,
    log_every_n_steps=50,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_windows = 0

    # Acumuladores a nivel de �POCA COMPLETA (independientes de la ventana
    # de logging cada log_every_n_steps), para poder comparar train vs val
    # en igualdad de condiciones (misma unidad de agregaci�n: toda la
    # �poca) y detectar el gap real de generalizaci�n.
    epoch_loss_sum = 0.0
    epoch_correct = 0
    epoch_windows = 0
    epoch_preds = []
    epoch_labels = []

    optimizer.zero_grad()
    segments_since_step = 0

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Training: Epoch [{epoch+1}/{num_epochs}]",
        leave=True,
    )

    for _, (windows, label, _) in pbar:
        seg_loss, seg_correct, seg_windows, seg_preds, seg_labels = segment_forward_backward(
            model, criterion, windows, label, device,
            max_windows_per_forward=max_windows_per_forward,
            loss_scale=1.0 / accumulation_steps,
        )

        running_loss += seg_loss * seg_windows
        running_correct += seg_correct
        running_windows += seg_windows

        epoch_loss_sum += seg_loss * seg_windows
        epoch_correct += seg_correct
        epoch_windows += seg_windows
        epoch_preds.append(seg_preds)
        epoch_labels.append(seg_labels)

        segments_since_step += 1
        if segments_since_step == accumulation_steps:
            # Recorte de norma de gradiente. Sin esto, descongelar los
            # bloques tempranos del backbone puede producir updates lo
            # bastante grandes como para destruir las features de
            # ImageNet en las primeras iteraciones: el modelo colapsa a
            # predecir siempre la clase mayoritaria y no se recupera.
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    # (�ltimo batch de la �poca), se descartan esos gradientes parciales
    # llamando zero_grad impl�citamente en la pr�xima �poca; no se hace
    # optimizer.step() con un accumulation incompleto para no sesgar la
    # escala del gradiente.

    epoch_train_loss = epoch_loss_sum / epoch_windows
    epoch_train_acc = epoch_correct / epoch_windows
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)
    if label_groups is None:
        epoch_train_macro_f1 = compute_macro_f1(epoch_preds, epoch_labels, num_classes)
    else:
        # En train se mapea el argmax fino a macro (en vez de sumar
        # probabilidades) para no tener que guardar las 9 probabilidades
        # de cada ventana. Es una metrica de diagnostico, sirve para leer
        # la brecha train-val; la de seleccion es la de validate().
        from fine_labels import MACRO_CLASSES, map_fine_to_macro
        epoch_train_macro_f1 = compute_macro_f1(
            map_fine_to_macro(epoch_preds, label_groups),
            map_fine_to_macro(epoch_labels, label_groups),
            len(MACRO_CLASSES),
        )

    # Logueados en el mismo global_step que val, para poder comparar
    # train vs val directamente en TensorBoard (mismo punto en el eje x).
    writer.add_scalar("train/epoch_loss", epoch_train_loss, global_step)
    writer.add_scalar("train/epoch_accuracy", epoch_train_acc, global_step)
    writer.add_scalar("train/epoch_macro_f1", epoch_train_macro_f1, global_step)

    return global_step, epoch_train_loss, epoch_train_acc, epoch_train_macro_f1


def validate(model, dataloader, criterion, device, epoch, num_epochs,
             max_windows_per_forward: int, num_classes: int,
             label_groups: Optional[list] = None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_windows = 0
    segments_with_a_correct_window = 0
    total_segments = 0

    all_preds = []
    all_labels = []
    all_macro_preds = []   # solo en modo fine

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

                if label_groups is not None:
                    # P(macro) = suma de las P(finas) de ese grupo. Sumar
                    # probabilidades (no logits) es lo correcto: la
                    # probabilidad de la union de eventos excluyentes es
                    # la suma de sus probabilidades.
                    from fine_labels import aggregate_probs
                    agg = aggregate_probs(outputs, label_groups)
                    all_macro_preds.append(agg.argmax(1).cpu())

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

    if label_groups is None:
        macro_f1 = compute_macro_f1(all_preds, all_labels, num_classes)
        macro_f1_fine = None
    else:
        from fine_labels import MACRO_CLASSES, map_fine_to_macro
        macro_f1_fine = compute_macro_f1(all_preds, all_labels, num_classes)
        macro_f1 = compute_macro_f1(
            torch.cat(all_macro_preds),
            map_fine_to_macro(all_labels, label_groups),
            len(MACRO_CLASSES),
        )

    return epoch_loss, epoch_acc, at_least_one_correct_tot, macro_f1, macro_f1_fine


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
    early_stopping_patience: Optional[int] = None,
    unfreeze_schedule: Optional[dict] = None,
    unfreeze_lr_decay: float = 0.3,
    seed: Optional[int] = 42,
    label_mode: str = "macro",
    partition_report: Optional[str] = None,
    grad_clip: Optional[float] = None,
    augment: Optional[str] = None,
    temporal_stride_jitter: Optional[list] = None,
    temporal_offset_jitter: bool = False,
    temporal_cutout_frames: int = 0,
    augment_strength_by_class: Optional[dict] = None,
    subject_subset: Optional[int] = None,
):
    """
    Entrenamiento con loss promediada por segmento: cada segmento aporta UNA
    contribuci�n de gradiente (promedio de la CE de todas sus windows), sin
    importar cu�ntas windows haya generado. `accumulation_steps` segmentos
    se acumulan antes de cada optimizer.step() (batch efectivo fijo).

    Si se pasa `trial` (optuna.Trial), se reporta val_macro_f1 al final de
    cada �poca para permitir pruning (direcci�n "maximize" en el study).
    Devuelve el mejor val_macro_f1 observado. val_loss se sigue trackeando
    y logueando en TensorBoard, pero ya no es la m�trica de selecci�n.

    El checkpoint "best" se guarda por mejor val_macro_f1 (antes era por
    val_acc), consistente con el criterio de selecci�n de Optuna.

    Si `early_stopping_patience` no es None, el entrenamiento corta antes
    de completar `num_epochs` si val_macro_f1 no mejora durante esa
    cantidad de �pocas consecutivas. Por defecto (None) queda deshabilitado
    para no alterar el comportamiento de la b�squeda de Optuna
    (search_epochs ya es corto); pensado para usarse en el retrain final
    de num_epochs largo.
    """
    loss_kwargs = dict(loss_kwargs or {})
    model_kwargs = dict(model_kwargs or {})

    # -----------------------------------------------------------------
    # Semilla
    #
    # Sin esto, dos corridas con hiperparametros identicos difieren por la
    # inicializacion de LSTM/atencion/clasificador y por el orden de
    # shuffling. Esa varianza medida sobre este dataset es de ~0.04 en
    # macro-F1 a nivel segmento (hasta 0.08 en epocas tempranas), o sea
    # mayor que el efecto de la mayoria de las ablaciones. Con la semilla
    # fija, las corridas difieren SOLO por la intervencion.
    #
    # No se activa torch.use_deterministic_algorithms: forzaria kernels
    # deterministas mas lentos y algunas ops de cuDNN (LSTM entre ellas)
    # no lo soportan. Queda una no-determinacion residual de cuDNN, pero
    # mucho menor que la de no sembrar nada.
    # -----------------------------------------------------------------
    generator = None
    if seed is not None:
        import random as _random

        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        print(f"Semilla fijada en {seed}")

    # ------------------------------------------------------------------
    # Pipeline de transforms
    #
    # Sin augmentation el orden es el de siempre (todo por frame). Con
    # augmentation hay que PARTIRLO, porque las transformaciones de clip
    # necesitan el rango [0, 1]:
    #
    #     por frame : ToPILImage -> Resize -> ToTensor      (queda en [0,1])
    #     por clip  : augment(clip, label)                  <- nivel SEGMENTO
    #     por clip  : ClipNormalize                         <- despues
    #
    # Normalizar antes de la augmentation romperia los ops de color
    # (adjust_brightness y compa�ia asumen [0,1]).
    # ------------------------------------------------------------------
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    base_transform = [
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ]

    if augment is None:
        transform = transforms.Compose(
            base_transform + [transforms.Normalize(mean=MEAN, std=STD)]
        )
        normalize = None
        train_augment = None
    else:
        from augmentation import build_clip_augment
        from dataset import ClipNormalize

        transform = transforms.Compose(base_transform)
        normalize = ClipNormalize(MEAN, STD)

        if augment not in ("photometric", "temporal", "both"):
            raise ValueError(
                f"augment debe ser 'photometric', 'temporal' o 'both', no {augment!r}"
            )

        train_augment = build_clip_augment(
            train=True,
            strength_by_class=dict(augment_strength_by_class or {}),
            use_photometric=augment in ("photometric", "both"),
            use_temporal_cutout=augment in ("temporal", "both")
                                 and temporal_cutout_frames > 0,
            temporal_cutout_frames=temporal_cutout_frames,
            layout="CTHW",
        )
        print(f"Augmentation: {augment}"
              f" | stride_jitter={temporal_stride_jitter}"
              f" | offset_jitter={temporal_offset_jitter}"
              f" | cutout_frames={temporal_cutout_frames}")

    ds_kwargs = dict(sequence_length=sequence_length,
                     sample_one_each=sample_one_each,
                     transform=transform,
                     label_mode=label_mode,
                     partition_report=partition_report,
                     normalize=normalize)

    # La augmentation y el jitter temporal van SOLO en train. En validacion
    # la evaluacion tiene que ser determinista, o la metrica de una misma
    # epoca cambiaria entre corridas y dejaria de ser comparable.
    # El submuestreo por sujeto va SOLO en train: validacion siempre
    # completa, o los puntos de la curva no serian comparables entre si.
    train_dataset = SegmentDataset(
        os.path.join(data_dir, "TRAIN"),
        subject_subset=subject_subset,
        augment=train_augment,
        temporal_stride_jitter=(list(temporal_stride_jitter)
                                if (augment in ("temporal", "both")
                                    and temporal_stride_jitter) else None),
        temporal_offset_jitter=(temporal_offset_jitter
                                and augment in ("temporal", "both")),
        **ds_kwargs)
    val_dataset = SegmentDataset(os.path.join(data_dir, "VALIDATION"), **ds_kwargs)

    # En modo "fine" la red tiene 9 salidas, pero la METRICA sigue siendo
    # macro-F1 sobre las 3 macro-clases (agregando probabilidades), para
    # que `val/macro_f1` sea comparable con toda la bateria anterior.
    label_groups = getattr(train_dataset, "label_groups", None)
    n_out = len(train_dataset.class_to_idx)
    if n_out != num_classes:
        print(f"AVISO: config dice num_classes={num_classes} pero el dataset "
              f"tiene {n_out} clases (label_mode={label_mode!r}). Se usa {n_out}.")
        num_classes = n_out

    # batch_size=1: cada __getitem__ ya es "todas las windows de un
    # segmento"; el agrupamiento real ocurre v�a accumulation_steps.
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=segment_collate_fn,
        generator=generator,   # shuffling reproducible (None = comportamiento previo)
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

    # --- Descongelamiento progresivo (opcional) ---
    # Los param groups se crean UNA sola vez, incluidos los que arrancan
    # congelados: OneCycleLR fija sus max_lr en el constructor y agregar
    # grupos despues lo rompe. Descongelar = togglear requires_grad.
    unfreezer = None
    if unfreeze_schedule is not None:
        from progressive_unfreeze import ProgressiveUnfreezer

        # OmegaConf puede entregar las claves como str; se normalizan a int
        schedule = {int(k): int(v) for k, v in dict(unfreeze_schedule).items()}
        unfreezer = ProgressiveUnfreezer(model, schedule, verbose=True)
        print(unfreezer.summary())

        optimizer = torch.optim.AdamW(
            unfreezer.build_param_groups(base_lr=lr, decay=unfreeze_lr_decay),
            weight_decay=weight_decay,
        )
        max_lr = unfreezer.max_lrs(base_lr=lr, decay=unfreeze_lr_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        max_lr = lr

    steps_per_epoch = max(1, len(train_loader) // accumulation_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
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
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(num_epochs):
        if unfreezer is not None:
            model.train()                  # re-aplica freeze_bn
            unfreezer.on_epoch_start(epoch)
            unfreezer.assert_bn_frozen()   # falla ruidosamente si BN volvio a train

        global_step, train_epoch_loss, train_epoch_acc, train_epoch_macro_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, writer, global_step, epoch, num_epochs,
            num_classes=num_classes,
            label_groups=label_groups,
            accumulation_steps=accumulation_steps,
            grad_clip=grad_clip,
            max_windows_per_forward=max_windows_per_forward,
            log_every_n_steps=50,
        )
        val_loss, val_acc, at_least_one_correct_tot, val_macro_f1, val_macro_f1_fine = validate(
            model, val_loader, criterion, device, epoch, num_epochs,
            max_windows_per_forward=max_windows_per_forward,
            num_classes=num_classes,
            label_groups=label_groups,
        )

        writer.add_scalar("val/loss", val_loss, global_step)
        writer.add_scalar("val/accuracy", val_acc, global_step)
        writer.add_scalar("val/at_least_one_correct", at_least_one_correct_tot, global_step)
        writer.add_scalar("val/macro_f1", val_macro_f1, global_step)
        if val_macro_f1_fine is not None:
            writer.add_scalar("val/macro_f1_fine", val_macro_f1_fine, global_step)

        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)

        improved = val_macro_f1 > best_val_macro_f1
        best_val_macro_f1 = max(best_val_macro_f1, val_macro_f1)

        if save_checkpoints:
            if improved:
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_macro_f1": val_macro_f1,
                    },
                    models_dir / "model_best.pth"
                )
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                },
                models_dir / f"model_{epoch}.pth"
            )

        if trial is not None:
            trial.report(val_macro_f1, epoch)
            if trial.should_prune():
                writer.close()
                raise optuna.TrialPruned()

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping en �poca {epoch}: sin mejora de val_macro_f1 "
                f"durante {epochs_without_improvement} �pocas consecutivas "
                f"(mejor: {best_val_macro_f1:.4f})"
            )
            break

    writer.close()
    return best_val_macro_f1