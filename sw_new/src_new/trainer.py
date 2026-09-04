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
    """
    Evalua a DOS granularidades a la vez:

    - NIVEL WINDOW: cada ventana es una muestra independiente.
    - NIVEL SEGMENTO: se promedian las probabilidades softmax de todas las
      ventanas del segmento y se toma el argmax.

    El nivel de SEGMENTO es el que importa: es la unidad de etiquetado, la
    unidad de decision de la aplicacion, y la unidad que pondera la perdida
    (window bagging hace que cada segmento aporte una sola contribucion de
    gradiente, sin importar cuantas ventanas genere).

    Las dos metricas pueden ordenar DISTINTO, no es una diferencia de ruido:
    en corridas largas se observo que una mejora mientras la otra empeora
    durante epocas consecutivas, y que elegir el checkpoint por la metrica
    de ventana cuesta hasta 0.047 de macro-F1 a nivel segmento.

    Devuelve:
        (loss, acc_window, at_least_one_correct,
         macro_f1_window, macro_f1_fine, macro_f1_segment, acc_segment)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_windows = 0
    segments_with_a_correct_window = 0
    total_segments = 0

    all_preds = []
    all_labels = []
    all_macro_preds = []   # solo en modo fine

    # Nivel segmento: una prediccion y una etiqueta por segmento
    seg_preds = []
    seg_labels = []

    if label_groups is not None:
        from fine_labels import aggregate_probs, map_fine_to_macro, MACRO_CLASSES
        n_out = len(MACRO_CLASSES)
    else:
        n_out = num_classes

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
            seg_probs = []          # probabilidades de este segmento

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

                if label_groups is None:
                    probs = torch.softmax(outputs, dim=1)
                else:
                    # P(macro) = suma de las P(finas) de ese grupo. Sumar
                    # probabilidades (no logits) es lo correcto: la
                    # probabilidad de la union de eventos excluyentes es
                    # la suma de sus probabilidades.
                    probs = aggregate_probs(outputs, label_groups)
                    all_macro_preds.append(probs.argmax(1).cpu())

                seg_probs.append(probs.cpu())

            seg_correct_mask = torch.cat(seg_correct_mask)

            total_loss += seg_loss_sum
            total_correct += seg_correct_mask.sum().item()
            total_windows += num_windows

            total_segments += 1
            if seg_correct_mask.any():
                segments_with_a_correct_window += 1

            # Agregacion del segmento: promedio de probabilidades sobre sus
            # ventanas. Promediar probabilidades (no logits ni votos) es la
            # forma correcta de combinar predicciones de un mismo evento.
            seg_prob = torch.cat(seg_probs).mean(dim=0)
            seg_preds.append(int(seg_prob.argmax().item()))
            lab = int(label.item())
            seg_labels.append(label_groups[lab] if label_groups is not None else lab)

    epoch_loss = total_loss / total_windows
    epoch_acc = total_correct / total_windows
    at_least_one_correct_tot = segments_with_a_correct_window / total_segments

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    if label_groups is None:
        macro_f1 = compute_macro_f1(all_preds, all_labels, num_classes)
        macro_f1_fine = None
    else:
        macro_f1_fine = compute_macro_f1(all_preds, all_labels, num_classes)
        macro_f1 = compute_macro_f1(
            torch.cat(all_macro_preds),
            map_fine_to_macro(all_labels, label_groups),
            n_out,
        )

    seg_preds_t = torch.tensor(seg_preds, dtype=torch.long)
    seg_labels_t = torch.tensor(seg_labels, dtype=torch.long)
    macro_f1_segment = compute_macro_f1(seg_preds_t, seg_labels_t, n_out)
    acc_segment = (seg_preds_t == seg_labels_t).float().mean().item()

    return (epoch_loss, epoch_acc, at_least_one_correct_tot,
            macro_f1, macro_f1_fine, macro_f1_segment, acc_segment)



def _rng_state() -> dict:
    """
    Estado de TODOS los generadores aleatorios del proceso principal.

    Sin esto, reanudar produce un orden de shuffling y una secuencia de
    augmentation distintos a los que habria tenido la corrida original: el
    entrenamiento continua, pero no es el mismo que se interrumpio.
    """
    import random as _random

    estado = {
        "python": _random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        estado["cuda"] = torch.cuda.get_rng_state_all()
    return estado


def _restore_rng_state(estado: dict):
    import random as _random

    if not estado:
        return
    if "python" in estado:
        _random.setstate(estado["python"])
    if "numpy" in estado:
        np.random.set_state(estado["numpy"])
    if "torch" in estado:
        torch.set_rng_state(estado["torch"].cpu().to(torch.uint8))
    if "cuda" in estado and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in estado["cuda"]])
        except Exception as e:
            print(f"  AVISO: no se pudo restaurar el estado RNG de CUDA ({e})")



def _optimizer_param_names(model, optimizer) -> list:
    """Devuelve los nombres de parametros de cada param_group del optimizer."""
    name_by_id = {id(param): name for name, param in model.named_parameters()}
    grupos = []
    for group in optimizer.param_groups:
        nombres = []
        for param in group["params"]:
            nombre = name_by_id.get(id(param))
            if nombre is None:
                raise RuntimeError(
                    "Hay un parametro del optimizer que no pertenece a model.named_parameters()."
                )
            nombres.append(nombre)
        grupos.append(nombres)
    return grupos


def _checkpoint_optimizer_param_names(model, ckpt: dict):
    """
    Recupera la correspondencia param_group -> nombres de parametros usada
    al guardar el optimizer.

    Checkpoints nuevos traen `optimizer_param_names` explicitamente. Para
    checkpoints antiguos se intenta reconstruir la estructura a partir del
    `unfreeze_schedule` guardado en `config`.
    """
    if "optimizer_param_names" in ckpt:
        return ckpt["optimizer_param_names"]

    guardado = ckpt.get("optimizer_state_dict")
    if not guardado:
        return None

    cfg_prev = ckpt.get("config", {}) or {}
    old_schedule = cfg_prev.get("unfreeze_schedule")

    # Snapshot: ProgressiveUnfreezer congela el backbone en __init__. La
    # reconstruccion de grupos no debe alterar requires_grad del modelo vivo.
    req_grad = {id(param): param.requires_grad for param in model.parameters()}
    try:
        if old_schedule is None:
            grupos_reconstruidos = [{"params": list(model.parameters())}]
        else:
            from progressive_unfreeze import ProgressiveUnfreezer

            schedule = {int(k): int(v) for k, v in dict(old_schedule).items()}
            old_unfreezer = ProgressiveUnfreezer(
                model, schedule=schedule, verbose=False
            )
            # Los valores concretos de LR no importan para reconstruir la
            # membresia y el orden de los grupos.
            grupos_reconstruidos = old_unfreezer.build_param_groups(
                base_lr=1.0, decay=1.0
            )
    finally:
        for param in model.parameters():
            param.requires_grad = req_grad[id(param)]

    grupos_guardados = guardado.get("param_groups", [])
    if len(grupos_reconstruidos) != len(grupos_guardados):
        return None

    name_by_id = {id(param): name for name, param in model.named_parameters()}
    nombres = []
    for grupo_real, grupo_sd in zip(grupos_reconstruidos, grupos_guardados):
        params_reales = list(grupo_real["params"])
        params_ids = list(grupo_sd["params"])
        if len(params_reales) != len(params_ids):
            return None
        nombres.append([name_by_id[id(param)] for param in params_reales])

    return nombres


def _clone_optimizer_state_for_param(state: dict, param: torch.nn.Parameter):
    """Copia un estado Adam/AdamW al dispositivo/dtype del parametro destino."""
    import copy

    out = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            # exp_avg, exp_avg_sq y max_exp_avg_sq tienen la forma exacta
            # del parametro. `step` suele ser un tensor escalar y conserva
            # su dtype original.
            if key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if tuple(value.shape) != tuple(param.shape):
                    return None
                out[key] = value.detach().clone().to(
                    device=param.device, dtype=param.dtype
                )
            else:
                out[key] = value.detach().clone().to(device=param.device)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _restore_optimizer_state_partial(model, optimizer, ckpt: dict) -> bool:
    """
    Restaura los momentos de Adam/AdamW POR NOMBRE DE PARAMETRO.

    Esto permite continuar el fine-tuning cuando cambia `unfreeze_schedule`:
    - parametros ya optimizados conservan `step`, `exp_avg`, `exp_avg_sq`;
    - capas recien incorporadas quedan con estado Adam limpio;
    - los param_groups, learning rates y weight decay son SIEMPRE los de la
      configuracion actual.

    Devuelve True si pudo reconstruir el mapeo del checkpoint, aunque algunos
    parametros queden necesariamente sin estado previo.
    """
    guardado = ckpt.get("optimizer_state_dict")
    if not guardado:
        print("  AVISO: el checkpoint no contiene optimizer_state_dict.")
        return False

    nombres_prev = _checkpoint_optimizer_param_names(model, ckpt)
    if nombres_prev is None:
        print(
            "  AVISO: no se pudo reconstruir la correspondencia de parametros "
            "del optimizer anterior. Se usan pesos del modelo, pero Adam "
            "arranca limpio."
        )
        return False

    grupos_prev = guardado.get("param_groups", [])
    if len(nombres_prev) != len(grupos_prev):
        print(
            "  AVISO: metadata del optimizer inconsistente; Adam arranca limpio."
        )
        return False

    # state_dict de optimizer usa IDs enteros internos. Los asociamos con
    # nombres usando el orden de cada param_group del checkpoint.
    estado_por_nombre = {}
    for nombres_grupo, grupo_sd in zip(nombres_prev, grupos_prev):
        ids_grupo = list(grupo_sd["params"])
        if len(nombres_grupo) != len(ids_grupo):
            print(
                "  AVISO: tamanos incompatibles al reconstruir el optimizer; "
                "Adam arranca limpio."
            )
            return False
        for nombre, pid in zip(nombres_grupo, ids_grupo):
            estado = guardado.get("state", {}).get(pid)
            if estado:
                estado_por_nombre[nombre] = estado

    name_by_id = {id(param): name for name, param in model.named_parameters()}
    restaurados = 0
    incompatibles = 0
    sin_historial = 0
    por_grupo = []

    # optimizer es nuevo: su `state` esta vacio. Solo inyectamos los estados
    # compatibles; los demas se inicializaran automaticamente en el primer
    # optimizer.step() que reciba gradiente.
    optimizer.state.clear()

    for group in optimizer.param_groups:
        grupo_rest = grupo_total = 0
        for param in group["params"]:
            grupo_total += 1
            nombre = name_by_id[id(param)]
            estado_prev = estado_por_nombre.get(nombre)
            if estado_prev is None:
                sin_historial += 1
                continue

            estado_nuevo = _clone_optimizer_state_for_param(estado_prev, param)
            if estado_nuevo is None:
                incompatibles += 1
                continue

            optimizer.state[param] = estado_nuevo
            restaurados += 1
            grupo_rest += 1

        por_grupo.append((group.get("name", "sin_nombre"), grupo_rest, grupo_total))

    print("  optimizer restaurado PARCIALMENTE por nombre de parametro:")
    for nombre, n_rest, n_total in por_grupo:
        print(f"     {nombre:<24} {n_rest:>4}/{n_total:<4} parametros con estado Adam previo")
    print(
        f"     total restaurados: {restaurados} | sin historial previo: "
        f"{sin_historial} | incompatibles por forma: {incompatibles}"
    )
    print(
        "     lr/weight_decay y estructura de param_groups provienen del "
        "config ACTUAL; el scheduler se construye despues desde cero salvo "
        "que resume_scheduler=true."
    )
    return True

def _build_checkpoint(model, optimizer, scheduler, generator, epoch, global_step,
                      estado_mejores: dict, config: dict, incluir_rng: bool = True) -> dict:
    """
    Checkpoint completo para reanudar exactamente donde se dejo.

    Se guardan los `state_dict()`, NO los objetos: serializar el optimizer
    entero con pickle ata el archivo a la definicion de la clase y, al
    cargarlo, deja el optimizer apuntando a tensores deserializados en vez
    de a los parametros del modelo vivo.
    """
    ck = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # Guardar nombres hace posible transferir Adam por parametro aunque
        # cambie el unfreeze_schedule y, con el, la estructura de grupos.
        "optimizer_param_names": _optimizer_param_names(model, optimizer),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }
    ck.update(estado_mejores)
    if generator is not None:
        ck["dataloader_generator_state"] = generator.get_state()
    if incluir_rng:
        ck["rng_state"] = _rng_state()
    return ck


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
    subject_subset_seed: int = 42,
    resume_from: Optional[str] = None,
    resume_weights_only: bool = True,
    resume_optimizer: Optional[bool] = None,
    resume_scheduler: Optional[bool] = None,
    resume_epoch: Optional[bool] = None,
    resume_restore_rng: Optional[bool] = None,
    balance_classes: bool = False,
    balance_unit: Optional[str] = None,
    windows_per_segment: Optional[int] = None,
    camera_view: Optional[str] = None,
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

    if camera_view is None:
        raise ValueError(
            "camera_view es obligatorio. Def�nelo expl�citamente en el YAML "
            "como camera_view: body o camera_view: face."
        )
    camera_view = str(camera_view).strip().lower()
    if camera_view not in ("body", "face"):
        raise ValueError(
            f"camera_view debe ser 'body' o 'face', no {camera_view!r}"
        )
    print(f"Camera view fijada: {camera_view}")

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

    ds_kwargs = dict(camera_view=camera_view,
                     sequence_length=sequence_length,
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
    # El balanceo va SOLO en train. Aplicarlo a validacion cambiaria el
    # piso trivial y la metrica dejaria de reflejar la distribucion real.
    train_dataset = SegmentDataset(
        os.path.join(data_dir, "TRAIN"),
        balance_classes=balance_classes,
        balance_unit=balance_unit,
        windows_per_segment=windows_per_segment,
        balance_seed=seed if seed is not None else 42,
        subject_subset=subject_subset,
        subject_subset_seed=subject_subset_seed,
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
    # OJO: len(class_to_idx) cuenta ACTIVIDADES, no clases. En el esquema
    # binario hay 5 actividades mapeadas a 2 clases.
    n_out = len(getattr(train_dataset, 'class_names', train_dataset.class_to_idx))
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

    # ------------------------------------------------------------------
    # Inicializacion desde un checkpoint existente
    #
    # resume_weights_only=True (default): carga SOLO los pesos y arranca un
    # entrenamiento nuevo (optimizer y scheduler desde cero, epoca 0). Es lo
    # que corresponde cuando se cambia de tarea -por ejemplo partir del
    # modelo de 3 clases para entrenar uno binario- o cuando se quiere
    # reentrenar con otra configuracion.
    #
    # resume_weights_only=False: reanuda una corrida interrumpida,
    # restaurando tambien el estado del optimizer y del scheduler. Solo
    # tiene sentido si TODO lo demas (lr, num_epochs, dataset) es identico:
    # OneCycleLR guarda su posicion en el ciclo, y reanudarlo con otro
    # presupuesto de epocas deja el learning rate en un punto arbitrario.
    #
    # La ultima capa del clasificador se descarta automaticamente si el
    # numero de clases no coincide: es lo habitual al cambiar de esquema de
    # etiquetado (3 clases -> 2), y sin esto `load_state_dict` fallaria.
    # ------------------------------------------------------------------
    ckpt = None
    _res_opt = _res_sch = _res_ep = _restore_rng = False
    if resume_from is not None:
        # Checkpoint completo de entrenamiento (pesos + optimizer + scheduler
        # + RNG + config). Desde PyTorch 2.6, torch.load usa weights_only=True
        # por defecto; nuestros checkpoints contienen metadata de OmegaConf,
        # por lo que se carga explicitamente en modo completo. Usar solo con
        # checkpoints propios o de una fuente de confianza.
        ckpt = torch.load(
            resume_from,
            map_location=device,
            weights_only=False,
        )
        sd = ckpt.get("model_state_dict", ckpt)

        propio = model.state_dict()
        descartadas = [
            k for k, v in sd.items()
            if k in propio and propio[k].shape != v.shape
        ]
        for k in descartadas:
            del sd[k]

        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Checkpoint cargado: {resume_from}")
        print(f"  epoca de origen : {ckpt.get('epoch', '?')}")

        prev_camera = ckpt.get("config", {}).get("camera_view")
        if prev_camera is not None:
            prev_camera = str(prev_camera).strip().lower()
            if prev_camera != camera_view:
                msg = (
                    f"El checkpoint fue guardado con camera_view={prev_camera!r} "
                    f"pero la corrida actual usa camera_view={camera_view!r}."
                )
                if resume_weights_only:
                    print("  AVISO CAMBIO DE CAMARA: " + msg)
                    print(
                        "  Se permite porque resume_weights_only=True "
                        "(transferencia de pesos deliberada)."
                    )
                else:
                    raise ValueError(
                        msg + " No se permite al reanudar una corrida completa."
                    )
        else:
            print(
                "  AVISO: checkpoint antiguo sin camera_view en metadata; "
                "no es posible verificar autom�ticamente la c�mara de origen."
            )
        if descartadas:
            print(f"  capas descartadas por forma incompatible: {descartadas}")
            print(f"  -> se reinicializan al azar (cambio de numero de clases)")
        if missing:
            print(f"  faltantes  : {missing}")
        if unexpected:
            print(f"  inesperadas: {unexpected}")
        print(f"  modo: {'solo pesos' if resume_weights_only else 'reanudar corrida completa'}")

        # `resume_restore_rng` por defecto sigue al modo: se restaura al
        # reanudar, no al partir de pesos.
        #
        # Restaurar el RNG en modo "solo pesos" hace que el shuffling
        # arranque donde lo dejo la corrida anterior en lugar de en un
        # estado limpio derivado de `seed`. Para un experimento nuevo eso
        # no aporta reproducibilidad -la da `seed`- y ata el orden de los
        # datos a una corrida que ya no es comparable. Se deja disponible
        # por si se quiere continuar el flujo de datos exacto.
        # Cada componente se controla por separado; por defecto siguen al
        # modo (todo al reanudar, nada al partir de pesos).
        _def = not resume_weights_only
        _res_opt = _def if resume_optimizer is None else resume_optimizer
        _res_sch = _def if resume_scheduler is None else resume_scheduler
        _res_ep = _def if resume_epoch is None else resume_epoch
        _restore_rng = _def if resume_restore_rng is None else resume_restore_rng
        if resume_weights_only and _restore_rng:
            _restore_rng_state(ckpt.get("rng_state", {}))
            if generator is not None and "dataloader_generator_state" in ckpt:
                generator.set_state(ckpt["dataloader_generator_state"])
            print("  estado RNG restaurado del checkpoint "
                  "(el shuffling continua el flujo anterior)")

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

    # ------------------------------------------------------------------
    # Estado del optimizer
    #
    # Cuando cambia `unfreeze_schedule`, los param_groups pueden cambiar de
    # tamano y `optimizer.load_state_dict()` deja de ser valido. Para el
    # fine-tuning por etapas se restauran los momentos de Adam por NOMBRE
    # de parametro: las capas ya entrenadas conservan su historial y las
    # recien descongeladas empiezan con Adam limpio. Los LR y param_groups
    # son siempre los de la configuracion actual.
    # ------------------------------------------------------------------
    if ckpt is not None and _res_opt:
        ok_opt = _restore_optimizer_state_partial(model, optimizer, ckpt)
        if not ok_opt:
            _res_opt = False

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
    best_val_macro_f1_seg = 0.0
    epochs_without_improvement = 0
    start_epoch = 0
    global_step = 0

    if ckpt is not None and (_res_sch or _res_ep):
        # El scheduler guarda su POSICION en el ciclo de OneCycleLR.
        # Restaurarlo continua el ciclo anterior, es decir que NO habra
        # warmup: si lo que se busca es un fine-tuning que arranque con
        # warmup desde pesos ya buenos, este flag debe quedar en False.
        if _res_sch and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print("  scheduler restaurado: el ciclo CONTINUA (sin warmup nuevo)")

        if _res_ep:
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_step = int(ckpt.get("global_step", 0))
            best_val_macro_f1 = float(ckpt.get("best_val_macro_f1", 0.0))
            best_val_macro_f1_seg = float(ckpt.get("best_val_macro_f1_segment", 0.0))
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            best_val_acc = float(ckpt.get("best_val_acc", 0.0))
            epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))

        # Estado de los generadores aleatorios: sin esto el shuffling y la
        # augmentation siguen una secuencia distinta a la original.
        _restore_rng_state(ckpt.get("rng_state", {}))
        if generator is not None and "dataloader_generator_state" in ckpt:
            generator.set_state(ckpt["dataloader_generator_state"])

        # Aviso si la configuracion cambio: OneCycleLR guarda su posicion en
        # el ciclo, asi que reanudar con otro num_epochs o lr deja el
        # learning rate en un punto arbitrario.
        cfg_prev = ckpt.get("config", {})
        cfg_ahora = {"num_epochs": num_epochs, "lr": lr,
                     "accumulation_steps": accumulation_steps,
                     "label_mode": label_mode}
        difs = {k: (cfg_prev.get(k), v) for k, v in cfg_ahora.items()
                if k in cfg_prev and cfg_prev[k] != v}
        if difs:
            print("  AVISO: la configuracion cambio respecto del checkpoint.")
            for k, (antes, ahora) in difs.items():
                print(f"     {k}: {antes} -> {ahora}")
            print("     El scheduler retoma su posicion del ciclo ANTERIOR; el "
                  "resultado no equivale ni a la corrida original ni a una nueva.")

        print(f"Reanudando desde la epoca {start_epoch}/{num_epochs}")
        print(f"  global_step {global_step} | mejor macro-F1 segmento "
              f"{best_val_macro_f1_seg:.4f} | epocas sin mejora "
              f"{epochs_without_improvement}")
        if "rng_state" not in ckpt:
            print("  AVISO: el checkpoint no trae estado RNG (es de una version "
                  "anterior). El shuffling no reproduce la corrida original.")
        if start_epoch >= num_epochs:
            print(f"AVISO: el checkpoint ya alcanzo num_epochs={num_epochs}. "
                  "No hay nada que entrenar.")

    for epoch in range(start_epoch, num_epochs):
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
        (val_loss, val_acc, at_least_one_correct_tot, val_macro_f1,
         val_macro_f1_fine, val_macro_f1_seg, val_acc_seg) = validate(
            model, val_loader, criterion, device, epoch, num_epochs,
            max_windows_per_forward=max_windows_per_forward,
            num_classes=num_classes,
            label_groups=label_groups,
        )

        writer.add_scalar("val/loss", val_loss, global_step)
        writer.add_scalar("val/accuracy", val_acc, global_step)
        writer.add_scalar("val/at_least_one_correct", at_least_one_correct_tot, global_step)
        writer.add_scalar("val/macro_f1", val_macro_f1, global_step)
        # Nivel SEGMENTO: la metrica alineada con la unidad de etiquetado,
        # con la unidad de decision de la aplicacion y con la ponderacion
        # de la perdida (window bagging).
        writer.add_scalar("val/macro_f1_segment", val_macro_f1_seg, global_step)
        writer.add_scalar("val/accuracy_segment", val_acc_seg, global_step)
        if val_macro_f1_fine is not None:
            writer.add_scalar("val/macro_f1_fine", val_macro_f1_fine, global_step)

        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)

        improved = val_macro_f1 > best_val_macro_f1
        best_val_macro_f1 = max(best_val_macro_f1, val_macro_f1)

        improved_seg = val_macro_f1_seg > best_val_macro_f1_seg
        best_val_macro_f1_seg = max(best_val_macro_f1_seg, val_macro_f1_seg)

        print(f"  val macro-F1  window: {val_macro_f1:.4f}"
              f"   segmento: {val_macro_f1_seg:.4f}"
              + ("  <- mejor segmento" if improved_seg else ""))

        # Estado que hay que preservar para reanudar exactamente aca.
        # `epochs_without_improvement` se calcula mas abajo, asi que se usa
        # el valor que tendra tras esta epoca.
        _sin_mejora = 0 if improved_seg else epochs_without_improvement + 1
        estado_mejores = {
            "best_val_macro_f1": best_val_macro_f1,
            "best_val_macro_f1_segment": best_val_macro_f1_seg,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "epochs_without_improvement": _sin_mejora,
            "val_macro_f1_window": val_macro_f1,
            "val_macro_f1_segment": val_macro_f1_seg,
        }
        config_ck = {
            "num_epochs": num_epochs,
            "lr": lr,
            "accumulation_steps": accumulation_steps,
            "label_mode": label_mode,
            "num_classes": num_classes,
            "camera_view": camera_view,
            "balance_classes": balance_classes,
            "balance_unit": balance_unit,
            "windows_per_segment": windows_per_segment,
            # Guardar tipos Python puros evita serializar DictConfig dentro
            # del checkpoint y mejora compatibilidad entre versiones.
            "unfreeze_schedule": (
                {int(k): int(v) for k, v in dict(unfreeze_schedule).items()}
                if unfreeze_schedule is not None else None
            ),
            "unfreeze_lr_decay": unfreeze_lr_decay,
            "seed": seed,
        }

        if save_checkpoints:
            ck = _build_checkpoint(
                model, optimizer, scheduler, generator, epoch, global_step,
                estado_mejores, config_ck,
            )

            # checkpoint_last.pth: punto canonico de reanudacion. Se
            # sobrescribe en cada epoca, asi que siempre refleja el estado
            # mas reciente sin acumular disco.
            torch.save(ck, models_dir / "checkpoint_last.pth")

            # DOS checkpoints "best", uno por cada granularidad. El de
            # segmento es el que corresponde usar; se conserva el de ventana
            # para no romper la comparabilidad con corridas anteriores.
            # Los dos maximos NO caen necesariamente en la misma epoca.
            if improved_seg:
                torch.save(ck, models_dir / "model_best_segment.pth")
            if improved:
                torch.save(ck, models_dir / "model_best.pth")

            # Historico por epoca (necesario para el barrido de checkpoints).
            # Sin estado RNG: son ~50 MB cada uno y no se usan para reanudar.
            torch.save(
                _build_checkpoint(
                    model, optimizer, scheduler, generator, epoch, global_step,
                    estado_mejores, config_ck, incluir_rng=False,
                ),
                models_dir / f"model_{epoch}.pth"
            )

        if trial is not None:
            trial.report(val_macro_f1_seg, epoch)
            if trial.should_prune():
                writer.close()
                raise optuna.TrialPruned()

        # El early stopping sigue la metrica de SEGMENTO, que es la
        # alineada con el objetivo.
        if improved_seg:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping en �poca {epoch}: sin mejora de "
                f"val_macro_f1_segment durante {epochs_without_improvement} "
                f"�pocas consecutivas (mejor: {best_val_macro_f1_seg:.4f})"
            )
            break

    writer.close()
    # Se devuelve la metrica de SEGMENTO: es la que debe optimizar
    # cualquier busqueda de hiperparametros sobre este pipeline.
    return best_val_macro_f1_seg