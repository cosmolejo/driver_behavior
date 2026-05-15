"""
Genera splits train/val/test del dataset DMD por SUJETO con estratificación
de clases. Guarda los índices en archivos .pt compatibles con el pipeline
existente.

Estrategia:
1. Lee el CSV para identificar sujetos únicos y sus distribuciones de clase.
2. Asigna sujetos a splits con un algoritmo greedy estratificado.
3. Construye la clase DMD para obtener los samples reales (que pueden ser
   más que filas del CSV en modo 'temporal' con stride).
4. Filtra los índices del dataset según el split de sujetos asignado.
5. Guarda los índices en train_indices.pt, val_indices.pt, test_indices.pt.
"""
import re
import random
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig

from data.dmd import DMD  # ajusta el import a tu estructura real


# Path en CSV:   dmd/gF/25/s3/gF_25_s3_<timestamp>_rgb_ann_distraction.json
# file_id final: gF_25_s3_<timestamp>_<camera>_240
CSV_PATH_RE = re.compile(r'dmd/(\w+)/(\d+)/')
FILE_ID_RE = re.compile(r'^(\w+)_(\d+)_')


def subject_from_csv_path(path: str) -> str | None:
    m = CSV_PATH_RE.match(path)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def subject_from_file_id(file_id: str) -> str | None:
    m = FILE_ID_RE.match(file_id)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def stratified_subject_split(
    subject_labels: dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 306638,
):
    """
    Asigna cada sujeto a un split minimizando el déficit cuadrático respecto
    a la distribución global de clases. Algoritmo greedy: sujetos grandes
    primero, asignados al split que más necesita su composición.
    """
    rng = random.Random(seed)

    global_counts = Counter()
    for labels in subject_labels.values():
        global_counts.update(labels)

    targets = {
        'train': {k: v * train_ratio for k, v in global_counts.items()},
        'val':   {k: v * val_ratio for k, v in global_counts.items()},
        'test':  {k: v * (1 - train_ratio - val_ratio) for k, v in global_counts.items()},
    }
    current = {s: Counter() for s in targets}

    subjects = list(subject_labels.keys())
    rng.shuffle(subjects)
    subjects.sort(key=lambda s: -sum(subject_labels[s].values()))

    def deficit_delta(split_name, labels):
        score = 0.0
        for label, target in targets[split_name].items():
            before = target - current[split_name][label]
            after = target - (current[split_name][label] + labels.get(label, 0))
            score += after ** 2 - before ** 2
        return score

    assignment = {}
    for subject in subjects:
        labels = subject_labels[subject]
        best_split = min(targets.keys(), key=lambda s: deficit_delta(s, labels))
        assignment[subject] = best_split
        current[best_split].update(labels)

    return assignment, current, global_counts


def collect_subjects_from_csv(csv_path: Path) -> dict:
    """Lee el CSV y devuelve subject_id -> Counter(label -> count)."""
    subject_labels = defaultdict(Counter)
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            path, label = row[0], row[3].strip()
            sid = subject_from_csv_path(path)
            if sid is None:
                print(f"WARN: no se pudo extraer sujeto de {path!r}")
                continue
            subject_labels[sid][label] += 1
    return dict(subject_labels)


def verify_no_leakage(splits_indices, samples):
    """Comprueba que ningún sujeto aparece en más de un split."""
    split_subjects = {name: set() for name in splits_indices}
    for split_name, indices in splits_indices.items():
        for idx in indices:
            sid = subject_from_file_id(samples[idx]['file_id'])
            if sid is not None:
                split_subjects[split_name].add(sid)

    names = list(split_subjects.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = split_subjects[a] & split_subjects[b]
            assert not overlap, f"LEAKAGE entre {a} y {b}: {overlap}"
    return split_subjects


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Leer CSV
    csv_path = Path(cfg.label_path) / "dmd_vicomtech.csv"
    print(f"Leyendo CSV: {csv_path}")
    subject_labels = collect_subjects_from_csv(csv_path)
    print(f"Sujetos únicos: {len(subject_labels)}")

    # 2. Asignar sujetos a splits
    train_ratio = cfg.get('split', {}).get('train_ratio', 0.7)
    val_ratio = cfg.get('split', {}).get('val_ratio', 0.15)
    seed = cfg.get('split', {}).get('seed', 306638)

    assignment, achieved, global_counts = stratified_subject_split(
        subject_labels, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
    )
    subjects_by_split = defaultdict(set)
    for subj, split_name in assignment.items():
        subjects_by_split[split_name].add(subj)

    # 3. Construir dataset para enumerar samples reales (puede ser >filas CSV
    #    en modo temporal con stride)
    print("\nConstruyendo dataset DMD para enumerar samples reales...")
    full_dataset = DMD(cfg)
    print(f"Total samples generados: {len(full_dataset)}")

    # 4. Filtrar índices del dataset según asignación de sujetos
    splits_indices = {'train': [], 'val': [], 'test': []}
    unassigned = 0
    for idx, sample in enumerate(full_dataset.samples):
        sid = subject_from_file_id(sample['file_id'])
        if sid is None:
            unassigned += 1
            continue
        for split_name, subjects_in_split in subjects_by_split.items():
            if sid in subjects_in_split:
                splits_indices[split_name].append(idx)
                break
        else:
            unassigned += 1

    if unassigned:
        print(f"WARN: {unassigned} samples sin asignar")

    # 5. Verificar no-leakage
    split_subjects = verify_no_leakage(splits_indices, full_dataset.samples)
    print("\n✓ Verificación: cero leakage de sujetos entre splits")

    # 6. Reporte
    total_global = sum(global_counts.values())
    print(f"\n{'='*70}")
    print(f"Distribución global (CSV): {dict(global_counts)} (total={total_global})")
    print(f"{'='*70}")
    print(f"{'Split':<6} {'Sujetos':<9} {'Samples':<9} Distribución")
    print("-" * 70)
    raw_labels = full_dataset.le.inverse_transform(full_dataset.y)
    for name in ['train', 'val', 'test']:
        n_subj = len(split_subjects[name])
        n_samples = len(splits_indices[name])
        sample_dist = Counter(raw_labels[idx] for idx in splits_indices[name])
        total = sum(sample_dist.values()) or 1
        dist_str = ', '.join(f'{k}={v}({v/total:.0%})' for k, v in sorted(sample_dist.items()))
        print(f"{name:<6} {n_subj:<9} {n_samples:<9} {dist_str}")

    print(f"\nSujetos por split:")
    for name in ['train', 'val', 'test']:
        print(f"  {name}: {sorted(split_subjects[name])}")

    # 7. Guardar
    torch.save(splits_indices['train'], '../datasets/train_indices.pt')
    torch.save(splits_indices['val'], '../datasets/val_indices.pt')
    torch.save(splits_indices['test'], '../datasets/test_indices.pt')
    torch.save({
        'seed': seed,
        'ratios': {'train': train_ratio, 'val': val_ratio, 'test': 1 - train_ratio - val_ratio},
        'subject_assignment': assignment,
        'subjects_by_split': {k: sorted(v) for k, v in split_subjects.items()},
    }, '../datasets/split_metadata.pt')

    print("\nGuardado: train_indices.pt, val_indices.pt, test_indices.pt, split_metadata.pt")


if __name__ == '__main__':
    main()