"""
Etiquetado de grano fino (9 actividades) con agregacion a las 3
macro-clases en inferencia.

Motivacion
----------
Las macro-clases no son igual de coherentes:

    safe      = 1 actividad   (safe_drive)
    reaching  = 2 actividades (reach_side 96.4%, reach_backseat 3.6%)
    unsafe    = 6 actividades (talking_to_passenger 39.7%, texting_right,
                               texting_left, phonecall_left, phonecall_right,
                               radio)

`unsafe` es un concepto DISYUNTIVO: agrupa seis actividades visualmente
distintas, con duraciones medianas que van de 28 frames
(talking_to_passenger) a 1014 (phonecall_right), un rango de 36x. Pedirle a
la red que aprenda "estas seis cosas son la misma" es mas dificil que
aprender cada una por separado, y es consistente con que `unsafe` sea la
peor clase en todas las evaluaciones y con que agregar capacidad al backbone
no la mejore.

La alternativa: entrenar con las 9 clases finas (cada una visualmente
coherente) y agregar a 3 en inferencia sumando las probabilidades de las
componentes. La agregacion es determinista, asi que se puede seguir
reportando macro-F1 a 3 clases y comparar contra toda la bateria anterior.

Estructura de directorios
-------------------------
El dataset esta organizado POR MACRO-CLASE:

    root/<macro>/<video_folder>/<segment_folder>/face/*.jpg

La actividad fina no aparece en el arbol de carpetas, hay que recuperarla de
`partition_report.csv` con la clave (video_folder, segment_folder), que es
unica: 6287 pares para 6287 filas.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# Orden alfabetico, igual criterio que `sorted(os.listdir())` en dataset.py
MACRO_CLASSES = ["reaching", "safe", "unsafe"]


def _read_partition_report(csv_path: str):
    """Lee el CSV sin depender de pandas (puede no estar en el entorno)."""
    import csv as _csv

    filas = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for fila in _csv.DictReader(fh):
            filas.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                          for k, v in fila.items()})
    return filas


def load_activity_lookup(csv_path: str) -> Dict[Tuple[str, str], str]:
    """
    Devuelve {(video_folder, segment_folder): activity}.

    Esa clave es la que puede reconstruirse desde la ruta en disco:
    dataset.py recorre root/<macro>/<session>/<video>/face, donde
    session == video_folder y video == segment_folder.
    """
    lookup = {}
    for f in _read_partition_report(csv_path):
        lookup[(f["video_folder"], f["segment_folder"])] = f["activity"]
    return lookup


def build_fine_class_to_idx(csv_path: str) -> Dict[str, int]:
    """
    {activity: indice}, en orden alfabetico para replicar el criterio de
    `sorted()` que usa dataset.py con las macro-clases.
    """
    acts = sorted({f["activity"] for f in _read_partition_report(csv_path)})
    return {a: i for i, a in enumerate(acts)}


def build_fine_to_macro(csv_path: str) -> Dict[str, str]:
    """{activity: macro_class}, derivado del CSV (no hardcodeado)."""
    out = {}
    for f in _read_partition_report(csv_path):
        a, l = f["activity"], f["label"]
        if a in out and out[a] != l:
            raise ValueError(
                f"La actividad '{a}' aparece con macro-clases distintas: "
                f"'{out[a]}' y '{l}'. El mapeo fino->macro no es una funcion."
            )
        out[a] = l
    return out


def build_label_groups(csv_path: str) -> List[int]:
    """
    Lista de largo 9 donde `groups[i]` es el indice de macro-clase de la
    clase fina `i`. Es lo que consume `aggregate_probs`.
    """
    fine_idx = build_fine_class_to_idx(csv_path)
    fine2macro = build_fine_to_macro(csv_path)
    groups = [0] * len(fine_idx)
    for act, i in fine_idx.items():
        groups[i] = MACRO_CLASSES.index(fine2macro[act])
    return groups


def aggregate_probs(
    logits: torch.Tensor,
    label_groups: List[int],
    num_macro: int = len(MACRO_CLASSES),
) -> torch.Tensor:
    """
    Agrega logits de N clases finas a probabilidades de `num_macro` clases,
    sumando las probabilidades de las componentes de cada grupo:

        P(unsafe) = P(texting_left) + P(texting_right) + P(phonecall_left)
                  + P(phonecall_right) + P(radio) + P(talking_to_passenger)

    Es la agregacion correcta: la probabilidad de la union de eventos
    mutuamente excluyentes es la suma de sus probabilidades. Sumar los
    logits en vez de las probabilidades NO seria equivalente.

    logits: (B, n_fine)  ->  devuelve (B, num_macro), suma 1 por fila.
    """
    probs = torch.softmax(logits, dim=1)
    idx = torch.as_tensor(label_groups, dtype=torch.long, device=probs.device)
    out = torch.zeros(probs.shape[0], num_macro, dtype=probs.dtype, device=probs.device)
    out.index_add_(1, idx, probs)
    return out


def map_fine_to_macro(fine_labels: torch.Tensor, label_groups: List[int]) -> torch.Tensor:
    """Traduce indices de clase fina a indices de macro-clase."""
    idx = torch.as_tensor(label_groups, dtype=torch.long, device=fine_labels.device)
    return idx[fine_labels]


def summary(csv_path: str) -> str:
    fine_idx = build_fine_class_to_idx(csv_path)
    fine2macro = build_fine_to_macro(csv_path)
    groups = build_label_groups(csv_path)

    filas = _read_partition_report(csv_path)
    conteo = {}
    for f in filas:
        conteo.setdefault(f["activity"], {}).setdefault(f["split"], 0)
        conteo[f["activity"]][f["split"]] += 1

    out = [f"{len(fine_idx)} clases finas -> {len(MACRO_CLASSES)} macro-clases", ""]
    out.append(f"{'idx':>4}  {'actividad':<24}{'macro':<12}{'grp':>4}"
               f"{'TRAIN':>8}{'VAL':>6}{'TEST':>6}")
    out.append("-" * 68)
    for act, i in sorted(fine_idx.items(), key=lambda kv: kv[1]):
        c = conteo.get(act, {})
        out.append(
            f"{i:>4}  {act:<24}{fine2macro[act]:<12}{groups[i]:>4}"
            f"{c.get('TRAIN',0):>8}{c.get('VALIDATION',0):>6}{c.get('TEST',0):>6}"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------
# Submuestreo por sujeto (curva de aprendizaje)
# ---------------------------------------------------------------------
def load_subject_lookup(csv_path: str) -> Dict[Tuple[str, str], str]:
    """{(video_folder, segment_folder): subject}."""
    return {(f["video_folder"], f["segment_folder"]): f["subject"]
            for f in _read_partition_report(csv_path)}


def select_subjects(csv_path: str, n: int, split: str = "TRAIN",
                    seed: int = 42) -> List[str]:
    """
    Elige `n` sujetos de `split` de forma DETERMINISTA y ANIDADA:
    select(5) siempre es subconjunto de select(10), etc.

    El anidamiento importa: si los conjuntos no estuvieran anidados, la
    diferencia entre dos puntos de la curva podria venir de QUE sujetos
    tocaron y no de CUANTOS.

    La seleccion es round-robin entre grupos (gA, gB, ...) en vez de un
    shuffle plano. Sin eso, un subconjunto chico podria caer entero en un
    solo grupo y "menos sujetos" quedaria confundido con "menos diversidad
    de grupo", que es otra variable.
    """
    import random as _random

    filas = [f for f in _read_partition_report(csv_path) if f["split"] == split]
    sujetos = sorted({f["subject"] for f in filas})
    if n >= len(sujetos):
        return sujetos

    rnd = _random.Random(seed)
    por_grupo = {}
    for s in sujetos:
        por_grupo.setdefault(s.split("_")[0], []).append(s)
    for g in por_grupo:
        rnd.shuffle(por_grupo[g])

    grupos = sorted(por_grupo)
    rnd.shuffle(grupos)

    orden, i = [], 0
    while len(orden) < len(sujetos):
        g = grupos[i % len(grupos)]
        if por_grupo[g]:
            orden.append(por_grupo[g].pop())
        i += 1
    return sorted(orden[:n])


# ---------------------------------------------------------------------
# Esquemas de etiquetado
# ---------------------------------------------------------------------
# Un esquema define QUE actividades entran y a que indice de clase van.
# Las actividades ausentes del mapa se EXCLUYEN del dataset.
#
# Motivacion del esquema binario: `unsafe` agrupa 6 actividades con
# duraciones medianas de 28 a 1014 frames y apariencias muy distintas
# (hablar con el pasajero, textear, llamar, manipular la radio). Aislar
# un subconjunto visualmente coherente -uso de telefono- convierte la
# tarea en un problema mucho mejor definido.
#
# Nota: el desbalance se INVIERTE segun la unidad de conteo. En TRAIN,
# safe:phone es 3.26:1 por segmento pero 0.97:1 por ventana, porque las
# llamadas duran ~1000 frames contra 148 de safe_drive. Con window
# bagging la perdida ve la proporcion por SEGMENTO.

PHONE_ACTIVITIES = [
    "texting_left", "texting_right", "phonecall_left", "phonecall_right",
]

LABEL_SCHEMES = {
    # 2 clases: conduccion normal vs uso de telefono.
    # Excluye reach_side, reach_backseat, talking_to_passenger y radio.
    "binary_phone": {
        "map": {"safe_drive": 0, **{a: 1 for a in PHONE_ACTIVITIES}},
        "names": ["safe", "phone"],
    },
    # 2 clases: conduccion normal vs gestos de alcance.
    # Excluye las 6 actividades de `unsafe`.
    #
    # A diferencia de binary_phone, aca el desbalance por SEGMENTO es casi
    # perfecto (1.11:1) pero por VENTANA llega a 4.07:1, porque reach_side
    # dura 62 frames de mediana contra 148 de safe_drive. Es el caso
    # opuesto al binario de telefono y una buena ilustracion de por que la
    # unidad de conteo importa.
    #
    # `reach_backseat` aporta solo el 3.6% de la clase (65 segmentos en
    # train): en la practica el modelo aprende sobre todo `reach_side`.
    "binary_reaching": {
        "map": {"safe_drive": 0, "reach_side": 1, "reach_backseat": 1},
        "names": ["safe", "reaching"],
    },
    # 3 clases pero solo con las actividades coherentes: separa el uso de
    # telefono de los gestos de alcance, descartando las ambiguas
    # (talking_to_passenger, radio).
    "ternary_clean": {
        "map": {"safe_drive": 0, "reach_side": 1, "reach_backseat": 1,
                **{a: 2 for a in PHONE_ACTIVITIES}},
        "names": ["safe", "reaching", "phone"],
    },
}


def get_scheme(name: str, csv_path: Optional[str] = None):
    """
    Devuelve (activity_map, class_names) para un esquema.

    - "macro": None, se usa el nombre de carpeta (comportamiento historico)
    - "fine" : las 9 actividades, mapa derivado del CSV
    - resto  : definido en LABEL_SCHEMES
    """
    if name == "macro":
        return None, list(MACRO_CLASSES)
    if name == "fine":
        if csv_path is None:
            raise ValueError("el esquema 'fine' requiere partition_report")
        idx = build_fine_class_to_idx(csv_path)
        return dict(idx), [a for a, _ in sorted(idx.items(), key=lambda kv: kv[1])]
    if name in LABEL_SCHEMES:
        e = LABEL_SCHEMES[name]
        return dict(e["map"]), list(e["names"])
    raise ValueError(
        f"esquema desconocido: {name!r}. "
        f"Opciones: 'macro', 'fine', {list(LABEL_SCHEMES)}"
    )


def scheme_summary(name: str, csv_path: str) -> str:
    amap, names = get_scheme(name, csv_path)
    filas = _read_partition_report(csv_path)
    out = [f"Esquema '{name}': {len(names)} clases -> {names}"]
    if amap is None:
        out.append("  (etiquetas tomadas del nombre de carpeta)")
        return "\n".join(out)

    conteo = {}
    for f in filas:
        a = f["activity"]
        if a in amap:
            conteo.setdefault(a, {}).setdefault(f["split"], 0)
            conteo[a][f["split"]] += 1
    excluidas = sorted({f["activity"] for f in filas} - set(amap))

    out.append(f"  {'actividad':<24}{'clase':<10}{'TRAIN':>7}{'VAL':>6}{'TEST':>6}")
    for a in sorted(amap, key=lambda x: (amap[x], x)):
        c = conteo.get(a, {})
        out.append(f"  {a:<24}{names[amap[a]]:<10}"
                   f"{c.get('TRAIN',0):>7}{c.get('VALIDATION',0):>6}{c.get('TEST',0):>6}")
    if excluidas:
        out.append(f"  EXCLUIDAS: {excluidas}")
    return "\n".join(out)