"""Tests de las funciones puras de eval_model.py (sin dataset ni pesos)."""
import sys
import types

import numpy as np
import torch

# eval_model importa dataset/model, que no existen en este sandbox.
# Se stubean para poder importar las funciones de metricas.
for name, attrs in (
    ("dataset", {"SegmentDataset": object, "segment_collate_fn": lambda x: x}),
    ("model", {"get_model": lambda *a, **k: None}),
):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod

from eval_model import (  # noqa: E402
    confusion_matrix,
    per_class_metrics,
    trivial_macro_f1,
    format_report,
    infer_model_kwargs,
)

# ------------------------------------------------------------------
# TEST 1: orientacion de la matriz (filas=real, columnas=predicho)
# ------------------------------------------------------------------
labels = np.array([0, 0, 0, 1, 1, 2])
preds = np.array([0, 0, 1, 1, 0, 2])
cm = confusion_matrix(preds, labels, 3)
# real=0: 2 predichas 0, 1 predicha 1  -> fila 0 = [2,1,0]
# real=1: 1 predicha 1, 1 predicha 0   -> fila 1 = [1,1,0]
# real=2: 1 predicha 2                 -> fila 2 = [0,0,1]
esperado = np.array([[2, 1, 0], [1, 1, 0], [0, 0, 1]])
assert np.array_equal(cm, esperado), f"orientacion mal:\n{cm}"
print("[TEST 1] OK: filas=real, columnas=predicho")

# ------------------------------------------------------------------
# TEST 2: precision/recall/f1 contra valores calculados a mano
# ------------------------------------------------------------------
m = per_class_metrics(cm)
# clase 0: tp=2, fp=1 (el real=1 predicho 0), fn=1
assert abs(m[0]["precision"] - 2 / 3) < 1e-9
assert abs(m[0]["recall"] - 2 / 3) < 1e-9
assert abs(m[0]["f1"] - 2 / 3) < 1e-9
assert m[0]["support"] == 3
# clase 2: perfecta
assert m[2]["f1"] == 1.0
print("[TEST 2] OK: precision/recall/f1 correctos")

# ------------------------------------------------------------------
# TEST 3: piso trivial con las proporciones reales del proyecto
#         (80.1 / 9.8 / 10.2)  -> esperado 0.2965
# ------------------------------------------------------------------
cm_prop = np.zeros((3, 3), dtype=np.int64)
cm_prop[0, 0] = 801
cm_prop[1, 1] = 98
cm_prop[2, 2] = 102
floor, majority = trivial_macro_f1(cm_prop)
p = 801 / 1001
esperado_floor = (2 * p / (1 + p)) / 3
assert abs(floor - esperado_floor) < 1e-9
assert majority == 0
print(f"[TEST 3] OK: piso trivial = {floor:.4f} (mayoritaria = clase {majority})")
assert abs(floor - 0.2965) < 0.002, f"esperaba ~0.2965, dio {floor:.4f}"

# ------------------------------------------------------------------
# TEST 4: deteccion de COLAPSO (todo predicho como mayoritaria)
# ------------------------------------------------------------------
cm_colapso = np.zeros((3, 3), dtype=np.int64)
cm_colapso[0, 0] = 801
cm_colapso[1, 0] = 98   # todas las reaching predichas como safe
cm_colapso[2, 0] = 102  # todas las unsafe predichas como safe
rep = format_report(cm_colapso, "colapso simulado")
assert "COLAPSO" in rep, "no detecto el colapso"
macro = sum(x["f1"] for x in per_class_metrics(cm_colapso)) / 3
floor_c, _ = trivial_macro_f1(cm_colapso)
assert abs(macro - floor_c) < 1e-9, "en colapso total macro-F1 debe igualar el piso"
print(f"[TEST 4] OK: colapso detectado (macro-F1 {macro:.4f} == piso {floor_c:.4f})")

# ------------------------------------------------------------------
# TEST 5: caso SIN colapso -> mensaje distinto
# ------------------------------------------------------------------
cm_ok = np.array([
    [700, 60, 41],
    [30, 50, 18],
    [35, 20, 47],
], dtype=np.int64)
rep_ok = format_report(cm_ok, "sin colapso simulado")
assert "NO hay colapso" in rep_ok
macro_ok = sum(x["f1"] for x in per_class_metrics(cm_ok)) / 3
floor_ok, _ = trivial_macro_f1(cm_ok)
assert macro_ok > floor_ok
print(f"[TEST 5] OK: sin colapso (macro-F1 {macro_ok:.4f} > piso {floor_ok:.4f})")

# ------------------------------------------------------------------
# TEST 6: clase ausente en el split no rompe (division por cero -> 0)
# ------------------------------------------------------------------
cm_falta = np.array([[10, 0, 0], [0, 0, 0], [2, 0, 5]], dtype=np.int64)
mm = per_class_metrics(cm_falta)
assert mm[1]["f1"] == 0.0 and mm[1]["support"] == 0
format_report(cm_falta, "clase ausente")
print("[TEST 6] OK: clase sin soporte no rompe")

# ------------------------------------------------------------------
# TEST 7: inferencia de arquitectura desde un state_dict sintetico
#         (hidden_dim=128, lstm_layers=1, num_classes=3 -> best trial)
# ------------------------------------------------------------------
hidden, layers, ncls = 128, 1, 3
sd = {}
for l in range(layers):
    for suf in ("", "_reverse"):
        sd[f"lstm.weight_ih_l{l}{suf}"] = torch.zeros(4 * hidden, 960 if l == 0 else 2 * hidden)
        sd[f"lstm.weight_hh_l{l}{suf}"] = torch.zeros(4 * hidden, hidden)
sd["classifier.0.weight"] = torch.zeros(hidden, hidden * 2)
sd["classifier.3.weight"] = torch.zeros(ncls, hidden)
inf = infer_model_kwargs(sd)
assert inf == {"hidden_dim": 128, "lstm_layers": 1, "num_classes": 3}, inf
print(f"[TEST 7] OK: arquitectura inferida = {inf}")

# y con 3 capas / hidden 256
sd2 = {}
for l in range(3):
    for suf in ("", "_reverse"):
        sd2[f"lstm.weight_ih_l{l}{suf}"] = torch.zeros(4 * 256, 960 if l == 0 else 512)
sd2["classifier.0.weight"] = torch.zeros(256, 512)
sd2["classifier.3.weight"] = torch.zeros(3, 256)
assert infer_model_kwargs(sd2) == {"hidden_dim": 256, "lstm_layers": 3, "num_classes": 3}
print("[TEST 7] OK: tambien con lstm_layers=3, hidden_dim=256")

# ------------------------------------------------------------------
# Muestra del reporte
# ------------------------------------------------------------------
print(format_report(cm_colapso, "EJEMPLO: como se veria un COLAPSO"))

print("\n=== TODOS LOS TESTS PASARON ===")