#!/usr/bin/env bash
#
# Bateria de experimentos: descongelamiento progresivo del backbone.
#
# Cada corrida es independiente: run_name propio (asi TensorBoard las
# separa en runs/<nombre>), log propio y marcador de completitud para
# poder retomar sin repetir.
#
# USO
#   ./run_unfreeze_experiments.sh --list          # ver la bateria
#   ./run_unfreeze_experiments.sh --dry-run       # ver comandos sin ejecutar
#   ./run_unfreeze_experiments.sh --phase 1       # correr solo la fase 1
#   ./run_unfreeze_experiments.sh exp00_frozen    # correr experimentos puntuales
#   ./run_unfreeze_experiments.sh                 # correr todo
#
#   --force             reejecuta aunque ya exista el marcador .done
#   --clean-checkpoints borra los model_<epoca>.pth al terminar cada
#                       corrida, conservando solo model_best.pth
#
# NOTA SOBRE EL FORMATO DEL SCHEDULE
#   Los schedules van con espacio despues de los dos puntos y entre
#   comillas: "{0: 0, 6: 2}". La forma compacta {0:0,6:2} NO sirve:
#   YAML interpreta 6:2 como notacion sexagesimal (= 362) y produce un
#   schedule corrupto sin emitir ningun error.
#
set -euo pipefail

CONFIG="config_unfreeze.yaml"
LOG_DIR="logs/unfreeze"
PYTHON="${PYTHON:-python}"

DRY_RUN=0
FORCE=0
CLEAN_CKPT=0
PHASE=""
SELECTED=()

# ---------------------------------------------------------------------
# Definicion de la bateria
#
# Formato:  nombre | fase | unfreeze_schedule | unfreeze_lr_decay | descripcion
#
# Con num_epochs=15, las epocas del schedule equivalen a:
#   epoca  3 -> 20% del entrenamiento
#   epoca  6 -> 40%
#   epoca 10 -> 67%
#   epoca 13 -> 87%
#
# Recordar la distribucion de parametros del backbone (2.97M en total):
#   ultimos 2 bloques (15-16) -> 32% de los pesos
#   ultimos 4 bloques (13-16) -> 73%
#   ultimos 8 bloques ( 9-16) -> 96%
# Los bloques 0-8 juntos son apenas el 4%, por eso no vale la pena
# descongelarlos: mucho costo de computo, muy poca capacidad extra.
# ---------------------------------------------------------------------
EXPERIMENTS=(
  # --- FASE 1: ¿el descongelamiento aporta algo? (2 corridas) ---
  "exp00_frozen|1|null|0.3|CONTROL: backbone 100% congelado todo el entrenamiento"
  "exp02_last4|1|{0: 0, 6: 2, 10: 4}|0.3|Progresivo 2->4 bloques (hasta 73% del backbone)"

  # --- FASE 2: si la fase 1 da positivo, ¿cuanto y cuando? (3 corridas) ---
  "exp01_last2|2|{0: 0, 6: 2}|0.3|Solo ultimos 2 bloques (32%). Menos capacidad que exp02"
  "exp03_last8|2|{0: 0, 6: 2, 10: 4, 13: 8}|0.3|Progresivo 2->4->8 (96%). Mas capacidad que exp02"
  "exp04_last2_early|2|{0: 0, 3: 2}|0.3|Igual a exp01 pero descongela al 20%. Ablacion de TIMING"

  # --- FASE 3: opcional, ajuste fino del lr diferenciado (1 corrida) ---
  "exp05_last4_decay01|3|{0: 0, 6: 2, 10: 4}|0.1|Igual a exp02 con lr de backbone 3x mas bajo"
)

# ---------------------------------------------------------------------
# Parseo de argumentos
# ---------------------------------------------------------------------
print_list() {
  printf "\n%-22s %-6s %-30s %-7s %s\n" "NOMBRE" "FASE" "SCHEDULE" "DECAY" "DESCRIPCION"
  printf '%.0s-' {1..118}; printf "\n"
  for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name phase sched decay desc <<< "$exp"
    printf "%-22s %-6s %-30s %-7s %s\n" "$name" "$phase" "$sched" "$decay" "$desc"
  done
  printf "\n"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)              print_list; exit 0 ;;
    --dry-run)           DRY_RUN=1; shift ;;
    --force)             FORCE=1; shift ;;
    --clean-checkpoints) CLEAN_CKPT=1; shift ;;
    --phase)             PHASE="$2"; shift 2 ;;
    -h|--help)           sed -n '2,30p' "$0"; exit 0 ;;
    -*)                  echo "Opcion desconocida: $1" >&2; exit 1 ;;
    *)                   SELECTED+=("$1"); shift ;;
  esac
done

# ---------------------------------------------------------------------
# Verificaciones previas
# ---------------------------------------------------------------------
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: no se encuentra $CONFIG en $(pwd)" >&2
  exit 1
fi

# El schedule DEBE venir en null en el config base, o los overrides de CLI
# se fusionan en lugar de reemplazar y los experimentos quedan corruptos.
if ! grep -qE '^\s*unfreeze_schedule:\s*null\s*$' "$CONFIG"; then
  echo "ERROR: '$CONFIG' debe tener 'unfreeze_schedule: null'." >&2
  echo "       Con un dict ahi, OmegaConf FUSIONA los overrides de CLI en" >&2
  echo "       vez de reemplazarlos y cada experimento correria con un" >&2
  echo "       schedule distinto al declarado, sin dar error." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------
# Ejecucion
# ---------------------------------------------------------------------
declare -a RESUMEN=()
INICIO_TOTAL=$(date +%s)

for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name phase sched decay desc <<< "$exp"

  # Filtros de seleccion
  if [[ ${#SELECTED[@]} -gt 0 ]]; then
    encontrado=0
    for s in "${SELECTED[@]}"; do [[ "$s" == "$name" ]] && encontrado=1; done
    [[ $encontrado -eq 0 ]] && continue
  fi
  if [[ -n "$PHASE" && "$phase" != "$PHASE" ]]; then
    continue
  fi

  MARCADOR="$LOG_DIR/$name.done"
  LOG="$LOG_DIR/$name.log"

  if [[ -f "$MARCADOR" && $FORCE -eq 0 ]]; then
    echo "[skip]  $name  (ya completado; usar --force para repetir)"
    RESUMEN+=("$name|omitido|-")
    continue
  fi

  # Se arma como array para que el schedule viaje como UN solo argumento
  # pese a contener espacios.
  CMD=("$PYTHON" main.py
       "config_path=$CONFIG"
       "run_name=$name"
       "unfreeze_lr_decay=$decay"
       "unfreeze_schedule=$sched")

  echo
  echo "======================================================================"
  echo " $name  (fase $phase)"
  echo " $desc"
  echo " schedule: $sched   |   lr_decay: $decay"
  echo "======================================================================"
  printf ' %q' "${CMD[@]}"; echo

  if [[ $DRY_RUN -eq 1 ]]; then
    RESUMEN+=("$name|dry-run|-")
    continue
  fi

  INICIO=$(date +%s)
  if "${CMD[@]}" 2>&1 | tee "$LOG"; then
    FIN=$(date +%s)
    DUR=$(( FIN - INICIO ))
    touch "$MARCADOR"
    echo "[ok]    $name terminado en $(( DUR / 3600 ))h $(( (DUR % 3600) / 60 ))m"
    RESUMEN+=("$name|ok|$(( DUR / 60 ))min")

    if [[ $CLEAN_CKPT -eq 1 ]]; then
      # Conserva unicamente model_best.pth (el de mejor val_macro_f1)
      find "models/$name" -name 'model_[0-9]*.pth' -delete 2>/dev/null || true
      echo "[clean] checkpoints por epoca de $name eliminados"
    fi
  else
    echo "[FALLO] $name  -- ver $LOG" >&2
    RESUMEN+=("$name|FALLO|-")
    # Sin marcador: se puede reintentar sin --force
  fi
done

# ---------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------
FIN_TOTAL=$(date +%s)
TOTAL=$(( FIN_TOTAL - INICIO_TOTAL ))

echo
echo "======================================================================"
echo " RESUMEN"
echo "======================================================================"
printf "%-24s %-10s %s\n" "EXPERIMENTO" "ESTADO" "DURACION"
for r in "${RESUMEN[@]}"; do
  IFS='|' read -r n e d <<< "$r"
  printf "%-24s %-10s %s\n" "$n" "$e" "$d"
done
echo
echo "Tiempo total: $(( TOTAL / 3600 ))h $(( (TOTAL % 3600) / 60 ))m"
echo
echo "Para comparar en TensorBoard:"
echo "    tensorboard --logdir runs/"
echo
echo "Metricas clave a contrastar entre corridas:"
echo "    val/macro_f1            <- la que decide"
echo "    train/epoch_macro_f1    <- para leer la brecha de generalizacion"
echo "    val/loss                <- para detectar divergencias"
echo
echo "Evaluacion detallada del mejor checkpoint de una corrida:"
echo "    python eval_model.py --checkpoint models/<nombre>/model_best.pth"