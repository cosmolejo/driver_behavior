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
# CALIBRADO PARA num_epochs=12 (epocas 0..11).
#
# Dos restricciones al elegir las epocas de descongelamiento:
#
#   a) Una etapa con epoca >= num_epochs NUNCA se activa, y no da error.
#      El experimento queda anulado en silencio.
#
#   b) Descongelar sirve de poco si ya casi no queda learning rate. Con
#      OneCycleLR (pct_start=0.1, div_factor=25) y 12 epocas, la fraccion
#      del max_lr al inicio de cada epoca es:
#
#        ep 1: 94%   ep 4: 84%   ep 7: 44%   ep 10:  8%
#        ep 2: 99%   ep 5: 72%   ep 8: 30%   ep 11:  2%
#        ep 3: 93%   ep 6: 59%   ep 9: 18%
#
#      Descongelar en la epoca 10 libera pesos que ya casi no se mueven.
#      Las etapas se ubican en 3, 6 y 8 (93%, 59% y 30% del lr).
#
# Los schedules estan ANIDADOS: exp02 = exp01 + una etapa, exp03 = exp02 +
# una etapa, todos con la primera etapa en la misma epoca. Asi exp01/02/03
# difieren SOLO en profundidad, y exp01/exp04 SOLO en el momento.
#
# Distribucion de parametros del backbone (2.97M en total):
#   ultimos 2 bloques (15-16) -> 32% de los pesos
#   ultimos 4 bloques (13-16) -> 73%
#   ultimos 8 bloques ( 9-16) -> 96%
# Los bloques 0-8 juntos son apenas el 4%, por eso no vale la pena
# descongelarlos: mucho costo de computo, muy poca capacidad extra.
# ---------------------------------------------------------------------
EXPERIMENTS=(
  # === FASE 1: ESCALERA ESTATICA (4 corridas, ~19 h) ==================
  # Cada corrida descongela un numero FIJO de bloques finales desde la
  # epoca 0 y lo mantiene todo el entrenamiento. Aisla "cuanta capacidad"
  # sin mezclarlo con "en que momento descongelar".
  #
  # El schedule {0: N} es exactamente eso: una sola etapa, en la epoca 0.
  #
  # Fraccion de los 2.97M parametros del backbone que queda entrenable:
  #    0 bloques ->   0%      4 bloques (13-16) -> 73%
  #    2 bloques ->  32%      8 bloques ( 9-16) -> 96%
  #   17 bloques -> 100%
  #
  # Se omite el escalon de 8 en esta fase: 96% vs 100% es casi la misma
  # capacidad, los bloques 0-8 juntos son solo el 4% de los pesos.
  #
  # lad00 es imprescindible aunque exp00_frozen ya haya corrido: aquel fue
  # SIN semilla fija, y comparar una corrida sembrada contra una no sembrada
  # reintroduce la varianza por semilla (~0.04 en macro-F1 de segmento) como
  # confusor. De paso, la diferencia entre lad00 y exp00_frozen mide esa
  # varianza para esta configuracion exacta.
  "lad00_frozen|1|null|0.3|Escalon 0: backbone congelado. CONTROL con semilla fija"
  "lad02_blocks2|1|{0: 2}|0.3|Escalon 2: bloques 15-16 entrenables desde ep 0 (32% del backbone)"
  "lad04_blocks4|1|{0: 4}|0.3|Escalon 4: bloques 13-16 entrenables desde ep 0 (73%)"
  "lad17_all|1|{0: 17}|0.3|Escalon 17: backbone completo entrenable desde ep 0 (100%)"

  # === FASE 2: RELLENO Y ABLACION DE LR (2 corridas, ~9.4 h) ==========
  # Solo si la fase 1 muestra una tendencia que valga precisar.
  "lad08_blocks8|2|{0: 8}|0.3|Escalon 8 (96%). Rellena el hueco entre 4 y 17"
  "lad17_all_uniform|2|{0: 17}|1.0|Escalon 17 con lr UNIFORME. = el freeze_backbone=False de Optuna"

  # === FASE 3: DESCONGELAMIENTO PROGRESIVO (5 corridas, ~23 h) ========
  # Solo si la escalera mostro que descongelar ayuda. Recien entonces tiene
  # sentido preguntar si CONVIENE hacerlo de forma progresiva dentro de un
  # mismo entrenamiento, en lugar de fijo desde el principio.
  "prog02_to4|3|{0: 0, 3: 2, 6: 4}|0.3|Progresivo 2->4. Contra lad04 aisla el efecto de PROGRESAR"
  "prog03_to8|3|{0: 0, 3: 2, 6: 4, 8: 8}|0.3|Progresivo 2->4->8. Contra lad08"
  "prog01_to2|3|{0: 0, 3: 2}|0.3|Progresivo solo a 2 bloques. Contra lad02"
  "prog04_to2_early|3|{0: 0, 1: 2}|0.3|Ablacion de TIMING contra prog01_to2"
  "prog05_to4_decay01|3|{0: 0, 3: 2, 6: 4}|0.1|prog02_to4 con lr de backbone 3x mas bajo"
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