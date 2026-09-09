#!/usr/bin/env bash
#
# Cola de la ultima noche antes del cierre.
#
# USO
#   nohup ./run_last_night.sh > logs/last_night.log 2>&1 &
#   nohup ./run_last_night.sh --after PID > logs/last_night.log 2>&1 &
#
#   --after PID   espera a que termine ese proceso antes de arrancar
#   --dry-run     imprime los comandos sin ejecutarlos
#
# El PYTHON es la ruta ABSOLUTA al interprete del env: `bash -c` arranca un
# shell no interactivo que NO carga la inicializacion de conda, asi que un
# `python` pelado falla con "command not found".
#
set -euo pipefail

PYTHON="${PYTHON:-/home/agomez/miniconda3/envs/tesis/bin/python}"
CONFIG_LC="configs/config_learning_curve.yaml"
CONFIG_UNF="configs/config_unfreeze.yaml"
LOG_DIR="logs/last_night"
AFTER=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --after)   AFTER="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Opcion desconocida: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: no existe el interprete $PYTHON" >&2
  echo "       Ajustar con: PYTHON=/ruta/al/python ./run_last_night.sh" >&2
  exit 1
fi

# ---------------------------------------------------------------------
# Cola: nombre | config | overrides | descripcion
#
# La curva de aprendizaje va primero: es el experimento que convierte la
# afirmacion "el techo esta en los datos" de inferencia en evidencia
# directa, y es lo que mas material da para escribir durante el cierre.
#
# Ordenada de MENOS a MAS sujetos: los puntos chicos corren mas rapido, asi
# que si la noche se corta igual quedan 2-3 puntos de curva utiles.
#
# El punto de 21 sujetos NO esta: ya existe como lad04_blocks4 (n=3).
# ---------------------------------------------------------------------
QUEUE=(
  "lc_s05|$CONFIG_LC|subject_subset=5|Curva: 5 sujetos (~1.2 h)"
  "lc_s10|$CONFIG_LC|subject_subset=10|Curva: 10 sujetos (~2.4 h)"
  "lc_s15|$CONFIG_LC|subject_subset=15|Curva: 15 sujetos (~3.5 h)"
  "lad04_ceweight|$CONFIG_UNF|loss_fn=CE_weight|Pesos de clase: la unica palanca sin probar contra 'unsafe'"
)

echo "Interprete: $PYTHON"
if [[ -n "$AFTER" ]]; then
  echo "Esperando al PID $AFTER..."
  while kill -0 "$AFTER" 2>/dev/null; do sleep 60; done
  echo "PID $AFTER termino. Arrancando."
  sleep 30
fi

INICIO=$(date +%s)
declare -a RESUMEN=()

for e in "${QUEUE[@]}"; do
  IFS='|' read -r name config overrides desc <<< "$e"
  MARCADOR="$LOG_DIR/$name.done"

  if [[ -f "$MARCADOR" ]]; then
    echo "[skip]  $name (ya completado)"
    RESUMEN+=("$name|omitido|-"); continue
  fi

  CMD=("$PYTHON" main.py "config_path=$config" "run_name=$name")
  while IFS= read -r tok; do
    [[ -n "$tok" ]] && CMD+=("$tok")
  done < <(echo "$overrides" | grep -oE '[a-z_]+=(\{[^}]*\}|[^ ]+)')

  echo
  echo "======================================================================"
  echo " $name  -- $desc"
  echo " $(date '+%Y-%m-%d %H:%M:%S')"
  echo "======================================================================"
  printf ' %q' "${CMD[@]}"; echo

  if [[ $DRY_RUN -eq 1 ]]; then RESUMEN+=("$name|dry-run|-"); continue; fi

  T0=$(date +%s)
  if "${CMD[@]}" 2>&1 | tee "$LOG_DIR/$name.log"; then
    D=$(( $(date +%s) - T0 )); touch "$MARCADOR"
    echo "[ok]    $name en $(( D/3600 ))h $(( (D%3600)/60 ))m"
    RESUMEN+=("$name|ok|$(( D/60 ))min")
  else
    echo "[FALLO] $name -- ver $LOG_DIR/$name.log" >&2
    RESUMEN+=("$name|FALLO|-")
  fi
done

TOTAL=$(( $(date +%s) - INICIO ))
echo
echo "======================================================================"
printf "%-20s %-10s %s\n" "EXPERIMENTO" "ESTADO" "DURACION"
for r in "${RESUMEN[@]}"; do
  IFS='|' read -r n s d <<< "$r"; printf "%-20s %-10s %s\n" "$n" "$s" "$d"
done
echo
echo "Total: $(( TOTAL/3600 ))h $(( (TOTAL%3600)/60 ))m"
echo
echo "Al volver:"
echo "  python sweep_checkpoints.py --config $CONFIG_UNF --csv-out curva.csv \\"
echo "      --runs ../models/lc_s05 ../models/lc_s10 ../models/lc_s15 ../models/lad04_blocks4"
echo
echo "Comparar MEDIAS DE MESETA (epocas 5-11) a nivel SEGMENTO."
echo "Referencia de 21 sujetos: lad04_blocks4 = 0.5631 +- 0.0036 (n=3)."