#!/usr/bin/env bash
#
# Espera a que termine el sanity check, verifica sus resultados y, solo si
# pasan, encadena la bateria de experimentos de descongelamiento.
#
# USO
#   # con el PID del proceso padre (recomendado)
#   nohup ./chain_after_sanity.sh 555751 > logs/chain.log 2>&1 &
#
#   # o dejando que lo detecte solo
#   nohup ./chain_after_sanity.sh > logs/chain.log 2>&1 &
#
#   --phase N       fase a encadenar (default: 1)
#   --skip-check    encadena sin verificar el sanity check (no recomendado)
#   --dry-run       muestra que haria, sin ejecutar
#
# IMPORTANTE
#   Correr con `nohup ... &` para que sobreviva al cierre de la terminal.
#   Sin eso, cerrar la sesion mata la cadena entera.
#
#   Usar el PID del proceso PADRE (el que ejecuta main.py), no el de los
#   pt_data_worker: esos son los workers del DataLoader y se recrean en
#   cada epoca, asi que la cadena arrancaria al terminar la epoca actual
#   en lugar de la corrida completa.
#

set -euo pipefail

SANITY_RUN="../runs/sanity_freeze_bn"
PHASE=1
SKIP_CHECK=0
DRY_RUN=0
PID=""
INTERVALO=60         # segundos entre chequeos
AVISO_CADA=15         # minutos entre mensajes de "sigo esperando"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)      PHASE="$2"; shift 2 ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
    -*)           echo "Opcion desconocida: $1" >&2; exit 1 ;;
    *)            PID="$1"; shift ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------
# Deteccion del PID padre
#
# pgrep -f matchea tambien a los workers del DataLoader. El padre es el
# unico cuyo PPID no esta dentro del propio conjunto de coincidencias.
# ---------------------------------------------------------------------
detectar_pid() {
  local patron="main.py config_path=config_sanity.yaml"
  local todos padre
  todos=$(pgrep -f "$patron" || true)
  [[ -z "$todos" ]] && return 1

  while read -r p; do
    [[ -z "$p" ]] && continue
    local ppid
    ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ') || continue
    if ! grep -qx "$ppid" <<< "$todos"; then
      padre="$p"
      break
    fi
  done <<< "$todos"

  [[ -n "${padre:-}" ]] && echo "$padre"
}

if [[ -z "$PID" ]]; then
  log "Detectando el proceso del sanity check..."
  PID=$(detectar_pid || true)
  if [[ -z "$PID" ]]; then
    log "No se encontro ningun proceso de sanity check corriendo."
    log "Si ya termino, ejecutar directamente:"
    log "    python check_sanity.py --run $SANITY_RUN"
    exit 1
  fi
  log "Detectado PID padre: $PID"
fi

# Guarda contra reutilizacion de PID: se registra el cmdline al inicio y se
# vuelve a comparar antes de dar por terminada la espera.
CMDLINE_INICIAL=$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null || echo "")
if [[ -z "$CMDLINE_INICIAL" ]]; then
  log "El PID $PID no existe. ¿Ya termino la corrida?"
  exit 1
fi
log "Esperando al PID $PID"
log "  cmdline: ${CMDLINE_INICIAL}"

# ---------------------------------------------------------------------
# Espera
# ---------------------------------------------------------------------
INICIO=$(date +%s)
CICLOS=0
while kill -0 "$PID" 2>/dev/null; do
  # Si el cmdline cambio, el PID fue reciclado por otro proceso
  ACTUAL=$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null || echo "")
  if [[ "$ACTUAL" != "$CMDLINE_INICIAL" ]]; then
    log "AVISO: el PID $PID fue reutilizado por otro proceso. Se asume terminado."
    break
  fi

  sleep "$INTERVALO"
  CICLOS=$(( CICLOS + 1 ))
  if (( CICLOS % AVISO_CADA == 0 )); then
    TRANSCURRIDO=$(( $(date +%s) - INICIO ))
    log "Sigo esperando... ($(( TRANSCURRIDO / 60 )) min)"
  fi
done

ESPERA=$(( $(date +%s) - INICIO ))
log "El sanity check termino. Espera: $(( ESPERA / 3600 ))h $(( (ESPERA % 3600) / 60 ))m"

# Margen para que se cierre el SummaryWriter y se vuelquen los eventos
sleep 20

# ---------------------------------------------------------------------
# Verificacion
# ---------------------------------------------------------------------
if [[ $SKIP_CHECK -eq 1 ]]; then
  log "Verificacion omitida por --skip-check"
else
  log "Verificando los resultados del sanity check..."
  echo
  if python check_sanity.py --run "$SANITY_RUN"; then
    echo
    log "El sanity check PASA. Se encadena la fase $PHASE."
  else
    echo
    log "El sanity check FALLA. NO se encadenan los experimentos."
    log "Revisar las curvas en TensorBoard antes de seguir:"
    log "    tensorboard --logdir runs/"
    exit 1
  fi
fi

# ---------------------------------------------------------------------
# Encadenado
# ---------------------------------------------------------------------
CMD=(./run_unfreeze_experiments.sh --phase "$PHASE")

echo
log "Lanzando: ${CMD[*]}"
if [[ $DRY_RUN -eq 1 ]]; then
  log "(--dry-run: no se ejecuta)"
  exit 0
fi

exec "${CMD[@]}"