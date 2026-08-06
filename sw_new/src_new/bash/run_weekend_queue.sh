#!/usr/bin/env bash
#
# Cola de experimentos para ejecucion desatendida (fin de semana).
#
# Corre una lista de experimentos en secuencia, cada uno con su propio
# run_name, log y marcador de completitud. Generico: cada entrada elige su
# archivo de config y sus overrides, asi que sirve para mezclar experimentos
# de escalera, de clases finas y de semilla en una misma cola.
#
# USO
#   ./run_weekend_queue.sh --list
#   ./run_weekend_queue.sh --dry-run
#   nohup ./run_weekend_queue.sh --max-hours 55 > logs/weekend.log 2>&1 &
#
#   --max-hours N   no arranca un experimento nuevo si no entra en el
#                   presupuesto (default 55 h). Evita que la cola siga
#                   corriendo el lunes cuando necesites la maquina.
#   --est-hours N   duracion estimada por experimento para ese calculo
#                   (default 5).
#   --only NOMBRE   corre solo ese experimento (repetible).
#   --force         reejecuta aunque exista el marcador .done
#   --dry-run       imprime los comandos sin ejecutarlos
#
# IMPORTANTE
#   Lanzar con `nohup ... &` para que sobreviva al cierre de sesion.
#
#   Los schedules van con espacio despues de los dos puntos y entre
#   comillas: "{0: 4}". La forma compacta {0:4} NO sirve: YAML lee 0:4
#   como notacion sexagesimal y produce un schedule corrupto sin error.
#
#   Los configs deben tener `unfreeze_schedule: null`. Si tuvieran un dict,
#   OmegaConf FUSIONA el override de CLI en vez de reemplazarlo.
#
set -euo pipefail

LOG_DIR="logs/weekend"
PYTHON="${PYTHON:-python}"
MAX_HOURS=55
EST_HOURS=5
DRY_RUN=0
FORCE=0
ONLY=()

# ---------------------------------------------------------------------
# LA COLA
#
# Formato:  nombre | config | overrides | descripcion
#
# Ordenada por informacion aportada, no por afinidad tematica: si la cola
# se corta a mitad, conviene que lo ya corrido sea lo mas decisivo.
# ---------------------------------------------------------------------
QUEUE=(
  # === VARIANZA (4 corridas) ==========================================
  # Ahora son lo mas urgente, no una confirmacion. El barrido de
  # checkpoints mostro que a nivel SEGMENTO -la metrica alineada con el
  # objetivo- la ventaja de lad04 sobre lad00 es de solo +0.030, con
  # desvios de meseta de 0.015-0.024: entre 1.3 y 2 desviaciones. El
  # efecto puede ser real o puede ser ruido, y con una corrida por config
  # no hay forma de distinguirlo. A nivel window la diferencia es +0.126,
  # pero esa metrica sobreestima el efecto porque la agregacion por
  # segmento ya cancela buena parte de los errores de ventana.
  "lad04_seed7|config_unfreeze.yaml|unfreeze_schedule={0: 4} seed=7|lad04 con semilla 7"
  "lad00_seed7|config_unfreeze.yaml|seed=7|CONTROL con semilla 7"
  "lad04_seed123|config_unfreeze.yaml|unfreeze_schedule={0: 4} seed=123|lad04 con semilla 123"
  "lad00_seed123|config_unfreeze.yaml|seed=123|CONTROL con semilla 123"

  # === DONDE EMPIEZA EL COLAPSO (2 corridas) ==========================
  # lad17_all COLAPSO: predijo siempre 'safe' desde la epoca 0 (macro-F1
  # 0.1888 window / 0.1929 segmento, exactamente el piso de ese
  # clasificador degenerado). No es un resultado de rendimiento sino un
  # fallo de entrenamiento.
  #
  # Clave: lad17 usaba decay=0.3, o sea el backbone tenia el MISMO lr que
  # lad04 (9e-5). Lo que lo mato no fue el learning rate sino QUE bloques
  # se liberaron: los tempranos (0-8) extraen bordes y texturas, y
  # perturbarlos destruye todo lo que viene despues.
  #
  # lad08 libera los bloques 9-16; si NO colapsa, el culpable son
  # especificamente los bloques 0-8. lad17_clip prueba si el recorte de
  # gradiente rescata el escalon superior.
  "lad08_blocks8|config_unfreeze.yaml|unfreeze_schedule={0: 8}|Escalon 8 (bloques 9-16). ¿Colapsa tambien?"
  "lad17_clip|config_unfreeze.yaml|unfreeze_schedule={0: 17} grad_clip=1.0|Escalon 17 con recorte de gradiente. ¿Se rescata?"

  # === BARRIDOS DE LEARNING RATE (4 corridas) =========================
  # Sobre lad04, el escalon ganador, para que la profundidad no sea otra
  # variable. El lr=3e-4 nunca se ajusto despues del fix de BatchNorm.
  #
  # lr efectivo del backbone = lr * unfreeze_lr_decay:
  #     decay 0.1 -> 3e-5     decay 0.3 -> 9e-5 (actual)    decay 1.0 -> 3e-4
  "lad04_decay10|config_unfreeze.yaml|unfreeze_schedule={0: 4} unfreeze_lr_decay=1.0|Backbone al mismo lr que la cabeza (3e-4)"
  "lad04_lr1e3|config_unfreeze.yaml|unfreeze_schedule={0: 4} lr=1e-3|max_lr 3.3x mas alto en toda la red"
  "lad04_decay01|config_unfreeze.yaml|unfreeze_schedule={0: 4} unfreeze_lr_decay=0.1|Backbone 3x mas lento (3e-5)"
  "lad04_lr1e4|config_unfreeze.yaml|unfreeze_schedule={0: 4} lr=1e-4|max_lr 3x mas bajo"

  # === PROGRESIVO CONTRA ESTATICO =====================================
  "prog02_to4|config_unfreeze.yaml|unfreeze_schedule={0: 0, 3: 2, 6: 4}|Progresivo 2->4 contra lad04"

  # === PRESUPUESTO DE EPOCAS (~9.4 h) =================================
  # Ultimo a proposito: si algun barrido de lr gano, conviene correr las
  # 24 epocas sobre ESA config. OJO: cambiar num_epochs altera el ciclo
  # entero de OneCycleLR, asi que NO es comparable con la escalera.
  "lad04_e24|config_unfreeze.yaml|unfreeze_schedule={0: 4} num_epochs=24|lad04 con 24 epocas. ¿Sube el techo?"

  # ELIMINADO: lad17_uniform (decay=1.0 sobre los 17 bloques). lad17 ya
  # colapso con decay=0.3; con el backbone 3.3x mas rapido el colapso es
  # casi seguro. Serian 4.7 h para confirmar algo que ya sabemos.
  #
  # Los experimentos de 9 clases finas van en la OTRA maquina.
)

# ---------------------------------------------------------------------
print_list() {
  printf "\n%-18s %-22s %-42s %s\n" "NOMBRE" "CONFIG" "OVERRIDES" "DESCRIPCION"
  printf '%.0s-' {1..150}; printf "\n"
  local i=1
  for e in "${QUEUE[@]}"; do
    IFS='|' read -r n c o d <<< "$e"
    printf "%2d. %-15s %-22s %-42s %s\n" "$i" "$n" "$c" "${o:-(ninguno)}" "$d"
    i=$((i+1))
  done
  printf "\n%d experimentos x ~%s h = ~%s h\n\n" \
    "${#QUEUE[@]}" "$EST_HOURS" "$(( ${#QUEUE[@]} * EST_HOURS ))"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)       print_list; exit 0 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --force)      FORCE=1; shift ;;
    --max-hours)  MAX_HOURS="$2"; shift 2 ;;
    --est-hours)  EST_HOURS="$2"; shift 2 ;;
    --only)       ONLY+=("$2"); shift 2 ;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
    *)            echo "Opcion desconocida: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

# Verificacion previa: los configs referenciados deben existir y tener el
# schedule en null, o los overrides se fusionarian silenciosamente.
declare -A VISTOS=()
for e in "${QUEUE[@]}"; do
  IFS='|' read -r n c o d <<< "$e"
  [[ -n "${VISTOS[$c]:-}" ]] && continue
  VISTOS[$c]=1
  if [[ ! -f "$c" ]]; then
    echo "ERROR: no existe el config '$c' en $(pwd)" >&2; exit 1
  fi
  if ! grep -qE '^\s*unfreeze_schedule:\s*null\s*$' "$c"; then
    echo "ERROR: '$c' debe tener 'unfreeze_schedule: null'." >&2
    echo "       Con un dict ahi, OmegaConf fusiona los overrides de CLI en" >&2
    echo "       vez de reemplazarlos, y los experimentos correrian con un" >&2
    echo "       schedule distinto al declarado, sin dar error." >&2
    exit 1
  fi
done
echo "Configs verificados: ${!VISTOS[*]}"

INICIO_TOTAL=$(date +%s)
declare -a RESUMEN=()

for e in "${QUEUE[@]}"; do
  IFS='|' read -r name config overrides desc <<< "$e"

  if [[ ${#ONLY[@]} -gt 0 ]]; then
    hit=0
    for s in "${ONLY[@]}"; do [[ "$s" == "$name" ]] && hit=1; done
    [[ $hit -eq 0 ]] && continue
  fi

  MARCADOR="$LOG_DIR/$name.done"
  LOG="$LOG_DIR/$name.log"

  if [[ -f "$MARCADOR" && $FORCE -eq 0 ]]; then
    echo "[skip]  $name  (ya completado)"
    RESUMEN+=("$name|omitido|-")
    continue
  fi

  # Presupuesto de tiempo: no arrancar algo que no va a llegar a terminar
  TRANSCURRIDO_H=$(( ( $(date +%s) - INICIO_TOTAL ) / 3600 ))
  if (( TRANSCURRIDO_H + EST_HOURS > MAX_HOURS )); then
    echo "[stop]  $name: quedan $(( MAX_HOURS - TRANSCURRIDO_H )) h del presupuesto"
    echo "        y se estiman $EST_HOURS h. Se corta la cola aca."
    RESUMEN+=("$name|sin tiempo|-")
    break
  fi

  # Los overrides pueden traer espacios dentro de un mismo token
  # ("unfreeze_schedule={0: 4}"), asi que se separan respetando llaves.
  CMD=("$PYTHON" main.py "config_path=$config" "run_name=$name")
  if [[ -n "$overrides" ]]; then
    while IFS= read -r tok; do
      [[ -n "$tok" ]] && CMD+=("$tok")
    done < <(echo "$overrides" | grep -oE '[a-z_]+=(\{[^}]*\}|[^ ]+)')
  fi

  echo
  echo "======================================================================"
  echo " $name"
  echo " $desc"
  echo " inicio: $(date '+%Y-%m-%d %H:%M:%S')   |   transcurrido: ${TRANSCURRIDO_H} h"
  echo "======================================================================"
  printf ' %q' "${CMD[@]}"; echo

  if [[ $DRY_RUN -eq 1 ]]; then
    RESUMEN+=("$name|dry-run|-")
    continue
  fi

  T0=$(date +%s)
  if "${CMD[@]}" 2>&1 | tee "$LOG"; then
    DUR=$(( $(date +%s) - T0 ))
    touch "$MARCADOR"
    echo "[ok]    $name en $(( DUR/3600 ))h $(( (DUR%3600)/60 ))m"
    RESUMEN+=("$name|ok|$(( DUR/60 ))min")
  else
    echo "[FALLO] $name -- ver $LOG" >&2
    RESUMEN+=("$name|FALLO|-")
    # Sin marcador: se reintenta en la proxima pasada sin --force
  fi
done

TOTAL=$(( $(date +%s) - INICIO_TOTAL ))
echo
echo "======================================================================"
echo " RESUMEN   ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "======================================================================"
printf "%-20s %-12s %s\n" "EXPERIMENTO" "ESTADO" "DURACION"
for r in "${RESUMEN[@]}"; do
  IFS='|' read -r n s d <<< "$r"
  printf "%-20s %-12s %s\n" "$n" "$s" "$d"
done
echo
echo "Tiempo total: $(( TOTAL/3600 ))h $(( (TOTAL%3600)/60 ))m"
echo
echo "Al volver, comparar a nivel SEGMENTO (no el val/macro_f1 de TensorBoard,"
echo "que es a nivel ventana y esta desalineado con el objetivo):"
echo
echo "  python sweep_checkpoints.py --runs \\"
echo "      ../models/lad00_frozen ../models/lad02_blocks2 \\"
echo "      ../models/lad04_blocks4 ../models/lad17_all \\"
echo "      ../models/lad08_blocks8 ../models/fine9_frozen \\"
echo "      --csv-out escalera_completa.csv"
echo
echo "Y para la varianza (3 semillas de la misma config):"
echo "  python sweep_checkpoints.py --runs \\"
echo "      ../models/lad04_blocks4 ../models/lad04_seed7 ../models/lad04_seed123"