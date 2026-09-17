#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_can_experiments.sh
#
# Runs the final Phone Use pipeline sequentially over all BODY
# demo videos and the CAN scenarios that fit each video's length.
#
# This variant uses the PyTorch checkpoint on CUDA to avoid skipped
# iterations during the CAN-rule evaluation.
#
# Experimental design:
#   - cobertura_compacta (~55 s): ALL BODY videos
#   - urbano_semaforos (~150 s): videos >= 155 s
#   - autovia_noche_lluvia (~150 s): videos >= 155 s
#   - dia_a_noche (~150 s): videos >= 155 s
#
# Scenarios are injected with --inject-loop, so the CAN timeline repeats
# for the whole length of each video instead of freezing on its final
# values once the script ends. Without the loop the tail of every run was
# a frozen CAN state (61-81% of rows), which both diluted the statistics
# and left M4:low_beam stuck active in the two night scenarios.
#
# The length restriction above still applies with looping: a video shorter
# than the scenario only ever sees the scenario's opening seconds, so it
# would never reach full rule coverage.
#
# The original ~590 s cobertura_completa scenario is intentionally
# not run because none of the current demo videos is long enough.
# ============================================================

# ---------- EDIT THESE PATHS ----------
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Set FORCE=1 to regenerate CSVs that already exist. Required when
# re-running after a methodological change such as adding --inject-loop.
FORCE="${FORCE:-0}"

PIPELINE="${PIPELINE:-./pipeline_can_evaluation.py}"
VIDEO_DIR="${VIDEO_DIR:-/Datos/Demo_vids}"

CONFIG="${CONFIG:-configs/config_binary_phone_balanced.yaml}"
CHECKPOINT="${CHECKPOINT:-/Datos/model_best_segment.pth}"
DEVICE="${DEVICE:-cuda}"

SCENARIO_DIR="${SCENARIO_DIR:-.}"
SCENARIO_COMPACT="${SCENARIO_COMPACT:-$SCENARIO_DIR/escenario_cobertura_compacta.yaml}"
SCENARIO_URBAN="${SCENARIO_URBAN:-$SCENARIO_DIR/escenario_urbano_semaforos.yaml}"
SCENARIO_HIGHWAY="${SCENARIO_HIGHWAY:-$SCENARIO_DIR/escenario_autovia_noche_lluvia.yaml}"
SCENARIO_DAYNIGHT="${SCENARIO_DAYNIGHT:-$SCENARIO_DIR/escenario_dia_a_noche.yaml}"

OUTPUT_DIR="${OUTPUT_DIR:-./can_results}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

# ---------- FINAL PIPELINE SETTINGS ----------
SEGMENT_WINDOWS="${SEGMENT_WINDOWS:-2}"
WINDOW_STRIDE="${WINDOW_STRIDE:-8}"
INFERENCE_EVERY="${INFERENCE_EVERY:-8}"
N_PERSIST="${N_PERSIST:-3}"
THRESHOLD="${THRESHOLD:-0.5}"

CAN_CHANNEL="${CAN_CHANNEL:-vcan0}"
CAN_INTERFACE="${CAN_INTERFACE:-socketcan}"

# Give a few seconds of margin beyond the nominal 150 s scenarios.
MIN_LONG_VIDEO_S="${MIN_LONG_VIDEO_S:-155}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

die() {
    echo "ERROR: $*" >&2
    exit 1
}

video_duration() {
    ffprobe -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        "$1"
}

float_ge() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'
}

safe_name() {
    basename "$1" .mp4 | sed 's/[^A-Za-z0-9._-]/_/g'
}

run_one() {
    local video="$1"
    local scenario="$2"
    local scenario_tag="$3"

    local vname
    vname="$(safe_name "$video")"

    local csv_out="$OUTPUT_DIR/${vname}__${scenario_tag}.csv"
    local log_out="$LOG_DIR/${vname}__${scenario_tag}.log"

    if [[ -s "$csv_out" ]]; then
        if [[ "$FORCE" == "1" ]]; then
            echo "FORCE: regenerating existing $csv_out"
            rm -f "$csv_out"
        else
            echo "SKIP: already exists: $csv_out"
            return 0
        fi
    fi

    echo
    echo "============================================================"
    echo "VIDEO    : $(basename "$video")"
    echo "SCENARIO : $scenario_tag"
    echo "CSV      : $csv_out"
    echo "============================================================"


    set +e
    "$PYTHON_BIN" "$PIPELINE" \
        --body-video "$video" \
        --body-config "$CONFIG" \
        --body-checkpoint "$CHECKPOINT" \
        --device "$DEVICE" \
        --segment-windows "$SEGMENT_WINDOWS" \
        --window-stride "$WINDOW_STRIDE" \
        --inference-every "$INFERENCE_EVERY" \
        --n-persist "$N_PERSIST" \
        --threshold "$THRESHOLD" \
        --can-channel "$CAN_CHANNEL" \
        --can-interface "$CAN_INTERFACE" \
        --inject-scenario "$scenario" \
        --inject-loop \
        --csv "$csv_out" \
        --verbose \
        2>&1 | tee "$log_out"

    status=${PIPESTATUS[0]}
    set -e

    if [[ $status -ne 0 ]]; then
        echo "FAILED: ${vname} / ${scenario_tag} (exit $status)" >&2
        return "$status"
    fi

    if [[ ! -s "$csv_out" ]]; then
        echo "FAILED: CSV was not created: $csv_out" >&2
        return 1
    fi

    echo "DONE: $csv_out"
}

# ------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------

command -v ffprobe >/dev/null 2>&1 || die "ffprobe not found. Install ffmpeg."
[[ -f "$PIPELINE" ]] || die "Pipeline not found: $PIPELINE"
[[ -d "$VIDEO_DIR" ]] || die "Video directory not found: $VIDEO_DIR"
[[ -f "$CONFIG" ]] || die "Config not found: $CONFIG"
[[ -f "$CHECKPOINT" ]] || die "Checkpoint not found: $CHECKPOINT"

for f in \
    "$SCENARIO_COMPACT" \
    "$SCENARIO_URBAN" \
    "$SCENARIO_HIGHWAY" \
    "$SCENARIO_DAYNIGHT"
do
    [[ -f "$f" ]] || die "Scenario not found: $f"
done

if ! ip link show "$CAN_CHANNEL" >/dev/null 2>&1; then
    die "CAN interface '$CAN_CHANNEL' does not exist or is not available."
fi

echo "Pipeline     : $PIPELINE"
echo "Checkpoint   : $CHECKPOINT"
echo "Device       : $DEVICE"
echo "Videos       : $VIDEO_DIR"
echo "Output       : $OUTPUT_DIR"
echo "CAN          : $CAN_CHANNEL ($CAN_INTERFACE)"
echo


# ------------------------------------------------------------
# Collect BODY videos only
# ------------------------------------------------------------

mapfile -t VIDEOS < <(
    find "$VIDEO_DIR" -maxdepth 1 -type f \
        -name '*_rgb_body_240.mp4' \
        -print | sort
)

[[ ${#VIDEOS[@]} -gt 0 ]] || die "No *_rgb_body_240.mp4 videos found."

echo "BODY videos found: ${#VIDEOS[@]}"

completed=0
failed=0

# ------------------------------------------------------------
# Run sequentially
# ------------------------------------------------------------

for video in "${VIDEOS[@]}"; do
    duration="$(video_duration "$video")"
    echo
    echo "Video: $(basename "$video") (${duration}s)"

    # Common 55 s scenario: all videos.
    if run_one "$video" "$SCENARIO_COMPACT" "cobertura_compacta"; then
        ((completed+=1))
    else
        ((failed+=1))
    fi

    # Three 150 s focused scenarios: only sufficiently long videos.
    if float_ge "$duration" "$MIN_LONG_VIDEO_S"; then
        if run_one "$video" "$SCENARIO_URBAN" "urbano_semaforos"; then
            ((completed+=1))
        else
            ((failed+=1))
        fi

        if run_one "$video" "$SCENARIO_HIGHWAY" "autovia_noche_lluvia"; then
            ((completed+=1))
        else
            ((failed+=1))
        fi

        if run_one "$video" "$SCENARIO_DAYNIGHT" "dia_a_noche"; then
            ((completed+=1))
        else
            ((failed+=1))
        fi
    else
        echo "Skipping 150 s scenarios: video shorter than ${MIN_LONG_VIDEO_S}s."
    fi
done

# ------------------------------------------------------------
# Package all generated CSVs
# ------------------------------------------------------------

timestamp="$(date +%Y%m%d_%H%M%S)"
archive="$OUTPUT_DIR/can_results_${timestamp}.tar.gz"

mapfile -t CSV_FILES < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.csv' -printf '%f\n' | sort)

if [[ ${#CSV_FILES[@]} -gt 0 ]]; then
    tar -czf "$archive" \
        --exclude='*.tar.gz' \
        -C "$OUTPUT_DIR" \
        "${CSV_FILES[@]}"
else
    echo "WARNING: no CSVs produced; nothing to archive." >&2
    archive="(none)"
fi

echo
echo "============================================================"
echo "Batch finished"
echo "Completed runs : $completed"
echo "Failed runs    : $failed"
echo "Archive        : $archive"
echo "============================================================"


if [[ $failed -gt 0 ]]; then
    exit 2
fi
