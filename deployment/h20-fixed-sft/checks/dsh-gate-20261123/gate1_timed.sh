#!/usr/bin/env bash
# Gate 1 with per-update wall-time measurement (claude2 timing rule needs it).
set -euo pipefail
B=/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121
RELEASE=$B/releases/157a493d5c556c81faca941b5bf610e8ca282ea0
G=$B/checks/dsh-gate-20261123
GPUS=${DRAWIO_GPUS:-4,5,6,7}
IFS=, read -ra IDS <<< "$GPUS"
[[ ${#IDS[@]} -eq 4 ]] || { echo "refuse: need 4 GPUs" >&2; exit 2; }
for id in "${IDS[@]}"; do case "$id" in 4|5|6|7) ;; *) echo "refuse: GPU $id outside 4-7" >&2; exit 2 ;; esac; done
MANIFEST=${1:?usage: gate1_timed.sh MANIFEST OUTPUT_DIR}
OUTPUT=${2:?usage: gate1_timed.sh MANIFEST OUTPUT_DIR}
[[ -e "$OUTPUT" ]] && { echo "refuse: output exists: $OUTPUT" >&2; exit 2; }
mkdir -p "$OUTPUT"
export DRAWIO_SFT_ROOT=$B
export DRAWIO_BASE_RUNTIME=/DATA/disk1/zbw_tmp4/drawio-agent-qwen38/runtime-env.sh
export DRAWIO_GPUS=$GPUS DRAWIO_PROCESSES=4
source "$DRAWIO_BASE_RUNTIME"
export PYTHONPATH=$RELEASE
PY=$B/env/bin/python
cd "$RELEASE"
"$PY" -m sft.prepare_full_run --manifest "$MANIFEST" --model "$B/models/Qwen3.5-9B" \
  --output "$OUTPUT/prepare" --base-config "$RELEASE/configs/sft-9b.json" > "$OUTPUT/prepare-full-run.json"
CONFIG="$OUTPUT/prepare/check-config.json"
[[ -f "$CONFIG" ]] || { echo "refuse: config missing" >&2; exit 2; }
set -o pipefail
bash "$RELEASE/scripts/run_full_sft.sh" train --config "$CONFIG" --manifest "$MANIFEST" \
  --output "$OUTPUT/check-run" --max-seq-length 32768 2>&1 \
  | "$PY" -u -c 'import sys,time
for line in sys.stdin:
    sys.stdout.write("%.3f %s" % (time.time(), line)); sys.stdout.flush()' \
  | tee "$OUTPUT/train-stamped.log"
"$PY" "$G/analyze_gate1.py" "$OUTPUT" "$MANIFEST" | tee "$OUTPUT/timing.json"
