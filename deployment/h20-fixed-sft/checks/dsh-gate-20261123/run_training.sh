#!/usr/bin/env bash
# Fixed-scheme full-parameter 9B SFT on the final manifest (physical GPU 4-7 only).
set -euo pipefail
B=/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121
RELEASE=$B/releases/157a493d5c556c81faca941b5bf610e8ca282ea0
GPUS=${DRAWIO_GPUS:-4,5,6,7}
IFS=, read -ra IDS <<< "$GPUS"
[[ ${#IDS[@]} -eq 4 ]] || { echo "refuse: training needs exactly 4 GPUs, got $GPUS" >&2; exit 2; }
for id in "${IDS[@]}"; do case "$id" in 4|5|6|7) ;; *) echo "refuse: GPU $id outside 4-7" >&2; exit 2 ;; esac; done
MANIFEST=${1:?usage: run_training.sh MANIFEST RUN_DIR}
RUN=${2:?usage: run_training.sh MANIFEST RUN_DIR}
[[ -e "$RUN" ]] && { echo "refuse: run dir exists: $RUN" >&2; exit 2; }
mkdir -p "$RUN"
export DRAWIO_SFT_ROOT=$B
export DRAWIO_BASE_RUNTIME=/DATA/disk1/zbw_tmp4/drawio-agent-qwen38/runtime-env.sh
export DRAWIO_GPUS=$GPUS DRAWIO_PROCESSES=4
source "$DRAWIO_BASE_RUNTIME"
export PYTHONPATH=$RELEASE
cd "$RELEASE"
"$B/env/bin/python" -m sft.prepare_full_run --manifest "$MANIFEST" \
  --model "$B/models/Qwen3.5-9B" --output "$RUN/prepare" \
  --base-config "$RELEASE/configs/sft-9b.json" | tee "$RUN/prepare-full-run.json"
CONFIG="$RUN/prepare/config.json"
[[ -f "$CONFIG" ]] || { echo "refuse: config missing" >&2; exit 2; }
bash "$RELEASE/scripts/run_full_sft.sh" train --config "$CONFIG" \
  --manifest "$MANIFEST" --output "$RUN/train" --max-seq-length 32768 2>&1 | tee "$RUN/train.log"
"$B/env/bin/python" -m sft.plot_loss --log "$RUN/train.log" --output "$RUN/loss" \
  --title "Qwen3.5-9B fixed-scheme SFT" | tee "$RUN/loss-summary.json"
