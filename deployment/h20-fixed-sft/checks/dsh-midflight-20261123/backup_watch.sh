#!/usr/bin/env bash
# Mirror the pipeline's key artifacts to the HF backup repo as they appear.
set -uo pipefail
B=/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121
R=$B/runs/causal-cot-20261122
O=$B/checks/dsh-midflight-20261123
A=$B/checks/dsh-acceptance-20261123
W=/DATA/disk1/zbw_tmp4/drawio-backup-20261123
PY=$B/env/bin/python
BK=$(dirname "$0")/backup_hf.py
source /DATA/disk1/zbw_tmp4/drawio-agent-qwen38/runtime-env.sh
export PYTHONPATH=$B/releases/157a493d5c556c81faca941b5bf610e8ca282ea0
mkdir -p "$W/uploaded"
export HF_ENDPOINT=https://hf-mirror.com
say() { echo "[$(date '+%F %T')] $*" >> "$W/watch.log"; }
say "backup watcher started"
while true; do
  if [[ -f "$O/curriculum-final.jsonl" && ! -f "$W/uploaded/manifest" ]]; then
    if "$PY" "$BK" "manifest/curriculum-final.jsonl=$O/curriculum-final.jsonl" >> "$W/watch.log" 2>&1; then
      touch "$W/uploaded/manifest"; say "manifest uploaded"
    else say "manifest upload FAILED"; fi
  fi
  if [[ -f "$R/supplement-v2/progress.json" && ! -f "$W/uploaded/supp2" ]]; then
    FIN=$(python3 -c "
import json
d=json.load(open('$R/supplement-v2/progress.json'));p=json.load(open('$R/supplement-v2/plan.json'))
print(1 if d['decisions']>=p['rows'] else 0)" 2>/dev/null || echo 0)
    if [[ "$FIN" == "1" ]]; then
      if tar --ignore-failed-read -czf "$W/supp2.tar.gz" -C "$R" supplement-v2/archive supplement-v2/episodes supplement-v2/plan.json supplement-v2/progress.json 2>/dev/null; then
        if "$PY" "$BK" "rewrites/supplement-v2-20261123.tar.gz=$W/supp2.tar.gz" >> "$W/watch.log" 2>&1; then
          touch "$W/uploaded/supp2"; say "supplement-v2 uploaded ($(stat -c%s "$W/supp2.tar.gz") bytes)"
        else say "supp2 upload FAILED"; fi
      fi
    fi
  fi
  if [[ -f "$R/gate1/verification.json" && ! -f "$W/uploaded/gate1-verified" ]]; then
    if "$PY" "$BK" "gate1/timing-verified.json=$R/gate1/timing-verified.json" "gate1/verification.json=$R/gate1/verification.json" "gate1/train-stamped.log=$R/gate1/train-stamped.log" >> "$W/watch.log" 2>&1; then
      touch "$W/uploaded/gate1-verified"; say "verified gate1 uploaded"
    fi
  fi
  RUN=$R/train-v1
  if [[ -f "$RUN/train.log" && -f "$RUN/train/fixed-loss/baseline.json" ]]; then
    STEP=$("$PY" -c 'from pathlib import Path;import sys;from sft.plot_loss import read_metrics
print(read_metrics(Path(sys.argv[1]))[-1]["step"])' "$RUN/train.log" 2>/dev/null || echo 0)
    BLOCK=$((STEP / 10))
    PREVIOUS=$(cat "$W/uploaded/trainlogs-block" 2>/dev/null || echo 0)
    if (( BLOCK > PREVIOUS )) || [[ -f "$RUN/train/result.json" && ! -f "$W/uploaded/trainlogs-final" ]]; then
      if "$PY" -m sft.plot_loss --log "$RUN/train.log" --output "$RUN/loss" --total-steps 558 >> "$W/watch.log" 2>&1 &&
         "$PY" "$BK" "train-v1/loss=$RUN/loss" "train-v1/fixed-loss=$RUN/train/fixed-loss" \
           "train-v1/train.log=$RUN/train.log" "train-v1/preflight.json=$RUN/train/preflight.json" >> "$W/watch.log" 2>&1; then
        echo "$BLOCK" > "$W/uploaded/trainlogs-block"
        [[ ! -f "$RUN/train/result.json" ]] || touch "$W/uploaded/trainlogs-final"
        say "loss curves and training logs uploaded at step $STEP"
      else say "training log upload FAILED"; fi
    fi
  fi
  if [[ -f "$RUN/train/result.json" && -f "$RUN/train/final/model.safetensors" && ! -f "$W/uploaded/final" ]] &&
     "$PY" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["status"] != "completed")' "$RUN/train/result.json"; then
    if "$PY" "$BK" "models/train-v1-final=$RUN/train/final" >> "$W/watch.log" 2>&1; then
      touch "$W/uploaded/final"; say "completed final weights uploaded"
    else say "final weights upload FAILED"; fi
  fi
  for M in base step115 new; do
    if [[ -f "$A/scoreboard/$M/summary.json" && ! -f "$W/uploaded/acceptance_$M" ]]; then
      if "$PY" "$BK" "acceptance/scoreboard/$M=$A/scoreboard/$M" >> "$W/watch.log" 2>&1; then
        touch "$W/uploaded/acceptance_$M"; say "acceptance $M uploaded"
      fi
    fi
  done
  sleep 300
done
