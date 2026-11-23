#!/usr/bin/env bash
# Mirror the pipeline's key artifacts to the HF backup repo as they appear.
set -uo pipefail
B=/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121
R=$B/runs/causal-cot-20261122
O=$B/checks/dsh-midflight-20261123
A=$B/checks/dsh-acceptance-20261123
W=/DATA/disk1/zbw_tmp4/drawio-backup-20261123
PY=$B/env/bin/python
BK=$O/backup_hf.py
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
  if [[ -f "$R/gate1/timing.json" && ! -f "$W/uploaded/gate1" ]]; then
    if "$PY" "$BK" "gate1/timing.json=$R/gate1/timing.json" "gate1/train.log=$R/gate1/check-run/train.log" >> "$W/watch.log" 2>&1; then
      touch "$W/uploaded/gate1"; say "gate1 uploaded"
    fi
  fi
  if [[ -d "$R/train-v1/train/loss" && ! -f "$W/uploaded/trainlogs" ]]; then
    if "$PY" "$BK" "train-v1/loss=$R/train-v1/train/loss" "train-v1/fixed-loss=$R/train-v1/train/fixed-loss" \
        "train-v1/train.log=$R/train-v1/train/train.log" "train-v1/preflight.json=$R/train-v1/preflight/preflight.json" >> "$W/watch.log" 2>&1; then
      touch "$W/uploaded/trainlogs"; say "loss curves uploaded"
    fi
  fi
  if [[ -f "$R/train-v1/train/final/model.safetensors" && ! -f "$W/uploaded/final" ]]; then
    if "$PY" "$BK" "models/train-v1-final=$R/train-v1/train/final" >> "$W/watch.log" 2>&1; then
      touch "$W/uploaded/final"; say "final weights uploaded"
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
