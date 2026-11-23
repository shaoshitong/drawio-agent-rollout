#!/usr/bin/env bash
# Run the frozen 20-case acceptance for several checkpoints, then score each model.
#   mode=review (default): two rounds with the DeepSeek reviewer; needs --key-file.
#   mode=single:           one round, no reviewer (declared deviation).
set -euo pipefail
B=/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121
A=$B/checks/dsh-acceptance-20261123
R=$B/runs/causal-cot-20261122
PY=$B/env/bin/python
MODELS=""
KEY=""
MODE=review
ROUND=round2
for arg in "$@"; do
  case "$arg" in
    --models=*) MODELS="${arg#*=}" ;;
    --key-file=*) KEY="${arg#*=}" ;;
    --mode=*) MODE="${arg#*=}" ;;
    --round=*) ROUND="${arg#*=}" ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done
[[ -n "$MODELS" ]] || { echo "usage: acceptance_all.sh --models=name=path[,name=path] [--key-file=F] [--mode=review|single] [--round=round1|round2]" >&2; exit 2; }
if [[ "$MODE" == "review" ]]; then
  [[ -n "$KEY" && -f "$KEY" ]] || { echo "refuse: review mode needs --key-file" >&2; exit 2; }
  ROUND=round2
else
  ROUND=round1
fi
source /DATA/disk1/zbw_tmp4/drawio-agent-qwen38/runtime-env.sh
export PYTHONPATH=$B/releases/28d8a383ebf711beb256cd1a6ede8d0f8be2dc08
export LD_LIBRARY_PATH=/DATA/disk1/zbw_tmp4/drawio-agent-qwen38/renderer/lib:${LD_LIBRARY_PATH:-}

EDITOR_OUT=$R/editor-acceptance-tp2
if curl -sf --max-time 3 http://127.0.0.1:32527/get_server_info >/dev/null 2>&1; then
  echo "editor already up on 32527"
else
  echo "starting TP2 editor on GPUs 4,5"
  "$PY" "$A/start_editor_tp2.py" --output "$EDITOR_OUT" --gpus 4,5 --port 32527 &
  EPID=$!
  trap 'kill "$EPID" 2>/dev/null || true' EXIT
  for _ in $(seq 1 90); do
    curl -sf --max-time 3 http://127.0.0.1:32527/get_server_info >/dev/null 2>&1 && break
    sleep 10
  done
  curl -sf --max-time 3 http://127.0.0.1:32527/get_server_info >/dev/null 2>&1 || { echo "refuse: editor not ready" >&2; exit 1; }
fi

if [[ "$MODE" == "review" ]]; then
  "$PY" "$A/run_acceptance.py" --models "$MODELS" --key-file "$KEY" --gpu 6 --port 32635
else
  IFS=, read -ra PAIRS <<< "$MODELS"
  for pair in "${PAIRS[@]}"; do
    name="${pair%%=*}"; path="${pair#*=}"
    for ref in "$("$PY" -c "import json,sys
for c in json.load(open(sys.argv[1]))['cases']: print(c['episode'], c['reference_png'])" "$A/acceptance-set.json")"; do
      ep="${ref%% *}"; png="${ref#* }"
      [[ -f "$A/runs/$name/$ep/case/round1.drawio" ]] && continue
      "$PY" "$A/run_single_round.py" --out "$A/runs/$name/$ep" --reference "$png" \
        --model-path "$path" --gpu 6 --port 32635 || echo "FAILED $name $ep" >&2
    done
  done
fi

IFS=, read -ra PAIRS <<< "$MODELS"
for pair in "${PAIRS[@]}"; do
  name="${pair%%=*}"
  "$PY" "$A/scoreboard.py" --model "$name" --round "$ROUND"
done
