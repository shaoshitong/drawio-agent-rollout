#!/usr/bin/env bash
set -euo pipefail

base=$1
adapter=$2
merged=$3
probes=$4
sample=$5
output=$6
python_bin=${PYTHON_BIN:-python}
mkdir -p "$output"
trap 'printf "%s\n" "$?" > "$output/exit-code"' EXIT

for variant in base peft merged; do
    model=$base
    if [[ $variant == merged ]]; then model=$merged; fi
    "$python_bin" -m sft.probe_action --base "$base" --model "$model" --adapter "$adapter" \
        --variant "$variant" --probes "$probes" --sample-id "$sample" \
        --output "$output/$variant" --max-new-tokens "${DRAWIO_PROBE_MAX_NEW_TOKENS:-4096}"
done
