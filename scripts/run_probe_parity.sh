#!/usr/bin/env bash
set -euo pipefail

base=$1
adapter=$2
merged=$3
probes=$4
output=$5
python_bin=${PYTHON_BIN:-python}
mkdir -p "$output"
trap 'printf "%s\n" "$?" > "$output/exit-code"' EXIT

for variant in base peft merged; do
    model=$base
    if [[ $variant == merged ]]; then model=$merged; fi
    "$python_bin" -m sft.probe_parity harvest --base "$base" --model "$model" \
        --adapter "$adapter" --variant "$variant" --probes "$probes" --output "$output"
done
"$python_bin" -m sft.probe_parity score --base "$base" --probes "$probes" --output "$output"
