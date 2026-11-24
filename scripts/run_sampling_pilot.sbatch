#!/usr/bin/env bash
#SBATCH --job-name=drawio-sampling-pilot
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
set -euo pipefail

base=$1
source_root=$2
runtime=$3
plan=$4
outputs=$5
export DRAWIO_SFT_ROOT="$base" DRAWIO_BASE_RUNTIME="$runtime"
export DRAWIO_GPUS=${CUDA_VISIBLE_DEVICES:?Slurm must assign a GPU}
export DRAWIO_PROCESSES=1 DRAWIO_SFT_MODULE=sft.train_unsloth UNSLOTH_COMPILE_DISABLE=1
mkdir -p "$outputs"
trap 'printf "%s\n" "$?" > "$outputs/exit-code"' EXIT

for arm in D0 B3; do
  output="$outputs/$arm"
  mkdir -p "$output"
  manifest=$("$base/env/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest"])' "$plan/$arm.json")
  bash "$source_root/scripts/run_sft.sh" train --config "$plan/$arm.json" \
    --manifest "$manifest" --output "$output" > "$output/run.log" 2>&1 &
  child=$!
  printf '{"arm":"%s","pid":%s,"slurm_job_id":"%s"}\n' \
    "$arm" "$child" "$SLURM_JOB_ID" > "$outputs/active.json"
  status=0
  wait "$child" || status=$?
  printf "%s\n" "$status" > "$output/exit-code"
  if [[ $status -ne 0 ]]; then exit "$status"; fi
done
