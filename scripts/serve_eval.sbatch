#!/usr/bin/env bash
#SBATCH --job-name=drawio-fair-eval
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
set -euo pipefail

runtime_root=$1
source_root=$2
model_path=$3
model_name=$4
output=$5
sampling=${6:-precise-coding}
assigned_devices=$CUDA_VISIBLE_DEVICES
source "$runtime_root/env-setup-runtime.sh"
export CUDA_VISIBLE_DEVICES="$assigned_devices"
export PYTHONPATH="$source_root" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$source_root"
exec "$runtime_root/env/bin/python" -m sft.serve_checkpoint \
  --model-path "$model_path" --generator-model "$model_name" \
  --out "$output" --stop-file "$output.stop" --port 32645 --sampling-preset "$sampling"
