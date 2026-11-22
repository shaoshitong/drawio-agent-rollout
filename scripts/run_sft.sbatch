#!/usr/bin/env bash
#SBATCH --job-name=drawio-sft
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail

base=$1
source_root=$2
runtime=$3
output=$4
shift 4

export DRAWIO_SFT_ROOT="$base" DRAWIO_BASE_RUNTIME="$runtime"
export DRAWIO_GPUS=${CUDA_VISIBLE_DEVICES:?Slurm must assign a GPU}
export DRAWIO_PROCESSES=1
mkdir -p "$output"
trap 'printf "%s\n" "$?" > "$output/exit-code"' EXIT

bash "$source_root/scripts/run_sft.sh" "$@" --output "$output"
