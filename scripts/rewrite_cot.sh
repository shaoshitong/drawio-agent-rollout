#!/usr/bin/env bash
set -euo pipefail
root=$1
shift
source "$root/runtime-env.sh"
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
source_root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$source_root"
cd "$source_root"
exec -a python "$root/envs/serving/bin/python" -m sft.rewrite_cot "$@"
