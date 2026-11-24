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
python_bin=${SFT_PREPROCESS_PYTHON:-"$root/envs/serving/bin/python"}
export PATH="$(dirname "$python_bin"):$PATH"
exec -a python "$python_bin" -m sft.rewrite_cot "$@"
