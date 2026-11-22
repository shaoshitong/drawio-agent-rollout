#!/usr/bin/env bash
set -euo pipefail

base=${DRAWIO_SFT_ROOT:?Set DRAWIO_SFT_ROOT to the remote experiment directory}
runtime=${DRAWIO_BASE_RUNTIME:?Set DRAWIO_BASE_RUNTIME to the existing CUDA runtime environment file}
task_module=${DRAWIO_SFT_MODULE:-sft.train_unsloth}
source "$runtime"
export PATH="$base/env/bin:$PATH"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"
export TMPDIR="$base/cache/tmp" XDG_CACHE_HOME="$base/cache"
export HF_HOME="$base/cache/huggingface" HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$base/cache/triton" CUDA_CACHE_PATH="$base/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$base/cache/inductor" TORCH_EXTENSIONS_DIR="$base/cache/extensions"
export NUMBA_CACHE_DIR="$base/cache/numba" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${DRAWIO_GPUS:-${CUDA_VISIBLE_DEVICES:-4}}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
cd "$base"
if [[ ${DRAWIO_PROCESSES:-1} -gt 1 ]]; then
  export UNSLOTH_COMPILE_DISABLE=1
  export UNSLOTH_COMPILE_LOCATION="$base/unsloth_compiled_cache_eager_ddp"
  exec -a python "$base/env/bin/python" -m torch.distributed.run \
    --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --local-addr=127.0.0.1 \
    --nproc_per_node="$DRAWIO_PROCESSES" --module "$task_module" "$@"
fi
exec -a python "$base/env/bin/python" -m "$task_module" "$@"
