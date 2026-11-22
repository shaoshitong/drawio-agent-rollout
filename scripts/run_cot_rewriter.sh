#!/usr/bin/env bash
set -euo pipefail
root=$1
shift
source "$root/runtime-env.sh"
source_root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$source_root/rollout/h20_bootstrap:$source_root/rollout:$source_root"
export DSH_MASK_PROCESS_TITLE=1
cd "$source_root"
exec -a python "$root/envs/serving/bin/python" -m sft.serve_rewriter --root "$root" "$@"
