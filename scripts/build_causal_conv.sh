#!/usr/bin/env bash
set -euo pipefail

base=$1
runtime=$2
source_archive=$3
source "$runtime"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE
export MAX_JOBS=${MAX_JOBS:-2}
export TMPDIR="$base/cache/build" PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$TMPDIR" "$base/wheelhouse"
trap 'printf "%s\n" "$?" > "$base/build-exit-code"' EXIT

"$base/env/bin/python" -m pip wheel --no-deps --no-build-isolation --no-index \
  --no-cache-dir --wheel-dir "$base/wheelhouse" "$source_archive"
