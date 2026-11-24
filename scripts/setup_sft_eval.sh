#!/usr/bin/env bash
set -euo pipefail
base=${1:?Usage: setup_sft_eval.sh EXISTING_SERVING_ROOT}
source "$base/env-setup-runtime.sh"
"$base/env/bin/python" -m pip install --no-compile --timeout 20 --retries 1 \
    --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    playwright==1.58.0 Pillow==11.3.0 peft==0.21.0 fla-core==0.4.1 \
    torch==2.9.1 transformers==5.3.0 triton==3.5.1
