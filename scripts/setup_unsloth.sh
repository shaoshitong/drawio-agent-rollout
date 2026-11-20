#!/usr/bin/env bash
set -euo pipefail

base=${1:?Usage: setup_unsloth.sh REMOTE_ROOT EXISTING_PYTHON}
bootstrap=${2:?Existing Python with a compatible torch installation is required}
export TMPDIR="$base/cache/tmp" PIP_CACHE_DIR="$base/cache/pip"
export XDG_CACHE_HOME="$base/cache" HF_HOME="$base/cache/huggingface"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$TMPDIR" "$base/sources" "$base/logs"
"$bootstrap" -m venv --system-site-packages "$base/env"
site=$("$base/env/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
inherited=$("$bootstrap" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$inherited" > "$site/serving-torch.pth"
for pair in unsloth-zoo:49980bede3e4720fc254a336859ad73465fc5390; do
  name=${pair%%:*}
  revision=${pair#*:}
  archive="$base/sources/$name-$revision.tar.gz"
  if [[ ! -d "$base/sources/$name-$revision" ]]; then
    curl -fL --retry 3 --connect-timeout 10 --max-time 600 \
      "https://codeload.github.com/unslothai/$name/tar.gz/$revision" -o "$archive"
    tar -xzf "$archive" -C "$base/sources"
  fi
done
cat > "$base/sources/constraints.txt" <<'EOF'
torch==2.9.1
torchvision==0.24.1
triton==3.5.1
transformers==5.17.0
trl==1.15.0
torchao==0.14.1
datasets==4.8.4
EOF
"$base/env/bin/python" -m pip install --no-compile \
  --index-url "${DRAWIO_PYPI_INDEX:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}" \
  --constraint "$base/sources/constraints.txt" \
  "$base/sources/unsloth-zoo-49980bede3e4720fc254a336859ad73465fc5390" \
  'unsloth[huggingfacenotorch]==2026.11.7' \
  transformers==5.17.0 trl==1.15.0 datasets==4.8.4 torchao==0.14.1 \
  qwen-vl-utils setproctitle \
  --report "$base/environment-install.json"
"$base/env/bin/python" -m pip freeze > "$base/environment-lock.txt"
