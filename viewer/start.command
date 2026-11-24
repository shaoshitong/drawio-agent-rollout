#!/bin/zsh
set -eu
viewer_dir="${0:A:h}"
if ! curl -fsS --max-time 2 http://127.0.0.1:8741/health >/dev/null 2>&1; then
  nohup /usr/bin/python3 -u "$viewer_dir/server.py" --port 8741 > /tmp/dsh-trajectory-lab.log 2>&1 < /dev/null &
  for attempt in {1..25}; do
    if curl -fsS --max-time 1 http://127.0.0.1:8741/health >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
fi
open http://127.0.0.1:8741/
