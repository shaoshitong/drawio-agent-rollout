#!/usr/bin/env bash
# Preserve stdin while waiting for an existing Linux process, then execute a command.
set -euo pipefail
pid=${1:?Usage: after_process.sh PID COMMAND...}
shift
identity=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)
while [[ -r /proc/$pid/stat ]]; do
    read -r state current < <(awk '{print $3, $22}' "/proc/$pid/stat")
    [[ $current == "$identity" && $state != Z ]] || break
    sleep 30
done
exec "$@"
