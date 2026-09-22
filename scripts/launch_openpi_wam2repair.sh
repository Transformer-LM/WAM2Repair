#!/usr/bin/env bash
set -euo pipefail
GPU=2
BASE=__WAM2REPAIR_ROOT__
ROOT=$BASE/workspace/openpi
LOG=$BASE/logs/wam2repair/openpi/server.log
mkdir -p "$(dirname "$LOG")"
exec >"$LOG" 2>&1
IFS=, read -r mem util < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
mem="${mem// /}"; util="${util// /}"
test "$mem" -lt 500; test "$util" -le 5
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1
cd "$ROOT"
set +e
.venv/bin/python scripts/serve_policy.py --port 10098 policy:checkpoint --policy.config=pi05_libero --policy.dir=$BASE/models/openpi/openpi-assets/checkpoints/pi05_base_pytorch
rc=$?
echo "OPENPI_EXIT_CODE=$rc" >>"$LOG"
exit "$rc"
