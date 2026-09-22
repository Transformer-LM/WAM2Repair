#!/usr/bin/env bash
# Launch only after offline transfer and SHA256 verification have completed.
set -euo pipefail
root=__WAM2REPAIR_ROOT__
checkpoint=$root/models/openpi/official/pi05_libero
test -f "$checkpoint/TRANSFER_VERIFIED.json"
run_id=pi05_libero_official_$(date -u +%Y%m%dT%H%M%SZ)
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
echo RUN_ID="$run_id"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
echo "$apps"
IFS=, read -r uuid mem util < <(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
test "$mem" -lt 500; test "$util" -le 5
if printf '%s' "$apps" | grep -F "$uuid"; then exit 20; fi
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export HF_HOME="$root/cache/huggingface" XDG_CACHE_HOME="$root/cache"
export OPENPI_DATA_HOME="$root/models/openpi" TMPDIR="$root/tmp"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$root/workspace/openpi"
exec .venv/bin/python scripts/serve_policy.py --port 10099 policy:checkpoint --policy.config=pi05_libero --policy.dir="$checkpoint"
