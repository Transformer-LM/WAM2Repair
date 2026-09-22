#!/usr/bin/env bash
set -euo pipefail
root=__WAM2REPAIR_ROOT__
run_id=video_overfit_$(date -u +%Y%m%dT%H%M%SZ)
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
echo RUN_ID="$run_id"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
echo "$apps"
IFS=, read -r uuid mem util < <(nvidia-smi -i 3 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
test "$mem" -lt 500; test "$util" -le 5
if printf '%s' "$apps" | grep -F "$uuid"; then exit 20; fi
export CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_HOME="$root/cache/torch" XDG_CACHE_HOME="$root/cache" TMPDIR="$root/tmp"
timeout 600 "$root/conda/envs/fastwam-py311/bin/python" -u "$root/workspace/policy-relevant-imagined-state-repair/implementation/video_repair_overfit.py" \
 --bank "$root/results/wam2repair/aligned_base_pilot_20260917" \
 --wam "$root/results/wam2repair/video_diagnostic_20260917T040341Z" \
 --out "$root/results/wam2repair/$run_id"
