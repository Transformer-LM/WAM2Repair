#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 2
root=__WAM2REPAIR_ROOT__
bank=$(realpath "$1"); wam=$(realpath "$2")
case "$bank" in "$root"/*|__WAM2REPAIR_ROOT__/*) ;; *) exit 21;; esac
case "$wam" in "$root"/*|__WAM2REPAIR_ROOT__/*) ;; *) exit 21;; esac
test -f "$bank/manifest.json"; test -f "$wam/result.json"
run_id=video_holdout_$(date -u +%Y%m%dT%H%M%SZ)
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
timeout 7200 "$root/conda/envs/fastwam-py311/bin/python" -u "$root/workspace/policy-relevant-imagined-state-repair/implementation/video_repair_holdout.py" \
 --bank "$bank" --wam "$wam" --steps 1000 --out "$root/results/wam2repair/$run_id"
