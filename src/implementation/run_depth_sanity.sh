#!/usr/bin/env bash
set -euo pipefail
root=__WAM2REPAIR_ROOT__
extra=()
if test "$#" -gt 0; then
 test "$#" -eq 1
 region=$(realpath "$1")
 case "$region" in "$root/results/wam2repair/"*|__WAM2REPAIR_ROOT__/results/wam2repair/*) ;; *) exit 21;; esac
 extra=(--regions "$region")
fi
impl="$root/workspace/policy-relevant-imagined-state-repair/implementation"
run_id=depth_sanity_$(date -u +%Y%m%dT%H%M%SZ)
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
echo RUN_ID="$run_id"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
echo "$apps"
IFS=, read -r uuid mem util < <(nvidia-smi -i 3 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
test "$mem" -lt 500; test "$util" -le 5
if printf '%s' "$apps" | grep -F "$uuid"; then exit 20; fi
export CUDA_VISIBLE_DEVICES="$uuid" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_HOME="$root/cache/torch" XDG_CACHE_HOME="$root/cache" TMPDIR="$root/tmp"
timeout 1800 "$root/conda/envs/fastwam-py311/bin/python" -u "$impl/depth_repair_sanity.py" \
 --bank "$root/results/wam2repair/bank_official_full_20260917" \
 --wam "$root/results/wam2repair/video_diagnostic_20260917T050120Z" \
 --geometry "$root/results/wam2repair/geometry_train_gate_20260917T094500Z" \
 --out "$root/results/wam2repair/$run_id" --steps 300 "${extra[@]}"
