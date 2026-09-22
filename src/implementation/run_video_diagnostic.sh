#!/usr/bin/env bash
set -euo pipefail
root=__WAM2REPAIR_ROOT__
run_id=video_diagnostic_$(date -u +%Y%m%dT%H%M%SZ)
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
echo RUN_ID="$run_id"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
echo "$apps"
IFS=, read -r uuid mem util < <(nvidia-smi -i 3 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
test "$mem" -lt 500; test "$util" -le 5
if printf '%s' "$apps" | grep -F "$uuid"; then exit 20; fi
export CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/workspace/FastWAM/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="$root/models/fastwam"
export HF_HOME="$root/cache/huggingface" XDG_CACHE_HOME="$root/cache" TORCH_HOME="$root/cache/torch"
export TMPDIR="$root/tmp" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
limit=${WAM_TIMEOUT_SECONDS:-900}
echo TIMEOUT_SECONDS="$limit"
timeout "$limit" "$root/conda/envs/fastwam-py311/bin/python" -u "$root/workspace/policy-relevant-imagined-state-repair/implementation/wam_video_diagnostic.py" --out "$root/results/wam2repair/$run_id" --steps 20 --action-controls "$@"
