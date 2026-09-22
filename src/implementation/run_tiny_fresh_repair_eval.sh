#!/usr/bin/env bash
# One-shot repair evaluation: locked train/test banks and GPU UUID isolation.
set -euo pipefail
root=__WAM2REPAIR_ROOT__
uuid=GPU-aa863a6c-8482-fd6c-e957-424e559e9df4
run_id=today_tiny_repair_$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$root/logs/wam2repair" "$root/results/wam2repair" "$root/tmp"
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
echo "RUN_ID=$run_id"
IFS=, read -r got_uuid memory utilization < <(nvidia-smi -i "$uuid" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
got_uuid=${got_uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
test "$got_uuid" = "$uuid"; test "$memory" -lt 500; test "$utilization" -le 5
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
if printf '%s' "$apps" | grep -F "$uuid"; then exit 20; fi
out="$root/results/wam2repair/$run_id"; test ! -e "$out"
export CUDA_VISIBLE_DEVICES="$uuid" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/workspace/policy-relevant-imagined-state-repair/implementation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TMPDIR="$root/tmp"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec timeout 1800 "$root/conda/envs/fastwam-py311/bin/python" -u "$root/workspace/policy-relevant-imagined-state-repair/implementation/tiny_fresh_repair_eval.py" \
  --train-bank "$root/results/wam2repair/today_tiny_train_merged_20260917T143033Z_2222834" \
  --train-wam "$root/results/wam2repair/today_wam_train_offline_20260918T000000Z" \
  --test-bank "$root/results/wam2repair/today_bank_e6_today_fresh_collect_20260917T143033Z_2222834" \
  --test-wam "$root/results/wam2repair/today_wam_test_offline_20260918T000000Z" \
  --out "$out" --steps 300 --height 112
