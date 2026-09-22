#!/usr/bin/env bash
set -euo pipefail
test "$#" -ge 1
mode="$1"
height=${REPAIR_HEIGHT:-224}
case "$height" in 112|224) ;; *) exit 23;; esac
root=__WAM2REPAIR_ROOT__
impl="$root/workspace/policy-relevant-imagined-state-repair/implementation"
python="$root/conda/envs/fastwam-py311/bin/python"
case "$mode" in
 sanity) steps=100; limit=900; extra=(--sanity-only);;
 full)
  test "$#" -eq 2
  gate=$(realpath "$2")
  case "$gate" in "$root/results/wam2repair/"*|__WAM2REPAIR_ROOT__/results/wam2repair/*) ;; *) exit 21;; esac
  "$python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="PASS" and r["resolution"]==[224,448]' "$gate/result.json"
  steps=1000; limit=7200; extra=();;
 *) exit 22;;
esac
run_id=video_highres_${mode}_${height}_$(date -u +%Y%m%dT%H%M%SZ)
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
timeout "$limit" "$python" -u "$impl/video_repair_holdout.py" \
 --bank "$root/results/wam2repair/bank_official_full_20260917" \
 --wam "$root/results/wam2repair/video_diagnostic_20260917T050120Z" \
 --height "$height" --steps "$steps" "${extra[@]}" --out "$root/results/wam2repair/$run_id"
