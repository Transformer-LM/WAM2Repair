#!/usr/bin/env bash
set -euo pipefail
root=__WAM2REPAIR_ROOT__
run_id=pi05_pilot_$(date -u +%Y%m%dT%H%M%SZ)
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
echo RUN_ID="$run_id"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
echo "$apps"
# Reuse only the already-running personal server; no new GPU process.
ps -p 2143996 -o user=,args= | grep '^__WAM2REPAIR_USER__ .*serve_policy.py.*10098'
gpu2=$(printf '%s\n' "$apps" | grep GPU-c034ca59-aa0c-4629-ce5d-a0b5d92e2747)
test "$(printf '%s\n' "$gpu2" | wc -l)" -eq 1
printf '%s' "$gpu2" | grep ', 2143996,'
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/workspace/third_party/LIBERO:$root/workspace/openpi/packages/openpi-client/src"
export LIBERO_CONFIG_PATH="$root/config/libero"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$root/renderer-runtime/osmesa/prefix/usr/lib/x86_64-linux-gnu:$root/renderer-runtime/osmesa/prefix/lib/x86_64-linux-gnu"
export MESA_SHADER_CACHE_DIR="$root/renderer-runtime/osmesa/cache" XDG_CACHE_HOME="$root/renderer-runtime/osmesa/cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 LP_NUM_THREADS=2
timeout 1200 "$root/venvs/libero-eval-py310/bin/python" -u "$root/workspace/policy-relevant-imagined-state-repair/implementation/pi05_libero_pilot.py" --out "$root/results/wam2repair/$run_id"
