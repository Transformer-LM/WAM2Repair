#!/usr/bin/env bash
set -euo pipefail
root=__WAM2REPAIR_ROOT__
run_id=factual_cpu_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$root/logs/wam2repair" "$root/results/wam2repair/factual"
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
echo "RUN_ID=$run_id"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
ps -u __WAM2REPAIR_USER__ -o pid,comm
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/workspace/third_party/LIBERO"
export LIBERO_CONFIG_PATH="$root/config/libero"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$root/renderer-runtime/osmesa/prefix/usr/lib/x86_64-linux-gnu:$root/renderer-runtime/osmesa/prefix/lib/x86_64-linux-gnu"
export MESA_SHADER_CACHE_DIR="$root/renderer-runtime/osmesa/cache"
export XDG_CACHE_HOME="$root/renderer-runtime/osmesa/cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 LP_NUM_THREADS=2
timeout 240 "$root/venvs/libero-eval-py310/bin/python" -u \
 "$root/workspace/policy-relevant-imagined-state-repair/implementation/collect_physical_bank.py" \
 --out "$root/results/wam2repair/factual/$run_id" \
 --action-source scripted-sanity --render-gpu -1 --candidates 4 --horizon 8
