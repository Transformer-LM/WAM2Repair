#!/usr/bin/env bash
set -euo pipefail
root=__WAM2REPAIR_ROOT__
run_id=region_targets_$(date -u +%Y%m%dT%H%M%SZ)
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
echo RUN_ID="$run_id"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/workspace/third_party/LIBERO" LIBERO_CONFIG_PATH="$root/config/libero"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$root/renderer-runtime/osmesa/prefix/usr/lib/x86_64-linux-gnu:$root/renderer-runtime/osmesa/prefix/lib/x86_64-linux-gnu"
export MESA_SHADER_CACHE_DIR="$root/renderer-runtime/osmesa/cache" XDG_CACHE_HOME="$root/renderer-runtime/osmesa/cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 LP_NUM_THREADS=2
timeout 600 "$root/venvs/libero-eval-py310/bin/python" -u "$root/workspace/policy-relevant-imagined-state-repair/implementation/build_region_targets.py" --root "$root/results/wam2repair" --out "$root/results/wam2repair/$run_id"
