#!/usr/bin/env bash
set -euo pipefail
root=__WAM2REPAIR_ROOT__
run_id=geometry_ray_$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$root/logs/wam2repair" "$root/results/wam2repair"
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
echo RUN_ID="$run_id"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
timeout 600 "$root/venvs/libero-eval-py310/bin/python" -u "$root/workspace/policy-relevant-imagined-state-repair/implementation/geometry_ray_gate.py" --source "$root/results/wam2repair/atomic_snapshot_20260917T130232Z_2218503" --out "$root/results/wam2repair/$run_id"
