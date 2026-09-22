#!/usr/bin/env bash
set -euo pipefail
root=__WAM2REPAIR_ROOT__
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
echo "$apps"
IFS=, read -r uuid mem util < <(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
test "$mem" -lt 500; test "$util" -le 5
if printf '%s' "$apps" | grep -F "$uuid"; then exit 20; fi
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME="$root/cache" TMPDIR="$root/tmp"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$root/workspace/openpi/.venv/bin/python" "$root/workspace/policy-relevant-imagined-state-repair/implementation/jax_kernel_witness.py"
