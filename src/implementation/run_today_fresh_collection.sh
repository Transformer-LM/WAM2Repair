#!/usr/bin/env bash
# CPU-only factual rollouts for the 2026-09-17 fresh tiny experiment.
set -euo pipefail
root=__WAM2REPAIR_ROOT__
test "$#" -le 1
port=${1:-10117}
[[ "$port" =~ ^[1-9][0-9]{0,4}$ ]]
(( 10#$port <= 65535 ))
checkpoint="$root/models/openpi/official/pi05_libero"
test -f "$checkpoint/TRANSFER_VERIFIED.json"
server=$(ps -u __WAM2REPAIR_USER__ -o pid=,args= | grep -E "[s]erve_policy\\.py --port ${port} policy:checkpoint" || true)
test "$(printf '%s\n' "$server" | sed '/^$/d' | wc -l)" -eq 1
printf '%s\n' "$server" | grep -F -- "--policy.config=pi05_libero"
printf '%s\n' "$server" | grep -F -- "--policy.dir=$checkpoint"
run_id=today_fresh_collect_$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$root/logs/wam2repair" "$root/results/wam2repair" "$root/renderer-runtime/osmesa/tmp"
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
echo "RUN_ID=$run_id"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/workspace/third_party/LIBERO:$root/workspace/openpi/packages/openpi-client/src"
export LIBERO_CONFIG_PATH="$root/config/libero" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$root/renderer-runtime/osmesa/prefix/usr/lib/x86_64-linux-gnu:$root/renderer-runtime/osmesa/prefix/lib/x86_64-linux-gnu"
export MESA_SHADER_CACHE_DIR="$root/renderer-runtime/osmesa/cache" XDG_CACHE_HOME="$root/renderer-runtime/osmesa/cache"
export TMPDIR="$root/renderer-runtime/osmesa/tmp" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for episode in 4 5 6; do
  out="$root/results/wam2repair/today_pi05_t0_e${episode}_${run_id}"
  test ! -e "$out"
  timeout 1800 "$root/venvs/libero-eval-py310/bin/python" -u "$root/workspace/policy-relevant-imagined-state-repair/implementation/pi05_libero_pilot.py" \
    --out "$out" --port "$port" --checkpoint "$checkpoint" \
    --task 0 --episode "$episode" --max-steps 48 --action-mode policy
  "$root/venvs/libero-eval-py310/bin/python" - "$out/result.json" "$out/trajectory.npz" "$episode" "$port" <<'PY'
import json, sys
from pathlib import Path
result, trajectory, episode, port = map(Path, sys.argv[1:])
assert result.is_file() and trajectory.is_file()
row = json.loads(result.read_text())
assert row['status'] == 'EXECUTED'
assert row['suite'] == 'libero_spatial' and row['task'] == 0
assert row['episode'] == int(str(episode)) and row['port'] == int(str(port))
PY
done
echo "COLLECTION_COMPLETE=$run_id"
