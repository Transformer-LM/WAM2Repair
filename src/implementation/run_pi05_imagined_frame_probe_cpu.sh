#!/usr/bin/env bash
# CPU/OSMesa client for the one-context post-hoc diagnostic; no GPU is visible.
set -euo pipefail
root=__WAM2REPAIR_ROOT__
test "$(id -un)" = __WAM2REPAIR_USER__
port=${1:-10119}; launch=${2:?server launch json required}
impl="$root/workspace/policy-relevant-imagined-state-repair/implementation"
test -f "$launch"; grep -Fq 'RUNNING_FIXED_NOISE_SERVER' "$launch"
pid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["server_pid"])' "$launch")
test "$(ps -o user= -p "$pid" | tr -d ' ')" = __WAM2REPAIR_USER__
ps -p "$pid" -o args= | grep -Fq 'pi05_fixed_noise_server.py'
mkdir -p "$root/logs/wam2repair" "$root/renderer-runtime/osmesa/tmp" "$root/renderer-runtime/osmesa/cache"
run_id="pi05_imagined_probe_$(date -u +%Y%m%dT%H%M%SZ)_$$"; out="$root/results/wam2repair/$run_id"
exec > >(tee "$root/logs/wam2repair/$run_id.log") 2>&1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/workspace/third_party/LIBERO:$root/workspace/openpi/packages/openpi-client/src"
export LIBERO_CONFIG_PATH="$root/config/libero" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$root/renderer-runtime/osmesa/prefix/usr/lib/x86_64-linux-gnu:$root/renderer-runtime/osmesa/prefix/lib/x86_64-linux-gnu"
export MESA_SHADER_CACHE_DIR="$root/renderer-runtime/osmesa/cache" XDG_CACHE_HOME="$root/renderer-runtime/osmesa/cache" TMPDIR="$root/renderer-runtime/osmesa/tmp"
set +e
timeout 7200 "$root/venvs/libero-eval-py310/bin/python" -u "$impl/pi05_imagined_frame_probe.py" --out "$out" --port "$port" \
 --trajectory "$root/results/wam2repair/today_pi05_t0_e6_today_fresh_collect_20260917T143033Z_2222834" \
 --bank "$root/results/wam2repair/today_bank_e6_today_fresh_collect_20260917T143033Z_2222834" \
 --raw-wam "$root/results/wam2repair/today_wam_test_offline_20260918T000000Z" \
 --wam-result "$root/results/wam2repair/today_wam_test_offline_20260918T000000Z" \
 --repair "$root/results/wam2repair/today_tiny_repair_20260917T165117Z_2228756/runtime_repair_only_verified_20260918.npz" \
 --repair-meta "$root/results/wam2repair/today_tiny_repair_20260917T165117Z_2228756/runtime_repair_only_verified_20260918.json" \
 --repair-result "$root/results/wam2repair/today_tiny_repair_20260917T165117Z_2228756/result.json" \
 --server-launch "$launch"
rc=$?
set -e
if [[ "$rc" -ne 0 && ! -f "$out/result.json" ]]; then
  "$root/venvs/libero-eval-py310/bin/python" - "$out" "$rc" <<'PY'
import json, os, sys
from pathlib import Path
out, rc = Path(sys.argv[1]), int(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tmp = out/'result.json.tmp'
tmp.write_text(json.dumps({'status':'FAILED_IMAGINED_FRAME_PROBE','failure_stage':'wrapper_nonzero_exit','exit_code':rc}, indent=2))
os.replace(tmp, out/'result.json')
PY
fi
exit "$rc"
