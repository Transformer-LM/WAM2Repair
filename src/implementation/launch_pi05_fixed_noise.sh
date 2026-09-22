#!/usr/bin/env bash
# Personal-only fixed-noise π0.5 service. It never selects a busy GPU.
set -euo pipefail
root=__WAM2REPAIR_ROOT__
test "$(id -un)" = __WAM2REPAIR_USER__
port=${1:-10119}
[[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] && ((10#$port <= 65535))
checkpoint="$(realpath "$root/models/openpi/official/pi05_libero")"
impl="$root/workspace/policy-relevant-imagined-state-repair/implementation"
test -f "$checkpoint/TRANSFER_VERIFIED.json" -a -f "$checkpoint/transfer_sha256.json"
if ss -ltn | grep -q ":$port "; then echo "port already listening: $port" >&2; exit 20; fi
mkdir -p "$root/logs/wam2repair" "$root/cache/wam2repair-openpi" "$root/tmp/wam2repair-openpi" "$root/results/wam2repair"
run_id="pi05_fixed_noise_$(date -u +%Y%m%dT%H%M%SZ)_$$"
log="$root/logs/wam2repair/$run_id.log"; state="$root/results/wam2repair/$run_id.server.json"
apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)"
gpu_rows="$(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
printf '%s\n%s\n' "$gpu_rows" "$apps" >> "$log"
chosen=''
for preferred in 2 3 0 1; do
  line="$(printf '%s\n' "$gpu_rows" | awk -F, -v i="$preferred" '$1+0==i {print; exit}')"
  IFS=, read -r index uuid memory util <<< "$line"; uuid="${uuid// /}"; memory="${memory// /}"; util="${util// /}"
  [[ -n "$uuid" && "$memory" -lt 500 && "$util" -le 5 ]] || continue
  if ! printf '%s\n' "$apps" | grep -Fq "$uuid"; then chosen="$uuid"; break; fi
done
[[ -n "$chosen" ]] || { echo 'no eligible GPU' >&2; exit 21; }
(
  export CUDA_VISIBLE_DEVICES="$chosen" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
  export XDG_CACHE_HOME="$root/cache/wam2repair-openpi" HF_HOME="$root/cache/wam2repair-openpi/hf" TMPDIR="$root/tmp/wam2repair-openpi"
  "$root/workspace/openpi/.venv/bin/python" -u "$impl/pi05_fixed_noise_server.py" --port "$port" --checkpoint "$checkpoint" >> "$log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 1800); do
    if ss -ltn | grep -q ":$port "; then
      tmp_state="$state.tmp.$$"
      printf '{"status":"RUNNING_FIXED_NOISE_SERVER","port":%s,"server_pid":%s,"gpu_uuid":"%s","checkpoint":"%s","server_code_sha256":"%s"}\n' "$port" "$server_pid" "$chosen" "$checkpoint" "$(sha256sum "$impl/pi05_fixed_noise_server.py" | awk '{print $1}')" > "$tmp_state"
      mv "$tmp_state" "$state"
      set +e; wait "$server_pid"; rc=$?; set -e
      stop_state="$root/results/wam2repair/$run_id.stop.json"
      tmp_state="$stop_state.tmp.$$"; printf '{"status":"STOPPED_FIXED_NOISE_SERVER","port":%s,"server_pid":%s,"exit_code":%s}\n' "$port" "$server_pid" "$rc" > "$tmp_state"; mv "$tmp_state" "$stop_state"
      exit "$rc"
    fi
    kill -0 "$server_pid" 2>/dev/null || exit 22
    sleep 1
  done
  kill "$server_pid" 2>/dev/null || true; exit 23
) &
echo "STARTING_FIXED_NOISE_SERVER run_id=$run_id gpu_uuid=$chosen state=$state log=$log"
