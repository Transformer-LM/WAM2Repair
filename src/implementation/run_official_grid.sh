#!/usr/bin/env bash
set -euo pipefail
root=__WAM2REPAIR_ROOT__
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$root/tmp"
exec > >(tee -a "$root/logs/wam2repair/official_grid_20260917.log") 2>&1
exec "$root/venvs/libero-eval-py310/bin/python" -u "$root/workspace/policy-relevant-imagined-state-repair/implementation/collect_official_grid.py"
