#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1
exec bash __WAM2REPAIR_ROOT__/workspace/policy-relevant-imagined-state-repair/implementation/run_depth_sanity.sh "$1"
