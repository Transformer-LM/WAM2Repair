# Portable server setup

This repository is intentionally environment-agnostic. A target server needs a user-writable work directory, a compatible GPU driver/runtime if GPU experiments are run, and separately installed copies of LIBERO, OpenPI, FastWAM, MuJoCo, and the licensed checkpoints.

## Required private paths

Create a private `.env` from `.env.example`. `WAM2REPAIR_ROOT` must be writable and must not be the cloned Git repository. Keep data, caches, logs, checkpoints, and generated results under that directory.

## Materialize the runtime

From the clone root:

```bash
set -a
source .env
set +a
python scripts/materialize_runtime.py \
  --root "$WAM2REPAIR_ROOT" \
  --user "${WAM2REPAIR_USER:-$(id -un)}" \
  --libero-root "$LIBERO_ROOT" \
  --fastwam-root "$FASTWAM_ROOT" \
  --openpi-root "$OPENPI_ROOT" \
  --pi05-checkpoint "$PI05_CHECKPOINT" \
  --action-dit-checkpoint "$FASTWAM_ACTION_DIT_CHECKPOINT"
```

The generated runtime copy contains target-machine paths. It is deliberately ignored by Git. Re-run materialization after pulling code changes.

## Reproducibility order

1. Validate LIBERO/MuJoCo rendering and atomic snapshot alignment.
2. Collect an episode-separated, contact-rich factual bank.
3. Construct WAM-only inputs and factual geometry targets without target leakage.
4. Select checkpoints using only validation episodes.
5. Evaluate a frozen repairer on held-out episodes.
6. Run candidate-action ranking only after geometry/contact prediction has adequate coverage and calibration.
7. Attempt pi0.5 closed-loop evaluation only after offline ranking is validated.

Do not treat synthetic corruption experiments, RGB-only metrics, or a tiny contact set as evidence of VLA improvement.
