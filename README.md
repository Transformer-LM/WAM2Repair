# WAM2Repair

WAM2Repair studies whether an external, action-conditioned repair module can make imagined futures from a frozen World Action Model more geometrically reliable for robot manipulation. The intended corrections are relative 6D pose, depth/temporal consistency, contact state, and robot-object penetration—not merely lower RGB reconstruction error.

## Current evidence boundary

The repository preserves both positive diagnostics and negative gates. Synthetic E0/Stage-1 results establish only an implementation sanity check. The current clean LIBERO held-out diagnostic improves RGB and temporal error over the raw WAM input, but has only two positive contact frames and substantial false positives. It does **not** establish visual penetration repair, valid action ranking, closed-loop pi0.5 improvement, or real-robot performance. See `docs/reports/`.

## Repository policy

This repository contains source, sanitized configurations, plans, protocols, result summaries, and a few small figures. It intentionally excludes model weights, datasets, factual banks, checkpoints, generated rollouts, credentials, host names, user-specific paths, and raw experiment logs.

## Portable deployment

1. Clone this repository on the target machine.
2. Install the separately obtained OpenPI, LIBERO, FastWAM, MuJoCo, and model checkpoints according to their upstream licenses.
3. Copy `.env.example` to a private `.env`, then set all absolute paths.
4. Materialize a machine-local runtime copy:

   ```bash
   python scripts/materialize_runtime.py --root "$WAM2REPAIR_ROOT" --libero-root "$LIBERO_ROOT" --fastwam-root "$FASTWAM_ROOT" --openpi-root "$OPENPI_ROOT" --pi05-checkpoint "$PI05_CHECKPOINT" --action-dit-checkpoint "$FASTWAM_ACTION_DIT_CHECKPOINT"
   ```

5. Run scripts from `$WAM2REPAIR_ROOT/runtime/wam2repair/`. The materializer replaces only the repository placeholder tokens with paths supplied by you; it does not copy weights or data.

Detailed deployment requirements and the experiment order are in `reproducibility/REMOTE_SETUP.md` and `docs/EXPERIMENT_PLAN_20260917_215105.md`.

## Layout

- `src/implementation/`: collection, repair, geometry, and evaluation programs.
- `scripts/`: generic OpenPI/LIBERO integration helpers and the runtime materializer.
- `configs/`: machine-independent configuration templates.
- `docs/`: method, dataset plan, protocols, and curated research state.
- `docs/reports/`: auditable result summaries, including failures and incomplete gates.
- `figures/`: small representative visualizations only.
