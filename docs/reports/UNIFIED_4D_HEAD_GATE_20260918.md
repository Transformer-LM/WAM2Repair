# Unified 4D-head gate: predeclaration

This gate follows the completed RGB/depth pilot and is declared before any
D0, WAM, geometry target, or evaluation output for episode 39 is created.

## Split and interface

| role | episodes |
|---|---|
| train | 34, 36 |
| validation / checkpoint selection | 37 |
| heldout | 39 |

The model input is only raw WAM RGB future, current RGB, eight candidate
actions, and proprio. Factual RGB, analytic ray depth, contact matrix, object
positions, and object rotations are target-only supervision.

## Fixed heads and losses

The one model must emit: repaired RGB, log-depth, an 8x8 contact-logit matrix,
seven 3D object positions, and seven rotation-6D object orientations at each
of the five registered times. Fixed loss weights are RGB 1.0, temporal RGB
0.1, valid analytic log-depth 0.1, contact BCE 0.05, position L1 0.1, and
rotation-6D L1 0.1. Validation total selects one checkpoint. The heldout
episode is collected/generated only after that checkpoint is persisted.

## Interpretation restriction

Head errors are task-0 supervised prediction diagnostics. They are not proof
of real contact recovery, non-penetration, physical consistency, candidate
action ranking, or pi0.5 closed-loop improvement. No repeated-current-frame
baseline is run.
