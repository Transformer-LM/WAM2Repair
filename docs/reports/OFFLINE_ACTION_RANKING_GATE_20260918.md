# Offline action-ranking gate: predeclaration

Episode 40 is selected from completed contact-coverage screening before dense
D0 candidate generation. It was not used to fit or select the unified 4D
repair checkpoint (`unified4d_selection_train34_36_val37_20260918T150000Z_2306600`).

## Protocol

- Replay the audited 40-action screen prefix for episode 40.
- Request exactly four fresh fixed-noise pi0.5 action candidates of eight
  actions at that same state; retain all four candidates, regardless of
  success/contact outcome.
- Generate one input-only WAM future per candidate. Apply only the frozen
  unified 4D checkpoint to each WAM future.
- Score a candidate by the mean predicted sigmoid contact probability over
  future frames and the symmetric matrix edges between gripper index 7 and
  movable-object indices `[0,1,2,4,5]`. No factual tensor enters this score.
- Factual reference ranking is the corresponding future contact-edge count.
  Report top-1 factual score, exact top-1 match (with deterministic lowest
  candidate-index tie break), and Spearman correlation. No model fitting,
  threshold tuning, or candidate rejection after factual scores are known.

This is an offline one-state candidate-ranking diagnostic, not a pi0.5
closed-loop task-success or physical-correction result. No repeated-current
frame baseline is used.

## Execution incident and fixed recovery

The first execution, `d0_action_rank_e40_k4_20260918T157000Z_2307000`, is
an invalid/incomplete collection and will not be scored: candidate 00 and 01
were written with zero reset error, but candidate 02 stopped when the frozen
per-frame exact ray/raster contract rejected one otherwise eligible boundary
pixel (agreement `56901/56902`).  The contract explicitly forbids
residual-based relaxation, so this is retained as a collection failure rather
than silently accepted or patched post hoc.

For one independent recovery attempt, use the next numerically scheduled
contact-positive screen episode, **42** (not selected by any generated WAM,
repair, or factual ranking outcome).  Replay its recorded 40-step prefix and
apply the identical `K=4`, horizon-8, seed-7, fixed-noise-123 protocol above.
All four candidates must pass the unchanged exact geometry contract.  If any
candidate fails, the e42 recovery is also invalid and no action-ranking score
will be reported from partial candidates.  e40 remains retained as an
operational failure in either case.
