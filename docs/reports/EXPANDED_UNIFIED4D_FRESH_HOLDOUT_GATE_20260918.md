# Expanded unified-4D fresh-holdout gate — 2026-09-18

## Locked before e46 factual collection

| role | source episodes | permitted use |
|---|---|---|
| train | 34, 45 | optimization only |
| validation | 48 | checkpoint selection only |
| heldout | 46 | one frozen evaluation only |
| excluded from this gate | 36, 37, 39, 49 | no selection or heldout metric |

Episode 46 has a completed prefix-only contact screen, but no D0 factual
candidate, pair, WAM future, geometry bridge, or target has been opened at the
time of this declaration. It is therefore the only heldout source for this
gate. Episode 49 has already had post-generation factual evaluation and is
explicitly excluded.

Inputs are only raw allowlisted WAM RGB, current RGB, action sequence and
proprio. Factual RGB/depth/contact/pose/rotation stay target-only. Fixed heads:
RGB residual repair, log-depth, gripper-object contact, object position and
rotation-6D. Fixed selection metric: validation sum of RGB MAE + temporal MAE
+ 0.1 log-depth MAE + 0.05 contact BCE + 0.1 position MAE + 0.1 rotation-6D
MAE, evaluated at predeclared checkpoints only.

No copy-current-frame/repeated-frame baseline is included. This gate cannot
support a penetration, certified-contact, action-ranking, task-success or VLA
claim. It can only test frozen heldout multi-head prediction on one previously
unopened episode.
