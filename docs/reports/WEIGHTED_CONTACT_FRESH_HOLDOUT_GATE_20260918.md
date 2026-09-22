# Class-balanced contact-head fresh-heldout gate — 2026-09-18

## Locked before e32 D0 collection

| role | source episodes | permitted use |
|---|---|---|
| train | 30, 34, 45 | optimization only |
| validation | 48 | checkpoint selection only |
| heldout | 32 | one frozen evaluation only |

Episode 32 has only a completed prefix contact screen at this point; it has no
D0 factual candidate, pair, WAM future, geometry bridge or target opened.

The only method change is `pos_weight=36.0` in the full contact-matrix BCE.
This number is fixed before fitting from the training factual-label imbalance:
26 positives / 960 total entries, i.e. `(960-26)/26 = 35.923`, rounded to 36.
All other architecture, RGB/depth/pose/rotation losses, loss weights, optimizer,
seed, 400 steps and 50-step checkpoint selection interval remain unchanged.

Primary diagnostic is heldout future gripper--movable contact precision, recall
and F1, reported alongside full-matrix metrics. No metric here certifies visual
contact, non-penetration, action ranking, task success or VLA benefit.
