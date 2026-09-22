# Expanded geometry-repair split: predeclaration

Declared before collecting any D0, WAM output, factual target, or repair score
for episodes 34--38. The contact-coverage screen is not a factual target and
does not expose the later eight-action repair window.

| role | episodes | use |
|---|---|---|
| train expansion | 34, 36 | D0 collection, pair build, WAM input-only generation and target-only geometry bridge permitted |
| validation | 37 | same collection/generation permitted; checkpoint selection only |
| held-out test | 38 | collect and generate only after the training recipe and validation-selected checkpoint are frozen |
| reserve | 39 | do not use in this split |

The previously opened episodes 4, 14, 17, 23, 24, 25, 27, 28, and 29 may be
used only as exploratory training material; none may be presented as a fresh
confirmation test. The planned heldout target/output for episode 38 must not
be opened or used for tuning before the checkpoint is frozen. No repeated
current-frame baseline is part of this protocol. RGB/depth geometry scores do
not establish contact, penetration, 6D pose, action ranking, or pi0.5
closed-loop benefit.
