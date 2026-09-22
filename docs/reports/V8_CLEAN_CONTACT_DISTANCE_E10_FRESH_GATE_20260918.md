# v8 clean contact-distance e10 fresh-heldout diagnostic — 2026-09-18

## Locked split

| role | LIBERO spatial task 0 episode | permitted use |
|---|---:|---|
| training | 30, 34, 36 | optimization only |
| validation | 11 | checkpoint selection only |
| fresh heldout | 10 | one frozen evaluation only |

All training episodes will be newly recollected under v8 with the explicit
`--declared-split train` provenance flag. Their old artifacts (including the
previously named e30 v8 result) are `sanity_only` and are not reused. The
fresh e10 and validation e11 have
only the no-target prefix screen
`pi05_object_contact_prefix40_cleanreservoir_e10_13_20260918T130000Z_2395400`:
e10 observed gripper--object contact at prefix steps 36 and 38, while e11
observed none. No e10 D0 candidate, WAM pair/future, or factual geometry
target exists at lock time; these must only be created after a checkpoint is
selected on e11.

This is deliberately a constrained diagnostic. Since e11 has no observed
contact pair, it cannot validate or distance-select the signed-distance head.
The heldout e10 contact/distance metrics can test only whether frozen outputs
are measurable on one unseen contact window. They cannot support generalization,
penetration/SDF, visual physical repair, WAM improvement, action ranking,
task success, or VLA benefit claims. No repeated-current-frame baseline is
included.
