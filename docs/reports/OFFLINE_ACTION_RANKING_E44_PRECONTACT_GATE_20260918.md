# Offline action-ranking recovery: e44 pre-contact gate

The completed e42 K=4 diagnostic is retained as uninformative because all
four factual contact-edge utilities tied.  It will not be used to select a
new state, threshold, or candidate.

## Fixed state and protocol

- Select episode 44, the next numeric contact-positive episode after e42 in
  the pre-existing screen `pi05_object_contact_prefix4_seeded_screen_ep40_47_20260918T156000Z_2305001`.
- The screen records first object contact at step 36. Replay exactly the first
  32 actions of its SHA-recorded 40-action prefix, placing the candidate state
  four steps before the observed first contact. The truncated prefix SHA is
  recorded before collection.
- Draw exactly four fresh pi0.5 candidates, each eight actions, from the same
  fixed-noise server sequence (seed 123); retain every candidate, including
  non-contact and non-success cases.
- Require the unchanged exact ray/raster factual contract for all candidates.
  Any collection failure invalidates the entire K=4 gate; partial candidates
  are not scored.
- Generate one input-only WAM future for every retained candidate and apply
  only the frozen e34/e36/e37 unified-4D checkpoint. Save predictions before
  factual geometry targets are opened. The reference utility is the future
  gripper--movable contact-edge count. If all counts tie, report the gate as
  uninformative rather than inventing a ranking.

This remains an offline one-state diagnostic, not a VLA rollout, closed-loop
success measure, or physical-repair claim. No repeated-current-frame baseline
is used.
