# Counterfactual action-candidate ranking gate

The fixed-noise pi0.5 candidates at e42 and e44 had non-discriminative
short-horizon factual contact utility. This gate therefore changes candidate
*generation*, not thresholds or scoring after seeing WAM results.

## Fixed candidate family

At one independently screen-bound pre-contact state, request one pi0.5 action
chunk and retain its first eight actions `a`.  Construct exactly four action
candidates before simulator rollout:

1. `policy`: `a` unchanged.
2. `attenuated`: translational and rotational action dimensions `a[:,:6] * .5`,
   with the gripper dimension retained.
3. `hold`: action dimensions 0--5 set to zero, with the policy gripper command
   retained.
4. `reverse`: dimensions 0--5 set to `-0.5 * a[:,:6]`, with the policy gripper
   command retained.

The input chunk SHA, all four resulting action-array SHAs, transform names and
the fixed coefficients are written in an immutable candidate manifest before
any candidate is replayed. This is a controlled policy-candidate diagnostic,
not a claim that these are samples from pi0.5's policy distribution.

## Factual utility and decision rules

- Replay every candidate from the same SHA-bound prefix state and require the
  existing exact ray/raster/reset checks.
- Before WAM generation, require that at least two candidates differ in a
  predeclared factual utility tuple: `(task success, future contact-edge count,
  negative bowl-to-plate final distance)`. The object indices/names are bound
  from the D0 manifest, not hand-entered afterward.
- If the tuple is tied for all candidates, retain the collection and stop:
  do not generate WAM or report ranking.
- If it varies, generate one input-only WAM future per candidate and score
  frozen repair outputs before factual targets are opened. The model-side
  utility proxy is future gripper--movable contact score plus predicted
  negative bowl-to-plate final distance, with fixed equal weights after each
  term's within-set min--max normalization. The factual utility is used only
  after the prediction ledger is written.

This is an offline controlled-action diagnostic, not closed-loop pi0.5 task
success and not VLA benefit. No repeated-current-frame baseline is included.
