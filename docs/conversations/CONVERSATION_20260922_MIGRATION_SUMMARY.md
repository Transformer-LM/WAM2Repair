# 2026-09-22: repository and migration conversation summary

This is a sanitized, human-written summary of the migration discussion. It is not an export of the chat platform's raw transcript.

## Decisions recorded

- The project repository is `Transformer-LM/WAM2Repair`, with active development on `research/wam2repair-4d-geometry`.
- The repository should preserve code, research ideas, method evolution, negative results, protocols, audits, and small result summaries.
- Model weights, factual banks, generated rollouts, checkpoints, credentials, host identities, and raw server logs stay out of Git.
- The portable deployment route is: clone the repository, configure private paths, materialize a private runtime copy, then privately supply dependencies and selected experiment assets.
- Synthetic repair evidence must not be represented as a WAM/VLA main result. Current clean V8 evidence remains an offline, small held-out diagnostic and does not establish penetration repair, valid action ranking, or pi0.5 closed-loop gain.
- Current active inference uses the PyTorch pi0.5 checkpoint (`pi05_base_pytorch/model.safetensors`). FastWAM and the repair networks are also PyTorch-based; MuJoCo/LIBERO supplies simulation rather than a learned PyTorch model.

## Private migration decision

The minimal project-specific clean-V8 bundle was downloaded locally under `WAM2Repair-private-assets/clean_v8_20260922`. It contains the FastWAM fine-tune checkpoint, P027C normalization statistics, selected unified4d repair checkpoint, and the coherent factual/pair/WAM/geometry artifacts. It intentionally excludes public base-model weights.

## Transcript boundary

The three neighboring redacted handoff/recovery files preserve the project-side conversation context available in the workspace. The raw Codex/ChatGPT UI transcript is not stored in this workspace and cannot be reconstructed or exported by this repository without a user-provided chat export.
