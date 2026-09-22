"""Predeclared train/validation/heldout raw-WAM RGB repair pilot without copy frames.

The heldout factual target is deliberately not opened until a validation-selected
checkpoint has been written. This remains an exploratory small-data pilot.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from real_wam_repair_pilot import RawWAMRepair, load_sample, sha


def episode_key(pair_path: Path):
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    source = pair.get("source_d0_episode")
    if not isinstance(source, dict) or not isinstance(source.get("suite"), str) or not isinstance(source.get("task"), int) or not isinstance(source.get("episode"), int):
        raise ValueError("pair lacks exact source episode identity")
    return (source["suite"], source["task"], source["episode"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-pair", type=Path, action="append", required=True)
    p.add_argument("--train-wam", type=Path, action="append", required=True)
    p.add_argument("--validation-pair", type=Path, required=True)
    p.add_argument("--validation-wam", type=Path, required=True)
    p.add_argument("--heldout-pair", type=Path, required=True)
    p.add_argument("--heldout-wam", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()
    if len(a.train_pair) != 2 or len(a.train_wam) != 2 or a.steps < 50 or a.steps % 50:
        p.error("requires exactly two train pairs/train WAMs and steps that are a positive multiple of 50")
    keys = [episode_key(x) for x in a.train_pair] + [episode_key(a.validation_pair), episode_key(a.heldout_pair)]
    if len(set(keys)) != 4:
        raise ValueError("train, validation, and heldout source episodes must all be distinct")
    a.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; torch.set_num_threads(4)
    train = [load_sample(pair, wam) for pair, wam in zip(a.train_pair, a.train_wam)]
    validation = load_sample(a.validation_pair, a.validation_wam)
    device = torch.device("cuda")
    def batch(rows, key): return torch.stack([x[key] for x in rows]).to(device)
    raw, target, current, actions, proprio = [batch(train, key) for key in ("raw", "target", "current", "actions", "proprio")]
    vr, vt, vc, va, vp = [batch([validation], key) for key in ("raw", "target", "current", "actions", "proprio")]
    model = RawWAMRepair().to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    def temporal_loss(pred, gt):
        return (pred[:, :, 1:] - gt[:, :, 1:]).abs().mean() + .1 * ((pred[:, :, 1:] - pred[:, :, :-1]) - (gt[:, :, 1:] - gt[:, :, :-1])).abs().mean()
    with torch.no_grad(): initial = float(temporal_loss(model(raw, current, actions, proprio), target))
    best_val, best_step, best_state, history = float("inf"), None, None, []
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True); loss = temporal_loss(model(raw, current, actions, proprio), target)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite train loss")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % 50 == 0:
            with torch.no_grad(): val = float((model(vr, vc, va, vp)[:, :, 1:] - vt[:, :, 1:]).abs().mean())
            if not np.isfinite(val): raise RuntimeError("nonfinite validation MAE")
            history.append({"step": step + 1, "train_loss": float(loss.detach()), "validation_mae": val})
            if val < best_val:
                best_val, best_step = val, step + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(json.dumps(history[-1]), flush=True)
    if best_state is None: raise RuntimeError("no validation checkpoint")
    model.load_state_dict(best_state); model.eval()
    checkpoint = a.out / "validation_selected_repair.pt"
    torch.save({"model": best_state, "best_step": best_step, "best_validation_mae": best_val}, checkpoint)
    # Heldout target is first opened here, strictly after checkpoint persistence.
    heldout = load_sample(a.heldout_pair, a.heldout_wam)
    hr, ht, hc, ha, hp = [batch([heldout], key) for key in ("raw", "target", "current", "actions", "proprio")]
    with torch.no_grad():
        repaired = model(hr, hc, ha, hp)
        raw_mae = float((hr[:, :, 1:] - ht[:, :, 1:]).abs().mean())
        repair_mae = float((repaired[:, :, 1:] - ht[:, :, 1:]).abs().mean())
        raw_temporal = float(((hr[:, :, 1:] - hr[:, :, :-1]) - (ht[:, :, 1:] - ht[:, :, :-1])).abs().mean())
        repair_temporal = float(((repaired[:, :, 1:] - repaired[:, :, :-1]) - (ht[:, :, 1:] - ht[:, :, :-1])).abs().mean())
    prediction = a.out / "heldout_raw_repair_target.npz"
    np.savez_compressed(prediction, raw=hr[0].permute(1, 2, 3, 0).cpu().numpy(), repair=repaired[0].permute(1, 2, 3, 0).cpu().numpy(), target=ht[0].permute(1, 2, 3, 0).cpu().numpy())
    report = {"schema": "wam2repair-real-raw-wam-val-heldout-pilot-v1", "status": "EVALUATED_VAL_SELECTED_EPISODE_HELDOUT_RGB_PILOT",
              "scope": "two-train/one-validation/one-heldout episode raw-WAM RGB-temporal pilot; no physical or VLA claim",
              "no_copy_current_frame_baseline": True, "heldout_target_opened_after_checkpoint": True,
              "seed": a.seed, "steps": a.steps, "initial_train_loss": initial, "validation_selection": {"best_step": best_step, "best_future_rgb_mae": best_val, "history": history},
              "provenance": {"train": [{"episode": episode_key(p), "pair_manifest_sha256": sha(p), "wam_result_sha256": sha(w)} for p, w in zip(a.train_pair, a.train_wam)],
                             "validation": {"episode": episode_key(a.validation_pair), "pair_manifest_sha256": sha(a.validation_pair), "wam_result_sha256": sha(a.validation_wam)},
                             "heldout": {"episode": episode_key(a.heldout_pair), "pair_manifest_sha256": sha(a.heldout_pair), "wam_result_sha256": sha(a.heldout_wam)}},
              "heldout": {"contact_frames": heldout["contact_frames"], "raw_mae": raw_mae, "repair_mae": repair_mae, "repair_minus_raw_mae": repair_mae - raw_mae, "raw_temporal_mae": raw_temporal, "repair_temporal_mae": repair_temporal},
              "artifacts": {"checkpoint": checkpoint.name, "checkpoint_sha256": sha(checkpoint), "heldout_predictions": prediction.name, "heldout_predictions_sha256": sha(prediction)},
              "limitations": ["Only four source episodes; exploratory because raw heldout trajectories were previously inspected for data gates.", "RGB/temporal metrics do not certify contact, penetration, or 6D pose repair.", "No copy/repeated-frame baseline, action ranking, pi0.5, or closed-loop VLA result."],
              "code_sha256": sha(Path(__file__))}
    (a.out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "heldout": report["heldout"], "best_step": best_step}), flush=True)


if __name__ == "__main__": main()
