"""Evaluate a frozen geometry-repair checkpoint on one heldout episode."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from real_wam_repair_pilot import load_sample, sha
from real_wam_repair_geometry_val_holdout import GeometryRawWAMRepair, combine, episode_key, load_geometry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selection-result", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--heldout-pair", type=Path, required=True)
    p.add_argument("--heldout-wam", type=Path, required=True)
    p.add_argument("--heldout-geometry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    selection = json.loads(a.selection_result.read_text(encoding="utf-8"))
    if selection.get("status") != "SELECTED_CHECKPOINT_WITHOUT_HELDOUT_ACCESS":
        raise ValueError("selection artifact is not heldout-free")
    if a.checkpoint.name != selection.get("checkpoint") or sha(a.checkpoint) != selection.get("checkpoint_sha256"):
        raise ValueError("checkpoint does not bind selection result")
    a.out.mkdir(parents=True, exist_ok=False)
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("best_step") != selection.get("best_step"):
        raise ValueError("checkpoint step mismatch")
    heldout = load_sample(a.heldout_pair, a.heldout_wam) | load_geometry(a.heldout_pair, a.heldout_geometry)
    device = torch.device("cuda")
    batch = {k: v.to(device) for k, v in combine([heldout]).items()}
    model = GeometryRawWAMRepair().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    with torch.no_grad():
        repair, predicted_log_depth = model(batch["raw"], batch["current"], batch["actions"], batch["proprio"])
        valid = batch["depth_valid"][:, 1:]
        raw_rgb = float((batch["raw"][:, :, 1:] - batch["target"][:, :, 1:]).abs().mean())
        repair_rgb = float((repair[:, :, 1:] - batch["target"][:, :, 1:]).abs().mean())
        raw_temporal = float(((batch["raw"][:, :, 1:] - batch["raw"][:, :, :-1]) - (batch["target"][:, :, 1:] - batch["target"][:, :, :-1])).abs().mean())
        repair_temporal = float(((repair[:, :, 1:] - repair[:, :, :-1]) - (batch["target"][:, :, 1:] - batch["target"][:, :, :-1])).abs().mean())
        predicted_depth = predicted_log_depth.exp()
        log_mae = float((predicted_log_depth[:, 1:][valid] - torch.log(batch["depth"][:, 1:].clamp_min(1e-4))[valid]).abs().mean())
        abs_rel = float(((predicted_depth[:, 1:][valid] - batch["depth"][:, 1:][valid]).abs() / batch["depth"][:, 1:][valid]).mean())
        if not all(torch.isfinite(t).all() for t in (repair, predicted_depth)):
            raise RuntimeError("nonfinite prediction")
    prediction = a.out / "heldout_raw_repair_predicted_depth_target.npz"
    np.savez_compressed(prediction, raw=batch["raw"][0].permute(1, 2, 3, 0).cpu().numpy(), repair=repair[0].permute(1, 2, 3, 0).cpu().numpy(), target=batch["target"][0].permute(1, 2, 3, 0).cpu().numpy(), predicted_depth_m=predicted_depth[0].cpu().numpy(), target_depth_m=batch["depth"][0].cpu().numpy(), target_depth_valid_mask=batch["depth_valid"][0].cpu().numpy())
    report = {"schema": "wam2repair-frozen-geometry-heldout-eval-v1", "status": "EVALUATED_FROZEN_CHECKPOINT_HELDOUT", "scope": "one frozen-checkpoint heldout RGB/temporal/depth evaluation; not a physical or VLA result", "no_copy_current_frame_baseline": True,
              "selection_result_sha256": sha(a.selection_result), "checkpoint_sha256": sha(a.checkpoint), "heldout_episode": episode_key(a.heldout_pair), "heldout_pair_sha256": sha(a.heldout_pair), "heldout_wam_sha256": sha(a.heldout_wam), "heldout_geometry_sha256": sha(a.heldout_geometry),
              "metrics": {"raw_rgb_mae": raw_rgb, "repair_rgb_mae": repair_rgb, "repair_minus_raw_rgb_mae": repair_rgb - raw_rgb, "raw_temporal_mae": raw_temporal, "repair_temporal_mae": repair_temporal, "predicted_depth_log_mae": log_mae, "predicted_depth_abs_rel": abs_rel, "depth_valid_fraction_future": float(valid.float().mean())},
              "contact_frames": heldout["contact_frames"], "prediction": prediction.name, "prediction_sha256": sha(prediction), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limitations": ["One heldout episode is exploratory only.", "Depth and RGB metrics do not certify contact, penetration, 6D pose, action ranking, or closed-loop pi0.5 benefit.", "Raw WAM produces no depth, so predicted-depth error has no raw-WAM depth baseline."]}
    (a.out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
