"""Select a geometry-supervised raw-WAM repair checkpoint without any heldout input.

This is deliberately separated from heldout evaluation: the process accepts
only two train episodes and one validation episode, so a later heldout episode
cannot be opened, hashed, or accidentally fed to the model before selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from real_wam_repair_pilot import load_sample, sha
from real_wam_repair_geometry_val_holdout import (
    GeometryRawWAMRepair, combine, episode_key, load_geometry,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-pair", type=Path, action="append", required=True)
    p.add_argument("--train-wam", type=Path, action="append", required=True)
    p.add_argument("--train-geometry", type=Path, action="append", required=True)
    p.add_argument("--validation-pair", type=Path, required=True)
    p.add_argument("--validation-wam", type=Path, required=True)
    p.add_argument("--validation-geometry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--depth-weight", type=float, default=0.10)
    a = p.parse_args()
    if len(a.train_pair) != 2 or len(a.train_wam) != 2 or len(a.train_geometry) != 2:
        p.error("requires exactly two train episodes")
    if a.steps < 50 or a.steps % 50 or a.depth_weight <= 0:
        p.error("steps must be >=50 and divisible by 50; depth weight must be positive")
    keys = [episode_key(x) for x in a.train_pair] + [episode_key(a.validation_pair)]
    if len(set(keys)) != 3:
        raise ValueError("train and validation source episodes must be distinct")
    a.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)

    train = [load_sample(pair, wam) | load_geometry(pair, geo)
             for pair, wam, geo in zip(a.train_pair, a.train_wam, a.train_geometry)]
    validation = load_sample(a.validation_pair, a.validation_wam) | load_geometry(a.validation_pair, a.validation_geometry)
    device = torch.device("cuda")
    tr = {k: v.to(device) for k, v in combine(train).items()}
    va = {k: v.to(device) for k, v in combine([validation]).items()}
    model = GeometryRawWAMRepair().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def loss_terms(pred_rgb, pred_log_depth, batch):
        rgb = (pred_rgb[:, :, 1:] - batch["target"][:, :, 1:]).abs().mean()
        temporal = ((pred_rgb[:, :, 1:] - pred_rgb[:, :, :-1]) -
                    (batch["target"][:, :, 1:] - batch["target"][:, :, :-1])).abs().mean()
        valid = batch["depth_valid"][:, 1:]
        depth = (pred_log_depth[:, 1:][valid] - torch.log(batch["depth"][:, 1:].clamp_min(1e-4))[valid]).abs().mean()
        return rgb, temporal, depth, rgb + 0.10 * temporal + a.depth_weight * depth

    best_value, best_step, best_state, history = float("inf"), None, None, []
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        repaired, log_depth = model(tr["raw"], tr["current"], tr["actions"], tr["proprio"])
        _, _, _, loss = loss_terms(repaired, log_depth, tr)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            with torch.no_grad():
                repaired, log_depth = model(va["raw"], va["current"], va["actions"], va["proprio"])
                vrgb, vtemporal, vdepth, value = loss_terms(repaired, log_depth, va)
            if not torch.isfinite(value):
                raise RuntimeError("nonfinite validation loss")
            record = {"step": step + 1, "train_loss": float(loss.detach()), "validation_total": float(value),
                      "validation_rgb_mae": float(vrgb), "validation_temporal_mae": float(vtemporal),
                      "validation_depth_log_mae": float(vdepth)}
            history.append(record)
            print(json.dumps(record), flush=True)
            if float(value) < best_value:
                best_value, best_step = float(value), step + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no validation checkpoint")
    checkpoint = a.out / "validation_selected_geometry_repair.pt"
    torch.save({"model": best_state, "best_step": best_step, "best_validation_total": best_value}, checkpoint)
    report = {"schema": "wam2repair-geometry-validation-selection-v1",
              "status": "SELECTED_CHECKPOINT_WITHOUT_HELDOUT_ACCESS",
              "scope": "two-train/one-validation geometry-supervised raw-WAM repair selection; no heldout target/path was accepted",
              "no_copy_current_frame_baseline": True,
              "seed": a.seed, "steps": a.steps, "depth_weight": a.depth_weight,
              "best_step": best_step, "best_validation_total": best_value, "history": history,
              "provenance": {"train": [{"episode": episode_key(q), "pair_manifest_sha256": sha(q), "wam_result_sha256": sha(w), "geometry_result_sha256": sha(g)} for q, w, g in zip(a.train_pair, a.train_wam, a.train_geometry)],
                             "validation": {"episode": episode_key(a.validation_pair), "pair_manifest_sha256": sha(a.validation_pair), "wam_result_sha256": sha(a.validation_wam), "geometry_result_sha256": sha(a.validation_geometry)}},
              "checkpoint": checkpoint.name, "checkpoint_sha256": sha(checkpoint),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "best_step": best_step, "out": str(a.out)}), flush=True)


if __name__ == "__main__":
    main()
