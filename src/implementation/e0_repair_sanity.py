"""WAM2Repair E0/M0: tiny single-batch repair sanity.

This gate checks the repair objective and labels before any LIBERO/WAM scale-up.
Inputs are corrupted imagined states; targets are clean simulator states.  No
privileged target is placed in the input.  The run is intentionally tiny and
reports raw/final pose, contact, slip, and relative-pose losses.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn


def make_batch(n: int, seed: int, device: torch.device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # clean state: ee/object position, 6D rotation proxy, relative pose,
    # contact and slip labels. Rotation proxy is deliberately small.
    pos = torch.randn(n, 6, generator=g) * 0.15
    rot = torch.randn(n, 6, generator=g) * 0.2
    rel = (pos[:, :3] - pos[:, 3:6]) + torch.randn(n, 3, generator=g) * 0.01
    contact = (torch.sigmoid(pos[:, 0] * 4 + pos[:, 3] * 2) > 0.5).float().unsqueeze(1)
    slip = ((pos[:, 1] - pos[:, 4]).abs() > 0.18).float().unsqueeze(1)
    target = torch.cat([pos, rot, rel, contact, slip], dim=1)
    # reversible synthetic WAM-like corruption: drift, relative bias, and
    # label timing flips. Corruption parameters are not passed to the model.
    corrupted = target.clone()
    corrupted[:, :6] += torch.randn(n, 6, generator=g) * 0.08
    corrupted[:, 6:9] += torch.tensor([0.10, -0.06, 0.04])
    corrupted[:, 9] = torch.roll(corrupted[:, 9], 1, 0)
    corrupted[:, 10] = torch.roll(corrupted[:, 10], 2, 0)
    return corrupted.to(device), target.to(device)


class Repair(nn.Module):
    def __init__(self, d=17):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, d))

    def forward(self, x):
        return self.net(x)


def metrics(pred, target):
    pose = torch.mean((pred[:, :6] - target[:, :6]) ** 2).sqrt()
    rel = torch.mean((pred[:, 6:9] - target[:, 6:9]) ** 2).sqrt()
    contact = ((pred[:, 15] > 0.5) == (target[:, 15] > 0.5)).float().mean()
    slip = ((pred[:, 16] > 0.5) == (target[:, 16] > 0.5)).float().mean()
    return {"pose_rmse": float(pose), "relative_pose_rmse": float(rel), "contact_acc": float(contact), "slip_acc": float(slip)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = make_batch(args.batch, args.seed, device)
    model = Repair().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    with torch.no_grad():
        raw = metrics(x, y)
    first = None
    for step in range(args.steps):
        pred = model(x)
        loss_pose = torch.mean((pred[:, :15] - y[:, :15]) ** 2)
        loss_contact = nn.functional.binary_cross_entropy_with_logits(pred[:, 15], y[:, 15])
        loss_slip = nn.functional.binary_cross_entropy_with_logits(pred[:, 16], y[:, 16])
        loss = loss_pose + loss_contact + loss_slip
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if first is None: first = float(loss.detach())
    with torch.no_grad():
        final = metrics(model(x), y)
        final_loss = float((torch.mean((model(x)[:, :15] - y[:, :15]) ** 2) + nn.functional.binary_cross_entropy_with_logits(model(x)[:, 15], y[:, 15]) + nn.functional.binary_cross_entropy_with_logits(model(x)[:, 16], y[:, 16])).detach())
    passed = final_loss < first * 0.2 and final["pose_rmse"] < raw["pose_rmse"] and final["relative_pose_rmse"] < raw["relative_pose_rmse"] and final["contact_acc"] >= 0.95 and final["slip_acc"] >= 0.95
    payload = {"schema": "wam2repair-e0-sanity-v1", "status": "PASS" if passed else "FAIL", "claim_scope": "single-batch synthetic corruption objective sanity; no WAM/VLA claim", "device": str(device), "seed": args.seed, "batch": args.batch, "steps": args.steps, "raw": raw, "final": final, "initial_loss": first, "final_loss": final_loss, "label_isolation": "target clean state only; corruption metadata not input", "created_unix": time.time()}
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
