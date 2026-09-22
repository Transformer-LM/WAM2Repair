"""Validation-selected raw-WAM repair pilot with target-only analytic-depth supervision.

The model receives raw WAM RGB, the current observation, actions and proprio only.
Factual future RGB and depth are targets: they are never model inputs.  The
heldout pair and its geometry target are deliberately opened only after the
validation-selected checkpoint is persisted.  This is a narrow geometry bridge,
not a physical-correction or VLA-success experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from real_wam_repair_pilot import as_video, load_sample, sha


def episode_key(pair_path: Path):
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    source = pair.get("source_d0_episode")
    if not isinstance(source, dict) or not isinstance(source.get("suite"), str) or not isinstance(source.get("task"), int) or not isinstance(source.get("episode"), int):
        raise ValueError("pair lacks exact source episode identity")
    return (source["suite"], source["task"], source["episode"])


def load_geometry(pair_manifest: Path, geometry_result: Path) -> dict:
    """Open only a pair-bound factual geometry target; fail closed on provenance."""
    report = json.loads(geometry_result.read_text(encoding="utf-8"))
    pair = json.loads(pair_manifest.read_text(encoding="utf-8"))
    if report.get("schema") != "wam2repair-target-only-geometry-bridge-v1" or report.get("status") != "BUILT_CONSERVATIVE_TARGET_ONLY_GEOMETRY_BRIDGE":
        raise ValueError("invalid geometry bridge artifact")
    if report.get("pair_manifest_sha256") != sha(pair_manifest):
        raise ValueError("geometry bridge does not bind this pair manifest")
    bindings = {"d0_manifest_sha256": "d0_manifest_sha256", "d0_candidate_sha256": "d0_candidate_sha256", "pair_target_sha256": "target_sha256"}
    for report_field, pair_field in bindings.items():
        if report.get(report_field) != pair.get(pair_field):
            raise ValueError(f"geometry bridge {report_field} does not bind this pair")
    if report.get("frame_offsets") != pair.get("frame_offsets") or pair.get("frame_offsets") != [0, 2, 4, 6, 8] or pair.get("horizon_actions") != 8:
        raise ValueError("geometry bridge/pair action-time contract mismatch")
    if report.get("target_only") is not True or report.get("invalid_depth") != "NaN":
        raise ValueError("geometry bridge target-only contract mismatch")
    geometry_path = (geometry_result.parent / Path(report["geometry_target"]).name).resolve()
    if sha(geometry_path) != report.get("geometry_target_sha256"):
        raise ValueError("geometry target SHA mismatch")
    with np.load(geometry_path, allow_pickle=False) as d:
        required = {"depth_m", "depth_valid_mask", "ray_geom_id", "contact_matrix", "object_pos", "object_rot", "frame_offsets"}
        v8_optional = {"contact_distance_observed", "contact_signed_distance_m"}
        fields = set(d.files)
        if not required.issubset(fields) or not fields.issubset(required | v8_optional):
            raise ValueError("geometry target field contract mismatch")
        depth, valid, offsets = d["depth_m"].copy(), d["depth_valid_mask"].copy(), d["frame_offsets"].astype(int)
        if depth.shape != (5, 224, 448) or valid.shape != depth.shape or offsets.tolist() != pair["frame_offsets"]:
            raise ValueError("geometry target tensor/timing contract mismatch")
        if not np.isfinite(depth[valid]).all() or np.isfinite(depth[~valid]).any() or not valid.any() or not valid[1:].any(axis=(1, 2)).all():
            raise ValueError("geometry depth/mask contract mismatch")
    # Area-style downsampling, retaining only bins whose valid depth support is substantial.
    finite_depth = np.where(valid, depth, 0.0).astype(np.float32)
    depth_t = torch.from_numpy(finite_depth)[:, None]
    valid_t = torch.from_numpy(valid.astype(np.float32))[:, None]
    numer = F.interpolate(depth_t, size=(112, 224), mode="area")
    denom = F.interpolate(valid_t, size=(112, 224), mode="area")
    small_valid = denom >= 0.75
    small_depth = (numer / denom.clamp_min(1e-6)).squeeze(1).numpy()
    small_depth[~small_valid.squeeze(1).numpy()] = np.nan
    if not np.isfinite(small_depth[small_valid.squeeze(1).numpy()]).all() or np.isfinite(small_depth[~small_valid.squeeze(1).numpy()]).any() or not small_valid.squeeze(1).numpy()[1:].any(axis=(1, 2)).all():
        raise ValueError("downsampled depth/mask contract mismatch")
    return {"depth": torch.from_numpy(small_depth.copy()).float(), "depth_valid": small_valid.squeeze(1),
            "geometry_result_sha256": sha(geometry_result), "geometry_target_sha256": sha(geometry_path),
            "coverage": [float(x) for x in report["coverage_per_frame"]]}


class GeometryRawWAMRepair(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(8 * 7 + 8, 64), nn.SiLU(), nn.Linear(64, 24))
        self.encoder = nn.Conv3d(6, 24, 3, padding=1)
        self.trunk = nn.Sequential(nn.SiLU(), nn.Conv3d(24, 24, 3, padding=1), nn.SiLU())
        self.rgb_residual = nn.Conv3d(24, 3, 3, padding=1)
        self.log_depth = nn.Conv3d(24, 1, 3, padding=1)
        nn.init.zeros_(self.rgb_residual.weight); nn.init.zeros_(self.rgb_residual.bias)

    def forward(self, raw, current, actions, proprio):
        current_video = current[:, :, None].expand(-1, -1, raw.shape[2], -1, -1)
        cond = self.condition(torch.cat([actions.flatten(1), proprio], dim=1))[:, :, None, None, None]
        hidden = self.trunk(self.encoder(torch.cat([raw, current_video], dim=1)) + cond)
        return (raw + self.rgb_residual(hidden)).clamp(0, 1), self.log_depth(hidden).squeeze(1)


def combine(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in ("raw", "target", "current", "actions", "proprio", "depth", "depth_valid")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-pair", type=Path, action="append", required=True)
    p.add_argument("--train-wam", type=Path, action="append", required=True)
    p.add_argument("--train-geometry", type=Path, action="append", required=True)
    p.add_argument("--validation-pair", type=Path, required=True)
    p.add_argument("--validation-wam", type=Path, required=True)
    p.add_argument("--validation-geometry", type=Path, required=True)
    p.add_argument("--heldout-pair", type=Path, required=True)
    p.add_argument("--heldout-wam", type=Path, required=True)
    p.add_argument("--heldout-geometry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--depth-weight", type=float, default=0.10)
    a = p.parse_args()
    if len(a.train_pair) != 2 or len(a.train_wam) != 2 or len(a.train_geometry) != 2 or a.steps < 50 or a.steps % 50 or a.depth_weight <= 0:
        p.error("requires exactly two train items, >=50 steps divisible by 50, and positive depth weight")
    keys = [episode_key(x) for x in a.train_pair] + [episode_key(a.validation_pair), episode_key(a.heldout_pair)]
    if len(set(keys)) != 4:
        raise ValueError("train, validation, and heldout source episodes must all be distinct")
    a.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; torch.set_num_threads(4)

    # Only training and validation factual targets are opened before selection.
    train = [load_sample(pair, wam) | load_geometry(pair, geo) for pair, wam, geo in zip(a.train_pair, a.train_wam, a.train_geometry)]
    validation = load_sample(a.validation_pair, a.validation_wam) | load_geometry(a.validation_pair, a.validation_geometry)
    device = torch.device("cuda")
    tr = {k: v.to(device) for k, v in combine(train).items()}
    va = {k: v.to(device) for k, v in combine([validation]).items()}
    model = GeometryRawWAMRepair().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def losses(pred_rgb, pred_log_depth, batch):
        rgb = (pred_rgb[:, :, 1:] - batch["target"][:, :, 1:]).abs().mean()
        temporal = ((pred_rgb[:, :, 1:] - pred_rgb[:, :, :-1]) - (batch["target"][:, :, 1:] - batch["target"][:, :, :-1])).abs().mean()
        valid = batch["depth_valid"][:, 1:]
        target_log_depth = torch.log(batch["depth"][:, 1:].clamp_min(1e-4))
        depth = (pred_log_depth[:, 1:][valid] - target_log_depth[valid]).abs().mean()
        return rgb, temporal, depth, rgb + 0.10 * temporal + a.depth_weight * depth

    best_val, best_step, best_state, history = float("inf"), None, None, []
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        pred_rgb, pred_depth = model(tr["raw"], tr["current"], tr["actions"], tr["proprio"])
        rgb, temporal, depth, loss = losses(pred_rgb, pred_depth, tr)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite training loss")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % 50 == 0:
            with torch.no_grad():
                vrgb, vdepth = model(va["raw"], va["current"], va["actions"], va["proprio"])
                vr, vt, vd, vl = losses(vrgb, vdepth, va)
                value = float(vl)
            if not np.isfinite(value): raise RuntimeError("nonfinite validation loss")
            history.append({"step": step + 1, "train_loss": float(loss.detach()), "validation_total": value, "validation_rgb_mae": float(vr), "validation_depth_log_mae": float(vd)})
            if value < best_val:
                best_val, best_step = value, step + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(json.dumps(history[-1]), flush=True)
    if best_state is None: raise RuntimeError("no validation checkpoint")
    model.load_state_dict(best_state); model.eval()
    checkpoint = a.out / "validation_selected_geometry_repair.pt"
    torch.save({"model": best_state, "best_step": best_step, "best_validation_total": best_val}, checkpoint)

    # Heldout factual RGB/depth is first opened after checkpoint persistence.
    heldout = load_sample(a.heldout_pair, a.heldout_wam) | load_geometry(a.heldout_pair, a.heldout_geometry)
    ho = {k: v.to(device) for k, v in combine([heldout]).items()}
    with torch.no_grad():
        repaired, predicted_log_depth = model(ho["raw"], ho["current"], ho["actions"], ho["proprio"])
        raw_rgb = float((ho["raw"][:, :, 1:] - ho["target"][:, :, 1:]).abs().mean())
        repair_rgb = float((repaired[:, :, 1:] - ho["target"][:, :, 1:]).abs().mean())
        raw_temporal = float(((ho["raw"][:, :, 1:] - ho["raw"][:, :, :-1]) - (ho["target"][:, :, 1:] - ho["target"][:, :, :-1])).abs().mean())
        repair_temporal = float(((repaired[:, :, 1:] - repaired[:, :, :-1]) - (ho["target"][:, :, 1:] - ho["target"][:, :, :-1])).abs().mean())
        valid = ho["depth_valid"][:, 1:]
        depth_log_mae = float((predicted_log_depth[:, 1:][valid] - torch.log(ho["depth"][:, 1:].clamp_min(1e-4))[valid]).abs().mean())
        depth_abs_rel = float(((predicted_log_depth[:, 1:].exp()[valid] - ho["depth"][:, 1:][valid]).abs() / ho["depth"][:, 1:][valid]).mean())
        predicted_depth = predicted_log_depth.exp()
        numeric = (repaired, predicted_log_depth, predicted_depth, torch.tensor([raw_rgb, repair_rgb, raw_temporal, repair_temporal, depth_log_mae, depth_abs_rel], device=device))
        if not all(torch.isfinite(x).all() for x in numeric):
            raise RuntimeError("nonfinite heldout RGB/depth prediction or metric")
    prediction = a.out / "heldout_raw_repair_predicted_depth_target.npz"
    np.savez_compressed(prediction, raw=ho["raw"][0].permute(1, 2, 3, 0).cpu().numpy(), repair=repaired[0].permute(1, 2, 3, 0).cpu().numpy(), target=ho["target"][0].permute(1, 2, 3, 0).cpu().numpy(), predicted_depth_m=predicted_depth[0].cpu().numpy(), target_depth_m=ho["depth"][0].cpu().numpy(), target_depth_valid_mask=ho["depth_valid"][0].cpu().numpy())
    report = {"schema": "wam2repair-real-raw-wam-geometry-val-heldout-pilot-v1", "status": "EVALUATED_VAL_SELECTED_EPISODE_HELDOUT_GEOMETRY_PILOT",
              "scope": "two-train/one-validation/one-heldout raw-WAM RGB+analytic-depth-supervised pilot; no physical or VLA claim", "no_copy_current_frame_baseline": True, "heldout_target_opened_after_checkpoint": True,
              "seed": a.seed, "steps": a.steps, "depth_weight": a.depth_weight, "validation_selection": {"best_step": best_step, "best_total": best_val, "history": history},
              "provenance": {"train": [{"episode": episode_key(p), "pair_manifest_sha256": sha(p), "wam_result_sha256": sha(w), "geometry_result_sha256": sha(g)} for p, w, g in zip(a.train_pair, a.train_wam, a.train_geometry)], "validation": {"episode": episode_key(a.validation_pair), "pair_manifest_sha256": sha(a.validation_pair), "wam_result_sha256": sha(a.validation_wam), "geometry_result_sha256": sha(a.validation_geometry)}, "heldout": {"episode": episode_key(a.heldout_pair), "pair_manifest_sha256": sha(a.heldout_pair), "wam_result_sha256": sha(a.heldout_wam), "geometry_result_sha256": sha(a.heldout_geometry)}},
              "heldout": {"raw_rgb_mae": raw_rgb, "repair_rgb_mae": repair_rgb, "repair_minus_raw_rgb_mae": repair_rgb - raw_rgb, "raw_temporal_mae": raw_temporal, "repair_temporal_mae": repair_temporal, "predicted_depth_log_mae": depth_log_mae, "predicted_depth_abs_rel": depth_abs_rel, "depth_valid_fraction_future": float(ho["depth_valid"][:, 1:].float().mean()), "contact_frames": heldout["contact_frames"]},
              "artifacts": {"checkpoint": checkpoint.name, "checkpoint_sha256": sha(checkpoint), "heldout_predictions": prediction.name, "heldout_predictions_sha256": sha(prediction)},
              "limitations": ["Only four source episodes; this is exploratory and not a generalization result.", "Predicted depth is supervised by conservative factual analytic depth, but depth error does not certify penetration, contact, or 6D-pose repair.", "No raw-WAM depth baseline exists because WAM produces RGB only; do not interpret depth error as an improvement over raw WAM.", "No pi0.5 action-ranking or closed-loop VLA result is included."], "code_sha256": sha(Path(__file__))}
    (a.out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "heldout": report["heldout"], "best_step": best_step}), flush=True)


if __name__ == "__main__":
    main()
