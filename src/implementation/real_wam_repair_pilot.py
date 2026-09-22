"""Small episode-isolated raw-WAM future repair pilot, without a copy-frame arm.

The model consumes only raw WAM future, current observation, actions and
proprio.  Factual videos are opened only while preparing supervised training
or post-hoc evaluation tensors.  This is an RGB/temporal pilot: simulator
contact/pose metadata is retained for audit but no RGB metric is presented as
physical verification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class RawWAMRepair(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(8 * 7 + 8, 64), nn.SiLU(), nn.Linear(64, 24))
        self.encoder = nn.Conv3d(6, 24, 3, padding=1)
        self.body = nn.Sequential(nn.SiLU(), nn.Conv3d(24, 24, 3, padding=1), nn.SiLU(), nn.Conv3d(24, 3, 3, padding=1))
        nn.init.zeros_(self.body[-1].weight); nn.init.zeros_(self.body[-1].bias)

    def forward(self, raw, current, actions, proprio):
        # Current observation is a condition, not a generated/evaluated baseline.
        current_video = current[:, :, None].expand(-1, -1, raw.shape[2], -1, -1)
        cond = self.condition(torch.cat([actions.flatten(1), proprio], dim=1))[:, :, None, None, None]
        return (raw + self.body(self.encoder(torch.cat([raw, current_video], dim=1)) + cond)).clamp(0, 1)


def as_video(x: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float() / 255.0
    return F.interpolate(x, size=(112, 224), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)


def load_sample(pair_manifest: Path, wam_result: Path) -> dict:
    pair = json.loads(pair_manifest.read_text(encoding="utf-8"))
    wam = json.loads(wam_result.read_text(encoding="utf-8"))
    if pair.get("schema") != "wam2repair-d0-wam-pair-v1" or pair.get("status") != "PACKED_INPUT_TARGET_SEPARATED":
        raise ValueError("invalid pair artifact")
    source_episode = pair.get("source_d0_episode")
    if not isinstance(source_episode, dict) or not isinstance(source_episode.get("suite"), str) or not isinstance(source_episode.get("task"), int) or not isinstance(source_episode.get("episode"), int):
        raise ValueError("pair lacks exact source D0 episode identity")
    if wam.get("schema") != "wam2repair-allowlisted-wam-generation-v1" or wam.get("status") != "GENERATED_INPUT_ONLY_REQUIRES_FACTUAL_EVAL":
        raise ValueError("invalid raw WAM artifact")
    input_path = (pair_manifest.parent / pair["input"]).resolve()
    target_path = (pair_manifest.parent / pair["target"]).resolve()
    if Path(wam.get("input_path", "")).resolve() != input_path or wam.get("input_sha256") != pair.get("input_sha256"):
        raise ValueError("WAM output does not bind to this pair input")
    allowlist = wam.get("input_allowlist")
    if not isinstance(allowlist, list) or len(allowlist) != 1 or Path(allowlist[0]).resolve() != input_path:
        raise ValueError("WAM allowlist is not exactly this pair input")
    video_path = wam_result.parent / Path(wam["video"]).name
    if sha(input_path) != pair["input_sha256"] or sha(target_path) != pair["target_sha256"] or sha(video_path) != wam["video_sha256"]:
        raise ValueError("artifact SHA mismatch")
    with np.load(input_path, allow_pickle=False) as d:
        if set(d.files) != {"current_rgb", "proprio", "actions"}:
            raise ValueError("invalid input fields")
        current, proprio, actions = d["current_rgb"], d["proprio"], d["actions"]
    with np.load(video_path, allow_pickle=False) as d:
        if set(d.files) != {"frames"}: raise ValueError("invalid WAM video fields")
        raw = d["frames"]
    with np.load(target_path, allow_pickle=False) as d:
        required = {"video", "frame_offsets", "times", "contact", "object_pos", "object_rot", "eef_pos", "eef_rot", "success"}
        if set(d.files) != required:
            raise ValueError("invalid privileged target fields")
        target, offsets, times, contact = d["video"], d["frame_offsets"].astype(int), d["times"].astype(np.float64), d["contact"]
        metadata = {k: d[k] for k in ("contact", "object_pos", "object_rot", "eef_pos", "eef_rot", "success")}
    if raw.shape != target.shape or raw.shape != (5, 224, 448, 3) or current.shape != (224, 448, 3) or actions.shape != (8, 7) or proprio.shape != (8,):
        raise ValueError("sample tensor contract mismatch")
    if offsets.shape != (5,) or offsets.tolist() != pair.get("frame_offsets") or not np.all(np.diff(offsets) > 0) or offsets[-1] != pair.get("horizon_actions"):
        raise ValueError("factual target action-offset contract mismatch")
    if times.shape != (5,) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("factual target time contract mismatch")
    if not np.isfinite(target).all() or any(x.ndim < 1 or x.shape[0] != 5 or not np.isfinite(x).all() for x in metadata.values()):
        raise ValueError("factual target metadata contract mismatch")
    return {"id": pair_manifest.parent.name, "raw": as_video(raw), "target": as_video(target),
            "current": as_video(current[None])[:, 0], "actions": torch.from_numpy(actions.copy()).float(),
            "proprio": torch.from_numpy(proprio.copy()).float(), "contact_frames": int(contact[:, :-1, -1].any(axis=1).sum()),
            "d0_manifest_sha256": pair["d0_manifest_sha256"], "d0_candidate_sha256": pair["d0_candidate_sha256"],
            "source_episode": source_episode,
            "pair_manifest_sha256": sha(pair_manifest), "wam_result_sha256": sha(wam_result), "target_frame_offsets": offsets.tolist()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-pair", type=Path, action="append", required=True)
    p.add_argument("--train-wam", type=Path, action="append", required=True)
    p.add_argument("--test-pair", type=Path, required=True)
    p.add_argument("--test-wam", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()
    if len(a.train_pair) != len(a.train_wam) or len(a.train_pair) != 2 or a.steps < 1:
        p.error("this predeclared tiny pilot requires exactly two train pairs, two train WAMs, and positive steps")
    a.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    train = [load_sample(x, y) for x, y in zip(a.train_pair, a.train_wam)]
    test = load_sample(a.test_pair, a.test_wam)
    episode_key = lambda r: (r["source_episode"]["suite"], r["source_episode"]["task"], r["source_episode"]["episode"])
    if len({episode_key(x) for x in train}) != 2:
        raise ValueError("the two training samples do not come from distinct source episodes")
    if episode_key(test) in {episode_key(x) for x in train}:
        raise ValueError("test source episode overlaps training")
    if test["d0_manifest_sha256"] in {x["d0_manifest_sha256"] for x in train}:
        raise ValueError("test shares a source D0 episode manifest with training")
    device = torch.device("cuda")
    def batch(rows, key): return torch.stack([r[key] for r in rows]).to(device)
    raw, target, current, actions, proprio = [batch(train, key) for key in ("raw", "target", "current", "actions", "proprio")]
    model = RawWAMRepair().to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    def objective(pred, gt):
        rgb = (pred[:, :, 1:] - gt[:, :, 1:]).abs().mean()
        temporal = ((pred[:, :, 1:] - pred[:, :, :-1]) - (gt[:, :, 1:] - gt[:, :, :-1])).abs().mean()
        return rgb + .1 * temporal
    with torch.no_grad(): initial = float(objective(model(raw, current, actions, proprio), target))
    history = []
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True); pred = model(raw, current, actions, proprio); loss = objective(pred, target)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % 50 == 0:
            history.append({"step": step + 1, "train_loss": float(loss.detach())}); print(json.dumps(history[-1]), flush=True)
    with torch.no_grad():
        final = float(objective(model(raw, current, actions, proprio), target))
        tr_raw = float((raw[:, :, 1:] - target[:, :, 1:]).abs().mean())
        tr_repair = float((model(raw, current, actions, proprio)[:, :, 1:] - target[:, :, 1:]).abs().mean())
        rows = [test]
        te_raw, te_target, te_current, te_actions, te_proprio = [batch(rows, key) for key in ("raw", "target", "current", "actions", "proprio")]
        te_repair = model(te_raw, te_current, te_actions, te_proprio)
        raw_mae = float((te_raw[:, :, 1:] - te_target[:, :, 1:]).abs().mean())
        repair_mae = float((te_repair[:, :, 1:] - te_target[:, :, 1:]).abs().mean())
        raw_temporal = float(((te_raw[:, :, 1:] - te_raw[:, :, :-1]) - (te_target[:, :, 1:] - te_target[:, :, :-1])).abs().mean())
        repair_temporal = float(((te_repair[:, :, 1:] - te_repair[:, :, :-1]) - (te_target[:, :, 1:] - te_target[:, :, :-1])).abs().mean())
    np.savez_compressed(a.out / "heldout_raw_repair_target.npz", raw=te_raw[0].permute(1, 2, 3, 0).cpu().numpy(), repair=te_repair[0].permute(1, 2, 3, 0).cpu().numpy(), target=te_target[0].permute(1, 2, 3, 0).cpu().numpy())
    torch.save({"model": model.state_dict()}, a.out / "repair.pt")
    report = {"schema": "wam2repair-real-raw-wam-repair-pilot-v1", "status": "EVALUATED_EPISODE_ISOLATED_RGB_PILOT",
              "scope": "two-train-pair/one-heldout-episode raw-WAM repair pilot; RGB/temporal only, no physical or VLA claim",
              "no_copy_current_frame_baseline": True, "seed": a.seed, "steps": a.steps, "initial_train_loss": initial, "final_train_loss": final,
              "train": {"pair_ids": [r["id"] for r in train], "raw_mae": tr_raw, "repair_mae": tr_repair, "contact_frames": [r["contact_frames"] for r in train]},
              "heldout": {"pair_id": test["id"], "contact_frames": test["contact_frames"], "raw_mae": raw_mae, "repair_mae": repair_mae,
                          "raw_temporal_mae": raw_temporal, "repair_temporal_mae": repair_temporal, "repair_minus_raw_mae": repair_mae - raw_mae},
              "history": history, "artifacts": {"heldout_predictions": "heldout_raw_repair_target.npz", "heldout_predictions_sha256": sha(a.out / "heldout_raw_repair_target.npz"),
                                                 "checkpoint": "repair.pt", "checkpoint_sha256": sha(a.out / "repair.pt")},
              "provenance": {"train": [{k: r[k] for k in ("id", "source_episode", "d0_manifest_sha256", "d0_candidate_sha256", "pair_manifest_sha256", "wam_result_sha256", "target_frame_offsets")} for r in train],
                             "heldout": {k: test[k] for k in ("id", "source_episode", "d0_manifest_sha256", "d0_candidate_sha256", "pair_manifest_sha256", "wam_result_sha256", "target_frame_offsets")}},
              "limitations": ["One held-out episode and two training episodes cannot establish generalization.", "RGB/temporal errors do not certify contact, penetration, or 6D pose correction.", "No closed-loop pi0.5 or action-ranking result is included.", "No copy/repeated-frame baseline was run."],
              "code_sha256": sha(Path(__file__))}
    (a.out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "heldout": report["heldout"]}), flush=True)


if __name__ == "__main__": main()
