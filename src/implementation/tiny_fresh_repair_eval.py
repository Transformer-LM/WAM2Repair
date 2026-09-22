"""Fixed-protocol fresh episode repair pilot; one held-out episode, no model selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from video_repair_overfit import UnifiedVideoRepair, tensor_video


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(bank, wam, required_split, height):
    manifest = json.loads((bank / "manifest.json").read_text())
    generated = json.loads((wam / "result.json").read_text())
    assert manifest["horizon"] == 16 and manifest["frame_offsets"] == [0, 4, 8, 12, 16]
    assert generated["status"] == "GENERATED_ALIGNED"
    assert generated["manifest_sha256"] == sha(bank / "manifest.json")
    rows = manifest["records"]
    assert rows and all(r["split"] == required_split for r in rows)
    assert len({r["id"] for r in rows}) == len(rows), "duplicate bank IDs"
    assert len({r["id"] for r in generated["records"]}) == len(generated["records"]), "duplicate WAM IDs"
    predicted = {r["id"]: r for r in generated["records"]}
    assert set(predicted) == {r["id"] for r in rows}
    convert = lambda x: tensor_video(x, size=(height, height * 2))
    values = {k: [] for k in ("raw", "current", "actions", "proprio", "target")}
    for row in rows:
        pr = predicted[row["id"]]
        files = [(bank / row["input"], row["input_sha256"]),
                 (bank / row["target"], row["target_sha256"]),
                 (wam / pr["prediction"], pr["sha256"])]
        assert all(sha(p) == h for p, h in files)
        with np.load(files[0][0], allow_pickle=False) as d:
            assert set(d.files) == {"current_rgb", "proprio", "actions"}
            assert d["current_rgb"].shape == (224, 448, 3) and d["current_rgb"].dtype == np.uint8
            assert d["actions"].shape == (16, 7) and d["proprio"].shape == (8,)
            assert all(np.isfinite(d[k]).all() for k in ("current_rgb", "proprio", "actions"))
            values["current"].append(convert(d["current_rgb"][None])[:, 0])
            values["actions"].append(torch.from_numpy(d["actions"].copy()).float())
            values["proprio"].append(torch.from_numpy(d["proprio"].copy()).float())
        with np.load(files[1][0], allow_pickle=False) as d:
            assert set(d.files) >= {"video", "times", "success"} and d["video"].shape == (5, 224, 448, 3) and d["video"].dtype == np.uint8
            assert np.isfinite(d["video"]).all()
            values["target"].append(convert(d["video"]))
        with np.load(files[2][0], allow_pickle=False) as d:
            assert set(d.files) == {"frames"}
            assert d["frames"].shape == (5, 224, 448, 3) and d["frames"].dtype == np.uint8 and np.isfinite(d["frames"]).all()
            values["raw"].append(convert(d["frames"]))
    stacked = {k: torch.stack(v).cuda() for k, v in values.items()}
    assert torch.allclose(stacked["current"], stacked["target"][:, :, 0]), "factual t0/current mismatch"
    protocol = {k: generated[k] for k in ("checkpoint_sha256", "steps", "seed", "model_paths")}
    return rows, stacked, sha(bank / "manifest.json"), sha(wam / "result.json"), protocol


def metric(x, target):
    return float((x[:, :, 1:] - target[:, :, 1:]).abs().mean())


def temporal_metric(x, target):
    return float(((x[:, :, 1:] - x[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])).abs().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-bank", type=Path, required=True)
    p.add_argument("--train-wam", type=Path, required=True)
    p.add_argument("--test-bank", type=Path, required=True)
    p.add_argument("--test-wam", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--height", type=int, choices=[112, 224], default=112)
    a = p.parse_args()
    assert a.steps == 300 and a.height == 112, "this one-shot protocol is locked to 300 steps at 112px"
    a.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    train_rows, train, train_manifest_sha, train_wam_sha, train_protocol = load(a.train_bank, a.train_wam, "train", a.height)
    test_rows, test, test_manifest_sha, test_wam_sha, test_protocol = load(a.test_bank, a.test_wam, "test", a.height)
    train_groups, test_groups = {r["group"] for r in train_rows}, {r["group"] for r in test_rows}
    assert train_groups == {"libero_spatial/task0/episode4", "libero_spatial/task0/episode5"} and len(train_rows) == 4
    assert test_groups == {"libero_spatial/task0/episode6"} and len(test_rows) == 1
    assert not train_groups & test_groups and not {r["id"] for r in train_rows} & {r["id"] for r in test_rows}
    assert train_protocol == test_protocol, "train/test WAM generation mismatch"
    torch.cuda.reset_peak_memory_stats()
    model = UnifiedVideoRepair().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Factual future is used only for the training loss; steps were fixed before seeing test.
    history = []
    for step in range(a.steps):
        pred = model(train["raw"], train["current"], train["actions"], train["proprio"])
        loss = (pred[:, :, 1:] - train["target"][:, :, 1:]).abs().mean()
        loss = loss + .1 * ((pred[:, :, 1:] - pred[:, :, :-1]) - (train["target"][:, :, 1:] - train["target"][:, :, :-1])).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step in (0, a.steps - 1): history.append({"step": step + 1, "loss": float(loss.detach())})
    model.eval()
    with torch.no_grad():
        raw, target, current = test["raw"], test["target"], test["current"]
        copy = current[:, :, None].expand_as(raw)
        repair = model(raw, current, test["actions"], test["proprio"])
        anchored = .25 * raw + .75 * copy
        # Diagnostic only: clean future is never available to the deployed repairer.
        identity = model(target, current, test["actions"], test["proprio"])
    predictions = {"raw": raw, "copy_current": copy, "anchored": anchored, "repair": repair,
                   "oracle_clean_identity": identity, "factual": target}
    out_npz = a.out / "heldout_predictions.npz"
    np.savez_compressed(out_npz, **{k: v[0].permute(1, 2, 3, 0).cpu().numpy() for k, v in predictions.items()})
    panels = []
    for t in range(5):
        panels.append(np.concatenate([(predictions[k][0, :, t].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                                      for k in ("raw", "copy_current", "repair", "factual")], axis=1))
    Image.fromarray(np.concatenate(panels, axis=0)).save(a.out / "heldout_raw_copy_repair_factual.png")
    values = {"raw": metric(raw, target), "copy_current": metric(copy, target), "anchored": metric(anchored, target), "repair": metric(repair, target)}
    verdict = "negative_not_better_than_raw" if values["repair"] >= values["raw"] else (
        "positive_vs_raw_but_not_better_than_simple_baseline" if values["repair"] >= min(values["copy_current"], values["anchored"]) else "positive_vs_raw_and_simple_baselines")
    checkpoint = a.out / "repair_model.pt"
    torch.save({"model": model.state_dict(), "protocol": {"seed": 17, "steps": a.steps, "height": a.height}}, checkpoint)
    report = {
        "status": "COMPLETED_TINY_HELDOUT_PILOT",
        "verdict": verdict,
        "scope": "one held-out episode, future-RGB repair only; not physical-correction or VLA-success evidence",
        "fixed_protocol": {"seed": 17, "steps": a.steps, "height": a.height, "selection": "none; no validation or test-time checkpoint selection"},
        "groups": {"train": sorted(train_groups), "test": sorted(test_groups)},
        "hashes": {"train_manifest": train_manifest_sha, "train_wam": train_wam_sha, "test_manifest": test_manifest_sha, "test_wam": test_wam_sha, "code": sha(Path(__file__)), "model_code": sha(Path(__file__).parent / "video_repair_overfit.py"), "checkpoint": sha(checkpoint)},
        "wam_protocol": train_protocol,
        "training": {"loss_history": history, "peak_memory_bytes": torch.cuda.max_memory_allocated()},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()},
        "metrics": {"raw_mae": values["raw"], "copy_current_mae": values["copy_current"],
                    "anchored_mae": values["anchored"], "repair_mae": values["repair"],
                    "oracle_clean_identity_mae": metric(identity, target),
                    "repair_minus_raw_mae": values["repair"] - values["raw"],
                    "raw_temporal_mae": temporal_metric(raw, target), "repair_temporal_mae": temporal_metric(repair, target)},
        "heldout_prediction": {"file": out_npz.name, "sha256": sha(out_npz)},
        "limitations": ["n=1 held-out episode", "no certified contact/penetration metric", "π0.5 does not consume imagined future", "no VLA control-success claim"],
    }
    (a.out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
