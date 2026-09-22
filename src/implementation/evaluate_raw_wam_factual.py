"""Independently evaluate one generated WAM video against its factual target.

This is deliberately post-generation code: the generator has no target path,
whereas this program requires explicit, SHA-bound pair, generation, and target
artifacts.  It reports image/temporal discrepancies only.  It does not infer
physical contact or penetration from RGB, and it does not implement a repair
model or a VLA evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pair-manifest", type=Path, required=True)
    p.add_argument("--wam-result", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    pair = load_json(a.pair_manifest)
    wam = load_json(a.wam_result)
    if pair.get("schema") != "wam2repair-d0-wam-pair-v1":
        raise ValueError("unrecognized pair manifest")
    if pair.get("status") != "PACKED_INPUT_TARGET_SEPARATED":
        raise ValueError("pair is not an input/target-separated artifact")
    if wam.get("schema") != "wam2repair-allowlisted-wam-generation-v1":
        raise ValueError("unrecognized WAM result")
    if wam.get("status") != "GENERATED_INPUT_ONLY_REQUIRES_FACTUAL_EVAL":
        raise ValueError("WAM artifact is not an input-only generated output")
    if wam.get("input_sha256") != pair.get("input_sha256"):
        raise ValueError("WAM generation input does not match pair allowlisted input")
    expected_input = (a.pair_manifest.parent / pair["input"]).resolve()
    if Path(wam.get("input_path", "")).resolve() != expected_input:
        raise ValueError("WAM result input path does not resolve to pair allowlisted input")
    allowlist = wam.get("input_allowlist")
    if not isinstance(allowlist, list) or len(allowlist) != 1 or Path(allowlist[0]).resolve() != expected_input:
        raise ValueError("WAM result allowlist is not exactly the paired input")
    if a.target.resolve() != (a.pair_manifest.parent / pair["target"]).resolve():
        raise ValueError("explicit target path does not resolve to pair privileged target")
    if sha(a.target) != pair.get("target_sha256"):
        raise ValueError("explicit factual target SHA does not match pair manifest")

    video_name = Path(str(wam["video"]))
    if video_name.name != str(video_name) or video_name.suffix != ".npz":
        raise ValueError("WAM video must be a local NPZ basename")
    video_path = a.wam_result.parent / video_name
    if not video_path.is_file() or sha(video_path) != wam.get("video_sha256"):
        raise ValueError("generated video SHA mismatch")
    with np.load(video_path, allow_pickle=False) as d:
        if set(d.files) != {"frames"}:
            raise ValueError("unexpected WAM video keys")
        pred = d["frames"].astype(np.float32)
    with np.load(a.target, allow_pickle=False) as d:
        required = {"video", "frame_offsets", "times", "contact", "object_pos", "object_rot", "eef_pos", "eef_rot", "success"}
        if set(d.files) != required:
            raise ValueError("unexpected privileged factual target keys")
        target = d["video"].astype(np.float32)
        offsets = d["frame_offsets"].astype(int)
        times = d["times"].astype(np.float64)
        contact = d["contact"].astype(int)
        object_pos = d["object_pos"]
        object_rot = d["object_rot"]
        eef_pos = d["eef_pos"]
        eef_rot = d["eef_rot"]
        success = d["success"]
    if pred.shape != target.shape or pred.shape != (5, 224, 448, 3):
        raise ValueError(f"WAM/factual shape mismatch: {pred.shape} vs {target.shape}")
    if not (np.isfinite(pred).all() and np.isfinite(target).all()):
        raise ValueError("nonfinite image tensor")
    if offsets.shape != (5,) or offsets.tolist() != pair.get("frame_offsets"):
        raise ValueError("factual frame offsets do not match registered pair timing")
    if not np.all(np.diff(offsets) > 0) or offsets[-1] != pair.get("horizon_actions"):
        raise ValueError("factual frame offsets are not a strictly increasing full-horizon schedule")
    if times.shape != (5,) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("factual timestamps are not finite and strictly increasing")
    metadata = {"contact": contact, "object_pos": object_pos, "object_rot": object_rot,
                "eef_pos": eef_pos, "eef_rot": eef_rot, "success": success}
    if any(x.ndim < 1 or x.shape[0] != 5 for x in metadata.values()):
        raise ValueError("factual metadata is not aligned to the five factual frames")
    if not all(np.isfinite(x).all() for x in metadata.values()):
        raise ValueError("nonfinite factual metadata")

    abs_err = np.abs(pred - target)
    sq_err = (pred - target) ** 2
    per_frame_mae = abs_err.mean(axis=(1, 2, 3))
    per_frame_mse = sq_err.mean(axis=(1, 2, 3))
    pred_delta = pred[1:] - pred[:-1]
    target_delta = target[1:] - target[:-1]
    temporal_delta_mae = np.abs(pred_delta - target_delta).mean(axis=(1, 2, 3))

    # Visual audit sheet is only a convenience for later human inspection.
    # WAM row / factual row / scaled absolute-error row; no repaired or copied row.
    tiles = []
    for t in range(5):
        err = np.clip(abs_err[t] * 3.0, 0, 255).astype(np.uint8)
        tiles.extend([pred[t].astype(np.uint8), target[t].astype(np.uint8), err])
    sheet = np.zeros((3 * 224, 5 * 448, 3), dtype=np.uint8)
    for t in range(5):
        for row in range(3):
            sheet[row * 224:(row + 1) * 224, t * 448:(t + 1) * 448] = tiles[3 * t + row]
    preview = Image.fromarray(sheet)
    draw = ImageDraw.Draw(preview)
    for t, offset in enumerate(offsets.tolist()):
        draw.text((t * 448 + 4, 4), f"frame {t}, action offset {offset}", fill=(255, 255, 0))
    preview.save(a.out / "wam_factual_error_sheet.png")

    report = {
        "schema": "wam2repair-raw-wam-factual-eval-v1",
        "status": "EVALUATED_RAW_WAM_VS_FACTUAL_NOT_PHYSICAL",
        "scope": "one post-generation factual RGB/temporal comparison; not a repair, contact, penetration, action-ranking, or VLA result",
        "pair_manifest_sha256": sha(a.pair_manifest),
        "wam_result_sha256": sha(a.wam_result),
        "target_sha256": sha(a.target),
        "wam_video_sha256": sha(video_path),
        "pair_input_sha256": pair["input_sha256"],
        "factual_frame_offsets": offsets.tolist(),
        "factual_times": [float(x) for x in times],
        "privileged_simulator_contact_metadata": contact.tolist(),
        "metrics": {
            "rgb_mae_0_255": float(abs_err.mean()),
            "rgb_mse_0_255_sq": float(sq_err.mean()),
            "per_frame_rgb_mae_0_255": [float(x) for x in per_frame_mae],
            "per_frame_rgb_mse_0_255_sq": [float(x) for x in per_frame_mse],
            "temporal_delta_mae_0_255": [float(x) for x in temporal_delta_mae],
        },
        "visual_sheet": "wam_factual_error_sheet.png",
        "prohibited_interpretations": [
            "No pixel metric certifies contact, penetration, 6D pose, or physical correctness.",
            "Single window has no statistical/generalization claim.",
            "No copy-current-frame baseline was run or reported.",
        ],
        "code_sha256": sha(Path(__file__)),
    }
    (a.out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(a.out), "rgb_mae": report["metrics"]["rgb_mae_0_255"]}), flush=True)


if __name__ == "__main__":
    main()
