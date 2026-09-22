"""Attach a target-only, conservative geometry bridge to one v7/v8 D0/WAM pair.

The WAM input remains untouched. RGB uses the existing PIL-bilinear WAM image
bridge, while sparse analytic depth/geometry uses deterministic nearest-neighbor
coordinate mapping after the same 180-degree rotation. A remapped depth pixel
is valid only if its exact source depth was valid, finite and ID-gated.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_rotated(x: np.ndarray) -> np.ndarray:
    """Map raw 256x256 storage to the 224x224 WAM coordinate grid."""
    if x.ndim == 2:
        return np.asarray(Image.fromarray(x[::-1, ::-1]).resize((224, 224), Image.Resampling.NEAREST))
    raise ValueError(f"expected HxW field, got {x.shape}")


def wam_rgb(raw: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(raw[::-1, ::-1]).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d0", type=Path, required=True)
    p.add_argument("--pair-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    d0_manifest = json.loads((a.d0 / "manifest.json").read_text(encoding="utf-8"))
    pair = json.loads(a.pair_manifest.read_text(encoding="utf-8"))
    schemas = {
        "wam2repair-physical-bank-v7-analytic-mj-ray-depth-per-frame-exact-id-gated",
        "wam2repair-physical-bank-v8-analytic-mj-ray-depth-per-frame-exact-id-gated-contact-distance-mask",
    }
    if d0_manifest.get("schema") not in schemas:
        raise ValueError("requires v7 or v8 analytic-ray D0")
    if pair.get("schema") != "wam2repair-d0-wam-pair-v1" or pair.get("d0_manifest_sha256") != sha(a.d0 / "manifest.json"):
        raise ValueError("pair does not bind this D0 manifest")
    records = d0_manifest.get("records", [])
    matched = [r for r in records if r.get("sha256") == pair.get("d0_candidate_sha256")]
    if len(matched) != 1: raise ValueError("pair candidate does not bind exactly one D0 record")
    source = a.d0 / matched[0]["file"]
    if sha(source) != pair["d0_candidate_sha256"]: raise ValueError("D0 candidate SHA mismatch")
    target_path = (a.pair_manifest.parent / pair["target"]).resolve()
    if sha(target_path) != pair["target_sha256"]: raise ValueError("pair factual target SHA mismatch")
    with np.load(target_path, allow_pickle=False) as target:
        if target["frame_offsets"].tolist() != pair["frame_offsets"]: raise ValueError("pair target timing mismatch")
        pair_video = target["video"].copy()
    offsets = np.asarray(pair["frame_offsets"], dtype=int)
    views = ("agentview", "robot0_eye_in_hand")
    with np.load(source, allow_pickle=False) as d:
        n = len(d["actions"])
        if n != pair["horizon_actions"] or offsets.tolist() != [0, 2, 4, 6, 8]: raise ValueError("unsupported D0 timing")
        depth_rows, valid_rows, geom_rows, rgb_rows = [], [], [], []
        for t in offsets:
            depths, valids, geoms, rgbs = [], [], [], []
            for view in views:
                raw_depth = d[view + "_depth_m"][t]
                raw_valid = d[view + "_depth_valid_mask"][t]
                raw_geom = d[view + "_ray_geom_id"][t]
                if raw_depth.shape != (256, 256) or raw_valid.shape != (256, 256) or raw_geom.shape != (256, 256):
                    raise ValueError("unexpected D0 geometry raster shape")
                raw_finite = np.isfinite(raw_depth)
                depth = nearest_rotated(np.where(raw_finite, raw_depth, 0.0).astype(np.float32))
                valid = nearest_rotated(raw_valid.astype(np.uint8)).astype(bool)
                finite = nearest_rotated(raw_finite.astype(np.uint8)).astype(bool)
                geom = nearest_rotated(raw_geom.astype(np.int32)).astype(np.int32)
                valid &= finite & np.isfinite(depth) & (depth > 0) & (geom >= 0)
                depth = depth.astype(np.float32); depth[~valid] = np.nan
                depths.append(depth); valids.append(valid); geoms.append(geom)
                rgbs.append(wam_rgb(d[view + "_rgb"][t]))
            depth_rows.append(np.concatenate(depths, axis=1))
            valid_rows.append(np.concatenate(valids, axis=1))
            geom_rows.append(np.concatenate(geoms, axis=1))
            rgb_rows.append(np.concatenate(rgbs, axis=1))
        rgb_rows = np.stack(rgb_rows)
        if not np.array_equal(rgb_rows, pair_video): raise ValueError("geometry source RGB does not reproduce pair factual target")
        depth = np.stack(depth_rows).astype(np.float32); valid = np.stack(valid_rows); geom = np.stack(geom_rows)
        if depth.shape != (5, 224, 448) or valid.shape != depth.shape or geom.shape != depth.shape:
            raise ValueError("bridged geometry shape mismatch")
        if np.isfinite(depth[valid]).sum() != valid.sum() or np.isfinite(depth[~valid]).any():
            raise ValueError("invalid depth/valid-mask contract")
        contact = d["contact_matrix"][offsets].copy()
        object_pos, object_rot = d["object_pos"][offsets].copy(), d["object_rot"][offsets].copy()
        if contact.shape[0] != 5 or object_pos.shape[0] != 5 or object_rot.shape[0] != 5:
            raise ValueError("factual geometry metadata frame count mismatch")
        if not np.isfinite(object_pos).all() or not np.isfinite(object_rot).all():
            raise ValueError("nonfinite factual pose metadata")
        distance_kwargs = {}
        distance_contract = "legacy-v7-zero-filled-distance-not-exported"
        if d0_manifest["schema"].startswith("wam2repair-physical-bank-v8-"):
            observed = d["contact_distance_observed"][offsets].copy()
            signed = d["contact_signed_distance_m"][offsets].copy()
            if observed.shape != contact.shape or signed.shape != contact.shape:
                raise ValueError("v8 contact-distance tensor shape mismatch")
            if not (np.isfinite(signed[observed]).all() and np.isnan(signed[~observed]).all()):
                raise ValueError("v8 contact signed-distance/mask contract violation")
            distance_kwargs = {"contact_distance_observed": observed, "contact_signed_distance_m": signed}
            distance_contract = "v8: finite only for MuJoCo-reported owned pairs; NaN otherwise; not an SDF or visual penetration label"
    out = a.out / "geometry_target.npz"
    np.savez_compressed(out, depth_m=depth, depth_valid_mask=valid, ray_geom_id=geom, contact_matrix=contact,
                        object_pos=object_pos, object_rot=object_rot, frame_offsets=offsets, **distance_kwargs)
    report = {"schema": "wam2repair-target-only-geometry-bridge-v1", "status": "BUILT_CONSERVATIVE_TARGET_ONLY_GEOMETRY_BRIDGE",
              "scope": "factual supervision metadata only; not a WAM input, model result, or physical repair claim",
              "pair_manifest_sha256": sha(a.pair_manifest), "d0_manifest_sha256": sha(a.d0 / "manifest.json"),
              "d0_candidate_sha256": sha(source), "pair_target_sha256": sha(target_path), "geometry_target": out.name, "geometry_target_sha256": sha(out),
              "frame_offsets": offsets.tolist(), "image_contract": "RGB: rotate_180+PIL-bilinear; geometry: rotate_180+PIL-nearest from exact valid analytic-ray source pixels",
              "coverage_per_frame": [float(x) for x in valid.mean(axis=(1, 2))],
              "invalid_depth": "NaN", "contact_distance_contract": distance_contract,
              "target_only": True, "code_sha256": sha(Path(__file__))}
    (a.out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "coverage": report["coverage_per_frame"]}), flush=True)


if __name__ == "__main__": main()
