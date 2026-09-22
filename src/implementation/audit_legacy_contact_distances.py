"""Audit legacy-v7 and explicit-mask-v8 D0 contact-distance tensors.

The v7 collector initialized ``contact_min_distance`` to zero and then stored
the minimum MuJoCo contact distance.  Consequently zero conflates "no observed
pair" and an exactly-zero contact. V8 adds ``contact_distance_observed`` and a
NaN-outside-mask signed value. This program is deliberately read-only: it
never calls either simulator quantity an image-level penetration label.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d0", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    manifest_path = a.d0 / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    legacy = "wam2repair-physical-bank-v7-analytic-mj-ray-depth-per-frame-exact-id-gated"
    v8 = "wam2repair-physical-bank-v8-analytic-mj-ray-depth-per-frame-exact-id-gated-contact-distance-mask"
    if schema not in (legacy, v8):
        raise ValueError("this audit only accepts immutable v7 or v8 D0")
    rows = []
    for record in manifest.get("records", []):
        f = a.d0 / record["file"]
        if sha(f) != record["sha256"]:
            raise ValueError(f"candidate SHA mismatch: {f.name}")
        with np.load(f, allow_pickle=False) as z:
            c = z["contact_matrix"].astype(bool)
            if schema == legacy:
                d = z["contact_min_distance"].astype(np.float64)
                observed = None
            else:
                d = z["contact_signed_distance_m"].astype(np.float64)
                observed = z["contact_distance_observed"].astype(bool)
        if d.shape != c.shape or d.ndim != 3 or d.shape[-1] != d.shape[-2]:
            raise ValueError(f"unexpected contact tensor contract: {f.name}")
        diagonal = np.eye(d.shape[-1], dtype=bool)[None]
        offdiag = ~diagonal
        if observed is not None:
            if observed.shape != d.shape:
                raise ValueError(f"v8 mask shape mismatch: {f.name}")
            if not (np.isfinite(d[observed]).all() and np.isnan(d[~observed]).all()):
                raise ValueError(f"v8 finite-inside/NaN-outside mask violation: {f.name}")
        strict_negative = (d < 0) & offdiag
        observed_contact = c & offdiag
        # A legacy zero among a contact edge may be exact contact, or an
        # implementation/precision corner.  It must remain explicitly unknown.
        ambiguous_zero = (d == 0) & offdiag if observed is None else np.zeros_like(c, dtype=bool)
        rows.append({
            "file": f.name,
            "candidate_sha256": record["sha256"],
            "frames": int(d.shape[0]),
            "strict_negative_directed_entries": int(strict_negative.sum()),
            "strict_negative_undirected_pairs": int(strict_negative.sum() // 2),
            "negative_min_m": (float(d[strict_negative].min()) if strict_negative.any() else None),
            "observed_contact_directed_entries": int(observed_contact.sum()),
            "ambiguous_zero_directed_entries": int(ambiguous_zero.sum()),
            "negative_without_contact_flag": int((strict_negative & ~observed_contact).sum()),
            "distance_encoding": ("legacy-v7-zero-filled" if observed is None else "v8-explicit-mask-nan-outside"),
            "distance_observed_directed_entries": (None if observed is None else int((observed & offdiag).sum())),
        })
    report = {
        "schema": "wam2repair-legacy-v7-contact-distance-audit-v1",
        "status": "AUDITED_LEGACY_CONTACT_DISTANCE_SEMANTICS",
        "scope": "read-only audit of simulator contact metadata; not a WAM-image penetration metric or repair result",
        "d0_manifest_sha256": sha(manifest_path),
        "records": rows,
        "aggregate": {
            "strict_negative_directed_entries": sum(r["strict_negative_directed_entries"] for r in rows),
            "observed_contact_directed_entries": sum(r["observed_contact_directed_entries"] for r in rows),
            "ambiguous_zero_directed_entries": sum(r["ambiguous_zero_directed_entries"] for r in rows),
        },
        "distance_encoding": ("legacy-v7-zero-filled" if schema == legacy else "v8-explicit-mask-nan-outside"),
        "interpretation": [
            "Negative MuJoCo contact distances are simulator constraint/contact quantities, not mesh-SDF penetration certificates.",
            "Legacy zeros are ambiguous by construction and must not be used as signed-distance supervision.",
            "No statement about WAM-generated RGB, visual contact, physical repair, action ranking, or VLA benefit follows from this audit.",
        ],
        "code_sha256": sha(Path(__file__)),
    }
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
