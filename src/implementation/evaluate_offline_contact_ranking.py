"""Frozen unified-4D offline candidate ranking with a prediction-first ledger.

Candidate scores are written and hashed before this program opens any geometry
target.  This is deliberately an offline diagnostic, never a policy rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from real_wam_repair_pilot import as_video, sha
from train_unified_4d_val_selection import Unified4DRepair

MOVABLE = (0, 1, 2, 4, 5)
GRIPPER = 7


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runtime(pair_manifest: Path, wam_result: Path) -> dict:
    """Read only unprivileged WAM input/video; never open pair target."""
    pair = json.loads(pair_manifest.read_text(encoding="utf-8"))
    wam = json.loads(wam_result.read_text(encoding="utf-8"))
    if pair.get("schema") != "wam2repair-d0-wam-pair-v1" or pair.get("status") != "PACKED_INPUT_TARGET_SEPARATED":
        raise ValueError("invalid pair")
    if wam.get("schema") != "wam2repair-allowlisted-wam-generation-v1" or wam.get("status") != "GENERATED_INPUT_ONLY_REQUIRES_FACTUAL_EVAL":
        raise ValueError("invalid WAM generation")
    inp = (pair_manifest.parent / pair["input"]).resolve()
    target = (pair_manifest.parent / pair["target"]).resolve()
    if not inp.is_file() or not target.is_file() or sha(inp) != pair.get("input_sha256"):
        raise ValueError("pair input provenance failure")
    if Path(wam.get("input_path", "")).resolve() != inp or wam.get("input_sha256") != pair["input_sha256"]:
        raise ValueError("WAM does not bind pair input")
    allowed = wam.get("input_allowlist")
    if not isinstance(allowed, list) or len(allowed) != 1 or Path(allowed[0]).resolve() != inp:
        raise ValueError("WAM allowlist failure")
    video = wam_result.parent / Path(wam["video"]).name
    if not video.is_file() or sha(video) != wam.get("video_sha256"):
        raise ValueError("WAM video provenance failure")
    with np.load(inp, allow_pickle=False) as z:
        if set(z.files) != {"current_rgb", "proprio", "actions"}: raise ValueError("input field contract")
        current, proprio, actions = z["current_rgb"], z["proprio"], z["actions"]
    with np.load(video, allow_pickle=False) as z:
        if set(z.files) != {"frames"}: raise ValueError("WAM field contract")
        raw = z["frames"]
    if raw.shape != (5, 224, 448, 3) or current.shape != (224, 448, 3) or actions.shape != (8, 7) or proprio.shape != (8,):
        raise ValueError("runtime tensor contract")
    return {"candidate": int(pair["candidate"]), "pair": str(pair_manifest), "wam": str(wam_result),
            "pair_manifest_sha256": digest(pair_manifest), "wam_result_sha256": digest(wam_result),
            "pair_target_path": str(target), "pair_target_sha256": pair["target_sha256"],
            "raw": as_video(raw), "current": as_video(current[None])[:, 0],
            "actions": torch.from_numpy(actions.copy()).float(), "proprio": torch.from_numpy(proprio.copy()).float()}


def ordinal_rank(values):
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    rank = [0] * len(values)
    for r, i in enumerate(order): rank[i] = r + 1
    return rank, order


def spearman(a, b):
    if len(a) < 2 or len(set(a)) < 2 or len(set(b)) < 2: return None
    x, y = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(x, y)[0, 1])


def factual_contact(runtime: dict, geometry_result: Path) -> int:
    report = json.loads(geometry_result.read_text(encoding="utf-8"))
    if report.get("schema") != "wam2repair-target-only-geometry-bridge-v1" or report.get("status") != "BUILT_CONSERVATIVE_TARGET_ONLY_GEOMETRY_BRIDGE":
        raise ValueError("invalid geometry bridge")
    if report.get("pair_manifest_sha256") != runtime["pair_manifest_sha256"] or report.get("pair_target_sha256") != runtime["pair_target_sha256"]:
        raise ValueError("geometry does not bind pair")
    path = geometry_result.parent / Path(report["geometry_target"]).name
    if sha(path) != report.get("geometry_target_sha256"): raise ValueError("geometry SHA")
    with np.load(path, allow_pickle=False) as z:
        contact = z["contact_matrix"]
        offsets = z["frame_offsets"]
    if contact.shape != (5, 8, 8) or offsets.tolist() != [0, 2, 4, 6, 8] or not np.isfinite(contact).all():
        raise ValueError("factual contact contract")
    if not np.array_equal(contact, np.swapaxes(contact, 1, 2)): raise ValueError("contact symmetry")
    return int(contact[1:, GRIPPER, list(MOVABLE)].sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pair", type=Path, action="append", required=True); p.add_argument("--wam", type=Path, action="append", required=True)
    p.add_argument("--geometry", type=Path, action="append", required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not (len(a.pair) == len(a.wam) == len(a.geometry) == 4): p.error("requires exactly four aligned candidates")
    a.out.mkdir(parents=True, exist_ok=False)
    selection = json.loads(a.selection.read_text())
    if selection.get("status") != "SELECTED_UNIFIED4D_CHECKPOINT_WITHOUT_HELDOUT_ACCESS": raise ValueError("not frozen selection")
    if sha(a.checkpoint) != selection.get("checkpoint_sha256"): raise ValueError("checkpoint not selection bound")
    rows = [load_runtime(x, y) for x, y in zip(a.pair, a.wam)]
    if sorted(r["candidate"] for r in rows) != [0, 1, 2, 3]: raise ValueError("candidate identity contract")
    model = Unified4DRepair().cuda().eval(); model.load_state_dict(torch.load(a.checkpoint, map_location="cuda")["model"])
    with torch.no_grad():
        for r in rows:
            out = model(r["raw"][None].cuda(), r["current"][None].cuda(), r["actions"][None].cuda(), r["proprio"][None].cuda())
            probabilities = torch.sigmoid(out[2][0, 1:, GRIPPER, list(MOVABLE)]).cpu().numpy()
            r["predicted_contact_score"] = float(probabilities.mean())
    ledger = {"schema": "wam2repair-offline-ranking-prediction-ledger-v1", "status": "PREDICTED_BEFORE_FACTUAL_OPEN",
              "scope": "frozen repair contact-head scores only; no factual tensor opened", "selection_sha256": sha(a.selection),
              "checkpoint_sha256": sha(a.checkpoint), "movable_indices": list(MOVABLE), "gripper_index": GRIPPER,
              "scores": [{k: r[k] for k in ("candidate", "pair_manifest_sha256", "wam_result_sha256", "predicted_contact_score")} for r in rows]}
    ledger_path = a.out / "predicted_scores_before_factual.json"; ledger_path.write_text(json.dumps(ledger, indent=2))
    ledger_sha = sha(ledger_path)
    factual = [factual_contact(r, g) for r, g in zip(rows, a.geometry)]
    scores = [r["predicted_contact_score"] for r in rows]
    pred_rank, pred_order = ordinal_rank(scores); factual_rank, factual_order = ordinal_rank(factual)
    factual_discriminative = len(set(factual)) > 1
    report = {"schema":"wam2repair-offline-contact-ranking-v1", "status":"EVALUATED_FROZEN_OFFLINE_CONTACT_RANKING" if factual_discriminative else "EVALUATED_FROZEN_OFFLINE_CONTACT_RANKING_UNINFORMATIVE_TIED_FACTUAL",
              "scope":"one-state, four-candidate offline diagnostic; not pi0.5 closed loop, task success, or physical-correction evidence",
              "no_copy_current_frame_baseline":True, "prediction_ledger":"predicted_scores_before_factual.json", "prediction_ledger_sha256":ledger_sha,
              "factual_opened_only_after_prediction_ledger_saved":True,
              "candidates":[{"candidate":r["candidate"],"predicted_contact_score":s,"factual_future_contact_edge_count":f} for r,s,f in zip(rows,scores,factual)],
              "predicted_order":pred_order,"factual_order":factual_order,"predicted_top1":pred_order[0],"factual_top1":factual_order[0] if factual_discriminative else None,
              "top1_exact_match":bool(pred_order[0] == factual_order[0]) if factual_discriminative else None,"predicted_top1_factual_score":factual[pred_order[0]],
              "spearman_ordinal":spearman(pred_rank,factual_rank) if factual_discriminative else None,"factual_ranking_discriminative":factual_discriminative,"tie_break":"lowest candidate index (only used when a ranking metric is defined)", "limitations":["Tiny one-state diagnostic.","All factual candidate scores tied, so this run cannot assess action discrimination.","Predicted contact head is not certified physical contact repair.","No VLA policy action was chosen or executed."]}
    (a.out / "result.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)

if __name__ == "__main__": main()
