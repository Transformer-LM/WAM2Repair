"""Frozen contact-plus-relative-pose ranking with prediction-first factual access."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from real_wam_repair_pilot import sha
from train_unified_4d_val_selection import Unified4DRepair
from evaluate_offline_contact_ranking import (
    GRIPPER, MOVABLE, digest, load_runtime, ordinal_rank, spearman,
)

BOWL, PLATE = 0, 5


def minmax(values):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


def factual_geometry(runtime, geometry_result: Path):
    """Open targets only after the immutable prediction ledger exists."""
    report = json.loads(geometry_result.read_text(encoding="utf-8"))
    if report.get("schema") != "wam2repair-target-only-geometry-bridge-v1" or report.get("status") != "BUILT_CONSERVATIVE_TARGET_ONLY_GEOMETRY_BRIDGE":
        raise ValueError("invalid geometry bridge")
    if report.get("pair_manifest_sha256") != runtime["pair_manifest_sha256"] or report.get("pair_target_sha256") != runtime["pair_target_sha256"]:
        raise ValueError("geometry does not bind pair")
    path = geometry_result.parent / Path(report["geometry_target"]).name
    if not path.is_file() or sha(path) != report.get("geometry_target_sha256"):
        raise ValueError("geometry SHA")
    with np.load(path, allow_pickle=False) as z:
        contact, pos, offsets = z["contact_matrix"], z["object_pos"], z["frame_offsets"]
    if contact.shape != (5, 8, 8) or pos.shape != (5, 7, 3) or offsets.tolist() != [0, 2, 4, 6, 8]:
        raise ValueError("factual geometry contract")
    if not np.array_equal(contact, np.swapaxes(contact, 1, 2)):
        raise ValueError("contact symmetry")
    return int(contact[1:, GRIPPER, list(MOVABLE)].sum()), -float(np.linalg.norm(pos[-1, BOWL] - pos[-1, PLATE]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pair", type=Path, action="append", required=True); p.add_argument("--wam", type=Path, action="append", required=True)
    p.add_argument("--geometry", type=Path, action="append", required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not (len(a.pair) == len(a.wam) == len(a.geometry) == 4):
        p.error("requires exactly four aligned candidates")
    a.out.mkdir(parents=True, exist_ok=False)
    selection = json.loads(a.selection.read_text(encoding="utf-8"))
    if selection.get("status") != "SELECTED_UNIFIED4D_CHECKPOINT_WITHOUT_HELDOUT_ACCESS":
        raise ValueError("not frozen selection")
    if sha(a.checkpoint) != selection.get("checkpoint_sha256"):
        raise ValueError("checkpoint not selection bound")
    rows = [load_runtime(x, y) for x, y in zip(a.pair, a.wam)]
    if sorted(r["candidate"] for r in rows) != [0, 1, 2, 3]:
        raise ValueError("candidate identity contract")
    model = Unified4DRepair().cuda().eval()
    model.load_state_dict(torch.load(a.checkpoint, map_location="cuda")["model"])
    with torch.no_grad():
        for r in rows:
            out = model(r["raw"][None].cuda(), r["current"][None].cuda(), r["actions"][None].cuda(), r["proprio"][None].cuda())
            contact = torch.sigmoid(out[2][0, 1:, GRIPPER, list(MOVABLE)]).mean()
            rel_dist = torch.linalg.vector_norm(out[3][0, -1, BOWL] - out[3][0, -1, PLATE])
            r["predicted_contact_score"] = float(contact.cpu())
            r["predicted_negative_final_bowl_plate_distance"] = -float(rel_dist.cpu())
    contact_norm = minmax([r["predicted_contact_score"] for r in rows])
    dist_norm = minmax([r["predicted_negative_final_bowl_plate_distance"] for r in rows])
    for r, cn, dn in zip(rows, contact_norm, dist_norm):
        r["predicted_contact_minmax"] = cn; r["predicted_distance_minmax"] = dn; r["predicted_combined_score"] = 0.5 * (cn + dn)
    ledger = {"schema":"wam2repair-offline-combined-ranking-prediction-ledger-v1", "status":"PREDICTED_BEFORE_FACTUAL_OPEN",
              "scope":"frozen repair contact and relative-pose heads; no factual tensor opened", "selection_sha256":sha(a.selection), "checkpoint_sha256":sha(a.checkpoint),
              "proxy":"equal-weight, per-set min-max normalized predicted future contact and negative final bowl-plate distance", "scores":[{k:r[k] for k in ("candidate", "pair_manifest_sha256", "wam_result_sha256", "predicted_contact_score", "predicted_negative_final_bowl_plate_distance", "predicted_contact_minmax", "predicted_distance_minmax", "predicted_combined_score")} for r in rows]}
    ledger_path = a.out / "predicted_scores_before_factual.json"; ledger_path.write_text(json.dumps(ledger, indent=2)); ledger_sha = sha(ledger_path)
    facts = [factual_geometry(r, g) for r, g in zip(rows, a.geometry)]
    fc_norm, fd_norm = minmax([x[0] for x in facts]), minmax([x[1] for x in facts])
    for r, (fc, fd), cn, dn in zip(rows, facts, fc_norm, fd_norm):
        r["factual_future_contact_edge_count"] = fc; r["factual_negative_final_bowl_plate_distance"] = fd
        r["factual_combined_score"] = .5 * (cn + dn)
    pred = [r["predicted_combined_score"] for r in rows]; factual = [r["factual_combined_score"] for r in rows]
    pred_rank, pred_order = ordinal_rank(pred); factual_rank, factual_order = ordinal_rank(factual)
    report = {"schema":"wam2repair-offline-combined-ranking-v1", "status":"EVALUATED_FROZEN_OFFLINE_COMBINED_RANKING",
              "scope":"one-state, four-candidate offline diagnostic; no pi0.5 closed loop, task-success result, or physical-correction proof", "no_copy_current_frame_baseline":True,
              "prediction_ledger":"predicted_scores_before_factual.json", "prediction_ledger_sha256":ledger_sha, "factual_opened_only_after_prediction_ledger_saved":True,
              "proxy":"equal-weight per-set min-max contact plus negative final bowl-plate distance", "candidates":[{k:r[k] for k in ("candidate", "predicted_contact_score", "predicted_negative_final_bowl_plate_distance", "predicted_contact_minmax", "predicted_distance_minmax", "predicted_combined_score", "factual_future_contact_edge_count", "factual_negative_final_bowl_plate_distance", "factual_combined_score")} for r in rows],
              "predicted_order":pred_order, "factual_order":factual_order, "predicted_top1":pred_order[0], "factual_top1":factual_order[0], "top1_exact_match":bool(pred_order[0] == factual_order[0]), "predicted_top1_factual_score":factual[pred_order[0]], "spearman_ordinal":spearman(pred_rank, factual_rank), "tie_break":"lowest candidate index", "limitations":["One controlled state only.", "All candidates have task_success=false; this does not test success selection.", "Predicted heads are not certified geometric/contact repair.", "No selected action was executed by pi0.5."]}
    (a.out / "result.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
