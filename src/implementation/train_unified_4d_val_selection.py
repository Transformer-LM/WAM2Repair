"""Heldout-free selection for a unified RGB/depth/contact/pose/distance head."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from real_wam_repair_pilot import load_sample, sha
from real_wam_repair_geometry_val_holdout import episode_key, load_geometry


def load_full_geometry(pair, result):
    base = load_geometry(pair, result)
    report = json.loads(result.read_text())
    path = result.parent / Path(report["geometry_target"]).name
    with np.load(path, allow_pickle=False) as d:
        contact = d["contact_matrix"].astype(np.float32)
        pos = d["object_pos"].astype(np.float32)
        rot = d["object_rot"].astype(np.float32)
        if "contact_distance_observed" not in d or "contact_signed_distance_m" not in d:
            raise ValueError("this distance-head gate requires v8 explicit contact-distance targets")
        distance_observed = d["contact_distance_observed"].astype(bool)
        signed_distance = d["contact_signed_distance_m"].astype(np.float32)
    if contact.shape != (5, 8, 8) or pos.shape != (5, 7, 3) or rot.shape != (5, 7, 3, 3):
        raise ValueError("fixed task-0 geometry-head tensor contract violated")
    if not (np.isfinite(contact).all() and np.isfinite(pos).all() and np.isfinite(rot).all()):
        raise ValueError("nonfinite 4D targets")
    if distance_observed.shape != contact.shape or signed_distance.shape != contact.shape:
        raise ValueError("v8 signed-distance target shape mismatch")
    if not (np.isfinite(signed_distance[distance_observed]).all() and np.isnan(signed_distance[~distance_observed]).all()):
        raise ValueError("v8 signed-distance mask contract violation")
    base.update(contact=torch.from_numpy(contact), pos=torch.from_numpy(pos), rot6=torch.from_numpy(rot[..., :2].reshape(5, 7, 6)),
                distance_observed=torch.from_numpy(distance_observed), signed_distance=torch.from_numpy(np.nan_to_num(signed_distance, nan=0.0)))
    return base


def batch(rows):
    return {key: torch.stack([r[key] for r in rows]) for key in ("raw", "target", "current", "actions", "proprio", "depth", "depth_valid", "contact", "pos", "rot6", "distance_observed", "signed_distance")}


class Unified4DRepair(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 24))
        self.encoder = nn.Conv3d(6, 24, 3, padding=1)
        self.trunk = nn.Sequential(nn.SiLU(), nn.Conv3d(24, 24, 3, padding=1), nn.SiLU())
        self.rgb = nn.Conv3d(24, 3, 3, padding=1)
        self.depth = nn.Conv3d(24, 1, 3, padding=1)
        self.state = nn.Sequential(nn.Linear(24, 48), nn.SiLU())
        self.contact = nn.Linear(48, 64)
        self.signed_distance = nn.Linear(48, 64)
        self.pos = nn.Linear(48, 21)
        self.rot6 = nn.Linear(48, 42)
        nn.init.zeros_(self.rgb.weight); nn.init.zeros_(self.rgb.bias)

    def forward(self, raw, current, actions, proprio):
        current = current[:, :, None].expand(-1, -1, raw.shape[2], -1, -1)
        cond = self.condition(torch.cat([actions.flatten(1), proprio], 1))[:, :, None, None, None]
        h = self.trunk(self.encoder(torch.cat([raw, current], 1)) + cond)
        pooled = F.adaptive_avg_pool3d(h, (raw.shape[2], 1, 1)).squeeze(-1).squeeze(-1).transpose(1, 2)
        state = self.state(pooled)
        return (raw + self.rgb(h)).clamp(0, 1), self.depth(h).squeeze(1), self.contact(state).view(-1, raw.shape[2], 8, 8), self.signed_distance(state).view(-1, raw.shape[2], 8, 8), self.pos(state).view(-1, raw.shape[2], 7, 3), self.rot6(state).view(-1, raw.shape[2], 7, 6)


def main():
    p = argparse.ArgumentParser()
    for name in ("train-pair", "train-wam", "train-geometry"):
        p.add_argument("--" + name, type=Path, action="append", required=True)
    p.add_argument("--validation-pair", type=Path, required=True); p.add_argument("--validation-wam", type=Path, required=True); p.add_argument("--validation-geometry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--steps", type=int, default=400); p.add_argument("--seed", type=int, default=17)
    p.add_argument("--contact-pos-weight", type=float, default=1.0)
    p.add_argument("--depth-loss-weight", type=float, default=0.1)
    p.add_argument("--signed-distance-loss-weight", type=float, default=0.05)
    a = p.parse_args()
    if len(a.train_pair) != len(a.train_wam) or len(a.train_pair) != len(a.train_geometry) or len(a.train_pair) < 2 or a.steps < 50 or a.steps % 50 or a.contact_pos_weight <= 0 or a.depth_loss_weight < 0 or a.signed_distance_loss_weight < 0:
        p.error("requires at least two aligned train items and steps divisible by 50")
    if len(set([episode_key(x) for x in a.train_pair] + [episode_key(a.validation_pair)])) != len(a.train_pair) + 1:
        raise ValueError("episode overlap")
    a.out.mkdir(parents=True, exist_ok=False); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); torch.set_num_threads(4)
    train = [load_sample(q, w) | load_full_geometry(q, g) for q,w,g in zip(a.train_pair,a.train_wam,a.train_geometry)]
    val = load_sample(a.validation_pair,a.validation_wam) | load_full_geometry(a.validation_pair,a.validation_geometry)
    dev=torch.device("cuda"); tr={k:v.to(dev) for k,v in batch(train).items()}; va={k:v.to(dev) for k,v in batch([val]).items()}
    model=Unified4DRepair().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    def loss(out,b):
        rgb, dep, con, signed, pos, rot=out; valid=b["depth_valid"][:,1:]
        lr=(rgb[:,:,1:]-b["target"][:,:,1:]).abs().mean(); lt=((rgb[:,:,1:]-rgb[:,:,:-1])-(b["target"][:,:,1:]-b["target"][:,:,:-1])).abs().mean()
        ld=(dep[:,1:][valid]-torch.log(b["depth"][:,1:].clamp_min(1e-4))[valid]).abs().mean(); lc=F.binary_cross_entropy_with_logits(con,b["contact"],pos_weight=torch.as_tensor(a.contact_pos_weight,device=con.device)); lp=(pos-b["pos"]).abs().mean(); lo=(rot-b["rot6"]).abs().mean()
        signed_mask = b["distance_observed"]
        ls = (signed[signed_mask] - b["signed_distance"][signed_mask]).abs().mean() if signed_mask.any() else signed.sum() * 0.0
        return {"rgb":lr,"temporal":lt,"depth":ld,"contact":lc,"signed_distance":ls,"signed_distance_observed_count":signed_mask.sum(),"position":lp,"rotation6":lo,"total":lr+.1*lt+a.depth_loss_weight*ld+.05*lc+a.signed_distance_loss_weight*ls+.1*lp+.1*lo}
    best=(float("inf"),None,None); history=[]
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True); terms=loss(model(tr["raw"],tr["current"],tr["actions"],tr["proprio"]),tr); terms["total"].backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (step+1)%50==0:
            with torch.no_grad(): v=loss(model(va["raw"],va["current"],va["actions"],va["proprio"]),va)
            rec={"step":step+1,"train_total":float(terms["total"]),**{"validation_"+k:float(x) for k,x in v.items()}}; history.append(rec); print(json.dumps(rec),flush=True)
            if float(v["total"])<best[0]: best=(float(v["total"]),step+1,{k:x.detach().cpu().clone() for k,x in model.state_dict().items()})
    if best[2] is None: raise RuntimeError("no checkpoint")
    ckpt=a.out/"unified4d_validation_selected.pt"; torch.save({"model":best[2],"best_step":best[1],"best_validation_total":best[0]},ckpt)
    report={"schema":"wam2repair-unified-4d-selection-v2-explicit-distance","status":"SELECTED_UNIFIED4D_CHECKPOINT_WITHOUT_HELDOUT_ACCESS","scope":"episode-isolated multi-train/one-validation unified 4D+masked-distance head selection; no heldout path accepted","no_copy_current_frame_baseline":True,"contact_pos_weight":a.contact_pos_weight,"depth_loss_weight":a.depth_loss_weight,"signed_distance_loss_weight":a.signed_distance_loss_weight,"best_step":best[1],"history":history,"checkpoint":ckpt.name,"checkpoint_sha256":sha(ckpt),"train_episodes":[episode_key(x) for x in a.train_pair],"validation_episode":episode_key(a.validation_pair),"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.out/"result.json").write_text(json.dumps(report,indent=2)); print(json.dumps({"status":report["status"],"best_step":best[1]}),flush=True)
if __name__=="__main__": main()
