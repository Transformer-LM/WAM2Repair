"""One heldout evaluation of a frozen unified 4D repair checkpoint."""
import argparse, hashlib, json
from pathlib import Path
import torch

from real_wam_repair_pilot import load_sample, sha
from real_wam_repair_geometry_val_holdout import episode_key
from train_unified_4d_val_selection import Unified4DRepair, batch, load_full_geometry

MOVABLE = (0, 1, 2, 4, 5)
GRIPPER = 7


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection-result',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--heldout-pair',type=Path,required=True); p.add_argument('--heldout-wam',type=Path,required=True); p.add_argument('--heldout-geometry',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); s=json.loads(a.selection_result.read_text())
    if s.get('status')!='SELECTED_UNIFIED4D_CHECKPOINT_WITHOUT_HELDOUT_ACCESS': raise ValueError('not heldout-free selection')
    if a.checkpoint.name!=s.get('checkpoint') or sha(a.checkpoint)!=s.get('checkpoint_sha256'): raise ValueError('checkpoint binding failed')
    a.out.mkdir(parents=True,exist_ok=False)
    sample=load_sample(a.heldout_pair,a.heldout_wam)|load_full_geometry(a.heldout_pair,a.heldout_geometry)
    dev=torch.device('cuda'); b={k:v.to(dev) for k,v in batch([sample]).items()}
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True); model=Unified4DRepair().to(dev); model.load_state_dict(payload['model']); model.eval()
    with torch.no_grad():
        rgb,depth,contact,signed_distance,pos,rot=model(b['raw'],b['current'],b['actions'],b['proprio']); valid=b['depth_valid'][:,1:]
        pred_contact = contact.sigmoid() >= .5
        true_contact = b['contact'] >= .5
        contact_tp = (pred_contact & true_contact).sum()
        contact_fp = (pred_contact & ~true_contact).sum()
        contact_fn = (~pred_contact & true_contact).sum()
        contact_precision = contact_tp.float() / (contact_tp + contact_fp).clamp_min(1)
        contact_recall = contact_tp.float() / (contact_tp + contact_fn).clamp_min(1)
        contact_f1 = 2 * contact_precision * contact_recall / (contact_precision + contact_recall).clamp_min(1e-12)
        # Report the task-relevant future gripper--movable edges separately:
        # the full 8x8 matrix is dominated by structural negatives.
        actionable_pred = pred_contact[:, 1:, GRIPPER, list(MOVABLE)]
        actionable_true = true_contact[:, 1:, GRIPPER, list(MOVABLE)]
        a_tp = (actionable_pred & actionable_true).sum()
        a_fp = (actionable_pred & ~actionable_true).sum()
        a_fn = (~actionable_pred & actionable_true).sum()
        a_precision = a_tp.float() / (a_tp + a_fp).clamp_min(1)
        a_recall = a_tp.float() / (a_tp + a_fn).clamp_min(1)
        a_f1 = 2 * a_precision * a_recall / (a_precision + a_recall).clamp_min(1e-12)
        signed_mask = b['distance_observed']
        signed_mae = ((signed_distance[signed_mask]-b['signed_distance'][signed_mask]).abs().mean() if signed_mask.any() else None)
        metrics={
          'raw_rgb_mae':float((b['raw'][:,:,1:]-b['target'][:,:,1:]).abs().mean()),
          'repair_rgb_mae':float((rgb[:,:,1:]-b['target'][:,:,1:]).abs().mean()),
          'raw_temporal_mae':float(((b['raw'][:,:,1:]-b['raw'][:,:,:-1])-(b['target'][:,:,1:]-b['target'][:,:,:-1])).abs().mean()),
          'repair_temporal_mae':float(((rgb[:,:,1:]-rgb[:,:,:-1])-(b['target'][:,:,1:]-b['target'][:,:,:-1])).abs().mean()),
          'depth_log_mae':float((depth[:,1:][valid]-torch.log(b['depth'][:,1:].clamp_min(1e-4))[valid]).abs().mean()),
          'contact_bce':float(torch.nn.functional.binary_cross_entropy_with_logits(contact,b['contact'])),
          'contact_accuracy':float((pred_contact==true_contact).float().mean()),
          'contact_positive_count':int(true_contact.sum()), 'contact_total_count':int(true_contact.numel()),
          'contact_precision':float(contact_precision), 'contact_recall':float(contact_recall), 'contact_f1':float(contact_f1),
          'future_gripper_movable_positive_count':int(actionable_true.sum()), 'future_gripper_movable_total_count':int(actionable_true.numel()),
          'future_gripper_movable_precision':float(a_precision), 'future_gripper_movable_recall':float(a_recall), 'future_gripper_movable_f1':float(a_f1),
          'signed_distance_observed_count':int(signed_mask.sum()),
          'signed_distance_masked_mae_m':(float(signed_mae) if signed_mae is not None else None),
          'position_mae':float((pos-b['pos']).abs().mean()), 'rotation6_mae':float((rot-b['rot6']).abs().mean())}
        metrics['repair_minus_raw_rgb_mae']=metrics['repair_rgb_mae']-metrics['raw_rgb_mae']
        metrics['repair_minus_raw_temporal_mae']=metrics['repair_temporal_mae']-metrics['raw_temporal_mae']
        if not all(torch.isfinite(t).all() for t in (rgb,depth,contact,signed_distance,pos,rot)): raise RuntimeError('nonfinite output')
    report={'schema':'wam2repair-unified4d-frozen-heldout-v2-explicit-distance','status':'EVALUATED_FROZEN_UNIFIED4D_HELDOUT','scope':'one frozen-checkpoint heldout multi-head prediction diagnostic; no physical or VLA claim','no_copy_current_frame_baseline':True,'selection_sha256':sha(a.selection_result),'checkpoint_sha256':sha(a.checkpoint),'heldout_episode':episode_key(a.heldout_pair),'pair_sha256':sha(a.heldout_pair),'wam_sha256':sha(a.heldout_wam),'geometry_sha256':sha(a.heldout_geometry),'metrics':metrics,'contact_frames':sample['contact_frames'],'limitations':['One heldout episode; no generalization conclusion.','Masked MuJoCo contact distance is not a mesh SDF, a WAM-image penetration certificate, or visual physical repair evidence.','No action-ranking or pi0.5 closed-loop experiment.'],'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.out/'result.json').write_text(json.dumps(report,indent=2)); print(json.dumps({'status':report['status'],'metrics':metrics}),flush=True)
if __name__=='__main__': main()
