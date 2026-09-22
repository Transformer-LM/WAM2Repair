"""Matched train-only RGB / RGB+depth learnability diagnostic; no physics claim."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from geometry_supervision_gate import digest
from video_repair_overfit import tensor_video


class DepthRepair(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(120, 64), nn.SiLU(), nn.Linear(64, 32))
        self.encoder = nn.Conv3d(6, 32, 3, padding=1)
        self.shared = nn.Sequential(nn.SiLU(), nn.Conv3d(32, 32, 3, padding=1), nn.SiLU())
        self.depth = nn.Conv3d(32, 1, 1)
        self.rgb = nn.Conv3d(33, 3, 3, padding=1)
        nn.init.zeros_(self.rgb.weight); nn.init.zeros_(self.rgb.bias)

    def forward(self, raw, current, actions, proprio):
        x = torch.cat([raw, current[:, :, None].expand_as(raw)], dim=1)
        c = self.condition(torch.cat([actions.flatten(1), proprio], 1))[:, :, None, None, None]
        h = self.shared(self.encoder(x) + c)
        depth = F.softplus(self.depth(h)) + 1e-4
        rgb = (raw + self.rgb(torch.cat([h, depth], 1))).clamp(0, 1)
        return rgb, depth


def main():
    p = argparse.ArgumentParser()
    for name in ['bank', 'wam', 'geometry', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--depth-weight', type=float, default=.1)
    p.add_argument('--regions', type=Path, help='Optional replay-certified target-only region masks')
    a = p.parse_args()
    assert a.steps > 0 and a.depth_weight > 0
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    bank = json.loads((a.bank / 'manifest.json').read_text())
    wam = json.loads((a.wam / 'result.json').read_text())
    geom = json.loads((a.geometry / 'result.json').read_text())
    assert geom['status'] == 'PASS_DATA_ALIGNMENT_ONLY'
    assert wam['status'] == 'GENERATED_ALIGNED'
    assert geom['bank_manifest_sha256'] == wam['manifest_sha256'] == digest(a.bank / 'manifest.json')
    rows = [r for r in bank['records'] if r['split'] == 'train'][:3]
    assert len(rows) == 3
    preds = {r['id']: r for r in wam['records']}
    targets = {r['id']: r for r in geom['records']}
    assert len(preds) == len(wam['records']) and len(targets) == 77
    arrays = {k: [] for k in ['raw', 'current', 'actions', 'proprio', 'rgb', 'depth']}
    masks = []
    if a.regions:
        region_report = json.loads((a.regions / 'result.json').read_text())
        assert region_report['status'] == 'PASS_REPLAY_ALIGNED_MASKS'
        assert region_report['bank_sha256'] == digest(a.bank / 'manifest.json')
        region_rows = {r['id']: r for r in region_report['records']}
        assert len(region_rows) == len(region_report['records']) == 3
    for r in rows:
        pr, gr = preds[r['id']], targets[r['id']]
        if a.regions:
            rr = region_rows[r['id']]
            assert rr['split'] == 'train'
            rp = a.regions / rr['target']; assert digest(rp) == rr['sha256']
            with np.load(rp, allow_pickle=False) as region_data:
                labels = region_data['labels']
                assert labels.shape == (5, 224, 448) and np.isin(labels, [0, 1, 2]).all()
                masks.append(torch.from_numpy((labels > 0).copy()).float()[None])
        assert gr['split'] == 'train' and gr['group'] == r['group']
        files = [(a.bank / r['input'], r['input_sha256']), (a.bank / r['target'], r['target_sha256']),
                 (a.wam / pr['prediction'], pr['sha256']), (a.geometry / gr['target'], gr['sha256'])]
        for path, sha in files:
            assert digest(path) == sha
        with np.load(files[0][0], allow_pickle=False) as d:
            assert set(d.files) == {'current_rgb', 'actions', 'proprio'}
            assert d['actions'].shape == (16, 7) and d['proprio'].shape == (8,)
            arrays['current'].append(tensor_video(d['current_rgb'][None], (224, 448))[:, 0])
            for k in ['actions', 'proprio']:
                arrays[k].append(torch.from_numpy(d[k].copy()).float())
        with np.load(files[1][0], allow_pickle=False) as d:
            arrays['rgb'].append(tensor_video(d['video'], (224, 448)))
            times = d['times']
        with np.load(files[2][0], allow_pickle=False) as d:
            arrays['raw'].append(tensor_video(d['frames'], (224, 448)))
        with np.load(files[3][0], allow_pickle=False) as d:
            np.testing.assert_array_equal(d['sim_time'], times)
            depths = []
            for cam in ['agentview', 'robot0_eye_in_hand']:
                native = d[cam + '_depth_m_native_raw']
                expected = np.stack([np.asarray(Image.fromarray(frame[::-1, ::-1]).resize((224, 224), Image.Resampling.NEAREST)) for frame in native])
                np.testing.assert_array_equal(expected, d[cam + '_depth_m_224_rot180'])
                depths.append(expected)
            depth = np.concatenate(depths, axis=2)
            assert np.isfinite(depth).all() and (depth > 0).all()
            arrays['depth'].append(torch.from_numpy(depth.copy()).float()[None])
    data = {k: torch.stack(v).cuda() for k, v in arrays.items()}
    inputs = [data[k] for k in ['raw', 'current', 'actions', 'proprio']]
    mask = torch.stack(masks).cuda()[:, :, 1:] if a.regions else None
    if mask is not None:
        assert (mask.sum(dim=(1, 2, 3, 4)) > 0).all()
    def region_mean(error):
        return (error * mask).sum() / (mask.sum() * error.shape[1])
    def components(rgb, depth):
        pix = (rgb[:, :, 1:] - data['rgb'][:, :, 1:]).abs().mean()
        temporal = ((rgb[:, :, 1:] - rgb[:, :, :-1]) -
                    (data['rgb'][:, :, 1:] - data['rgb'][:, :, :-1])).abs().mean()
        dep = (depth[:, :, 1:].log() - data['depth'][:, :, 1:].log()).abs().mean()
        return pix, temporal, dep
    report = {'scope': 'three train windows only; no held-out/physics/control claim', 'seed': a.seed,
              'steps': a.steps, 'lr': a.lr, 'depth_weight': a.depth_weight, 'weight_decay': 1e-4,
              'temporal_weight': .1, 'resolution': [224, 448], 'sample_ids': [r['id'] for r in rows],
              'code_sha256': digest(Path(__file__)), 'bank_sha256': digest(a.bank / 'manifest.json'),
              'wam_sha256': digest(a.wam / 'result.json'), 'geometry_sha256': digest(a.geometry / 'result.json'),
              'depth_role': 'target only; predicted depth conditions RGB decoder; no pretrained depth estimator',
              'methods': {}, 'raw_rgb_mae': float((data['raw'][:, :, 1:] - data['rgb'][:, :, 1:]).abs().mean()),
              'copy_rgb_mae': float((data['current'][:, :, None] - data['rgb'][:, :, 1:]).abs().mean())}
    if a.regions:
        report.update(region_sha256=digest(a.regions / 'result.json'), region_fraction=float(mask.mean()),
                      raw_region_rgb_mae=float(region_mean((data['raw'][:, :, 1:] - data['rgb'][:, :, 1:]).abs())),
                      copy_region_rgb_mae=float(region_mean((data['current'][:, :, None] - data['rgb'][:, :, 1:]).abs())))
    start = time.monotonic()
    modes = [('rgb_only', 0.), ('rgb_depth', a.depth_weight)]
    if a.regions:
        modes.append(('rgb_roi_depth', a.depth_weight))
    for mode, weight in modes:
        torch.manual_seed(a.seed)
        model = DepthRepair().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        history = []
        for step in range(a.steps + 1):
            rgb, depth = model(*inputs)
            pix, temporal, dep = components(rgb, depth)
            dep_objective = region_mean((depth[:, :, 1:].log() - data['depth'][:, :, 1:].log()).abs()) if mode == 'rgb_roi_depth' else dep
            loss = pix + .1 * temporal + weight * dep_objective
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            if step % 25 == 0 or step == a.steps:
                row = {'step': step, 'rgb_mae': float(pix.detach()), 'depth_log_mae': float(dep.detach()), 'temporal_mae': float(temporal.detach()), 'loss': float(loss.detach())}
                if mask is not None:
                    row.update(region_rgb_mae=float(region_mean((rgb[:, :, 1:] - data['rgb'][:, :, 1:]).abs()).detach()),
                               region_depth_log_mae=float(region_mean((depth[:, :, 1:].log() - data['depth'][:, :, 1:].log()).abs()).detach()))
                history.append(row)
                print(json.dumps({'mode': mode, **row}), flush=True)
            if step == a.steps:
                break
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        # Nonzero gradients demonstrate coupling, not correct visible physics.
        grad_shared = torch.autograd.grad(dep_objective, model.encoder.weight, retain_graph=True)[0]
        rgb_to_depth = torch.autograd.grad(pix, model.depth.weight, retain_graph=True)[0]
        coupling = {'depth_loss_to_encoder_norm': float(grad_shared.norm()), 'rgb_loss_to_depth_head_norm': float(rgb_to_depth.norm())}
        assert all(np.isfinite(v) and v > 0 for v in coupling.values()), coupling
        report['methods'][mode] = {'history': history, 'coupling': coupling,
                                  'parameters': sum(p.numel() for p in model.parameters()),
                                  'initial': history[0], 'final': history[-1]}
        np.savez_compressed(a.out / (mode + '_train_predictions.npz'),
                            rgb=rgb.detach().cpu().numpy(), depth_m=depth.detach().cpu().numpy())
        torch.save({'model': model.state_dict(), 'seed': a.seed, 'steps': a.steps, 'mode': mode}, a.out / (mode + '.pt'))
        del rgb, depth, pix, temporal, dep, dep_objective, loss, model, opt
        torch.cuda.empty_cache()
    report['elapsed_seconds'] = time.monotonic() - start
    report['peak_memory_bytes'] = torch.cuda.max_memory_allocated()
    report['status'] = 'COMPLETED_TRAIN_ONLY_DIAGNOSTIC'
    report['learnability_check'] = all(m['final']['rgb_mae'] < m['initial']['rgb_mae'] for m in report['methods'].values()) and report['methods']['rgb_depth']['final']['depth_log_mae'] < report['methods']['rgb_depth']['initial']['depth_log_mae']
    if a.regions:
        m = report['methods']['rgb_roi_depth']
        report['roi_learnability_check'] = m['final']['region_depth_log_mae'] < m['initial']['region_depth_log_mae']
    (a.out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'learnability_check': report['learnability_check']}), flush=True)


if __name__ == '__main__':
    main()
