"""CPU train-only geometry-target export. Does not certify predicted-video physics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rotation_check(r):
    assert np.isfinite(r).all()
    np.testing.assert_allclose(r.swapaxes(-1, -2) @ r,
                               np.broadcast_to(np.eye(3), r.shape), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(r), 1, atol=1e-5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    queue = a.root / 'official_grid_20260917'
    jobs = json.loads((queue / 'manifest.json').read_text())['jobs']
    state = json.loads((queue / 'queue_state.json').read_text())
    assert state.get('status') == 'ALL_COMPLETED'
    bank = a.root / 'bank_official_full_20260917'
    bank_manifest = bank / 'manifest.json'
    rows = json.loads(bank_manifest.read_text())['records']
    assert len({j['id'] for j in jobs}) == len(jobs)
    assert len({r['id'] for r in rows}) == len(rows)
    assert sum(j['split'] == 'train' for j in jobs) == 8
    assert sum(r['split'] == 'train' for r in rows) == 77
    groups = {s: {r['group'] for r in rows if r['split'] == s}
              for s in ['train', 'validation', 'test']}
    assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test']
                or groups['validation'] & groups['test'])
    cameras = ['agentview', 'robot0_eye_in_hand']
    report = {'schema': 'w2r-geometry-train-gate-v1', 'scope': 'train-only factual supervision, not repair results',
              'bank_manifest_sha256': digest(bank_manifest), 'script_sha256': digest(Path(__file__)),
              'records': [], 'sources': [], 'limitations': [
                  'No depth estimator or geometric repair model is trained by this script.',
                  'Zero contact_min_distance is not a measured separation; never use as SDF.',
                  'Contact is instantaneous simulator contact, not grasp or force truth.',
                  'Camera projection convention requires a separate rendered-landmark witness.',
                  'All exported future geometry is target-only, forbidden as inference input.',
                  'No mesh/SDF penetration target, segmentation, or slip target is certified.']}
    for job in jobs:
        if job['split'] != 'train':
            continue
        assert state['jobs'][job['id']]['status'] == 'completed'
        evidence = state['jobs'][job['id']]['evidence']
        source = Path(evidence['trajectory'])
        assert source.resolve().is_relative_to(a.root.resolve())
        source_hash = digest(source)
        assert source_hash == evidence['trajectory_sha256']
        prefix = f"bank_official_t{job['task']}_e{job['episode']}_{job['action_mode']}_20260917"
        selected = [r for r in rows if r['id'].startswith(prefix + '_window_')]
        assert selected and all(r['split'] == 'train' for r in selected)
        with np.load(source, allow_pickle=False) as archive:
            keys = ['sim_time', 'actions', 'object_pos', 'object_rot', 'eef_pos', 'eef_rot',
                    'contact_matrix', 'contact_min_distance', 'eef_relative_pos', 'eef_relative_rot']
            keys += [c + suffix for c in cameras for suffix in ['_rgb', '_depth_m', '_K', '_camera_to_world']]
            d = {k: archive[k] for k in keys}
        for k, value in d.items():
            assert np.isfinite(value).all(), k
        n = len(d['sim_time'])
        assert n == len(d['actions']) + 1
        np.testing.assert_allclose(np.diff(d['sim_time']), .05, atol=1e-7)
        rotation_check(d['object_rot']); rotation_check(d['eef_rot'])
        np.testing.assert_allclose(d['eef_relative_pos'],
            np.einsum('tni,tij->tnj', d['object_pos'] - d['eef_pos'][:, None], d['eef_rot']), atol=1e-6)
        np.testing.assert_allclose(d['eef_relative_rot'],
            np.einsum('tji,tnjk->tnik', d['eef_rot'], d['object_rot']), atol=1e-6)
        contact = d['contact_matrix']
        assert np.array_equal(contact, contact.swapaxes(-1, -2))
        assert not np.diagonal(contact, axis1=-2, axis2=-1).any()
        camera_stats = {}
        for cam in cameras:
            rgb, depth = d[cam + '_rgb'], d[cam + '_depth_m']
            if depth.ndim == 4:
                assert depth.shape[-1] == 1
                depth = depth[..., 0]
            assert depth.shape == rgb.shape[:-1] and len(depth) == n
            assert (depth > 0).all()
            d[cam + '_depth_m'] = depth
            intrinsics = d[cam + '_K']
            assert intrinsics.shape == (n, 3, 3)
            assert (intrinsics[:, 0, 0] > 0).all() and (intrinsics[:, 1, 1] > 0).all()
            transform = d[cam + '_camera_to_world']
            rotation_check(transform[..., :3, :3])
            np.testing.assert_allclose(transform[..., 3, :], np.broadcast_to([0, 0, 0, 1], (n, 4)), atol=1e-6)
            camera_stats[cam] = {'depth_min_m': float(depth.min()), 'depth_max_m': float(depth.max()),
                                 'native_shape': list(depth.shape), 'K_shape': list(d[cam + '_K'].shape)}
        for r in selected:
            idx = r['start'] + np.array([0, 4, 8, 12, 16])
            ip, tp = bank / r['input'], bank / r['target']
            assert digest(ip) == r['input_sha256'] and digest(tp) == r['target_sha256']
            with np.load(ip, allow_pickle=False) as inputs, np.load(tp, allow_pickle=False) as target:
                assert set(inputs.files) == {'current_rgb', 'proprio', 'actions'}
                np.testing.assert_array_equal(inputs['actions'], d['actions'][r['start']:r['start'] + 16])
                np.testing.assert_allclose(target['times'], d['sim_time'][idx])
                for key in ['object_pos', 'object_rot', 'eef_pos', 'eef_rot']:
                    np.testing.assert_array_equal(target[key], d[key][idx])
                np.testing.assert_array_equal(target['contact'], contact[idx])
                rgb_frames = np.stack([np.concatenate([
                    np.asarray(Image.fromarray(d[c + '_rgb'][t, ::-1, ::-1]).resize((224, 224), Image.Resampling.BILINEAR))
                    for c in cameras], axis=1) for t in idx])
                np.testing.assert_array_equal(target['video'], rgb_frames)
                np.testing.assert_array_equal(inputs['current_rgb'], rgb_frames[0])
            payload = {key: d[key][idx] for key in ['object_pos', 'object_rot', 'eef_pos', 'eef_rot',
                       'eef_relative_pos', 'eef_relative_rot', 'contact_matrix', 'sim_time']}
            for cam in cameras:
                payload[cam + '_depth_m_native_raw'] = d[cam + '_depth_m'][idx]
                payload[cam + '_K_native_raw'] = d[cam + '_K'][idx]
                payload[cam + '_camera_to_world'] = d[cam + '_camera_to_world'][idx]
                # Nearest keeps metric depth at discontinuities; not the RGB bilinear kernel.
                payload[cam + '_depth_m_224_rot180'] = np.stack([
                    np.asarray(Image.fromarray(d[cam + '_depth_m'][t, ::-1, ::-1]).resize((224, 224), Image.Resampling.NEAREST))
                    for t in idx])
            dest = a.out / (r['id'] + '_geometry_target.npz')
            np.savez_compressed(dest, **payload)
            report['records'].append({'id': r['id'], 'group': r['group'], 'split': 'train',
                                      'source_sha256': source_hash, 'target': dest.name, 'sha256': digest(dest)})
        report['sources'].append({'job': job['id'], 'source': str(source), 'sha256': source_hash,
            'frames': n, 'windows': len(selected), 'gripper_contact_frames': int(contact[:, :-1, -1].any(axis=1).sum()),
            'minimum_recorded_contact_distance': float(d['contact_min_distance'].min()), 'cameras': camera_stats})
        print(json.dumps({'job': job['id'], 'windows': len(selected), 'status': 'VALIDATED'}), flush=True)
    assert len(report['records']) == 77
    assert len({r['id'] for r in report['records']}) == 77
    assert {r['id'] for r in report['records']} == {r['id'] for r in rows if r['split'] == 'train'}
    report['status'] = 'PASS_DATA_ALIGNMENT_ONLY'
    report['windows'] = len(report['records'])
    (a.out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'windows': report['windows']}), flush=True)


if __name__ == '__main__':
    main()
