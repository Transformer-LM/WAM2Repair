"""Validate factual alignment only; no training or WAM performance claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bank', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bank / 'manifest.json').read_text())
    report = {'scope': 'factual alignment only', 'records': []}
    reference = None
    for record in manifest['records']:
        path = args.bank / record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as d:
            n = len(d['actions'])
            assert n == manifest['args']['horizon']
            assert len(d['qpos']) == n + 1
            assert len(d['rewards']) == n
            assert len(d['success']) == n + 1
            assert np.all(np.diff(d['sim_time']) > 0)
            for key in d.files:
                if d[key].dtype.kind in 'fc':
                    assert np.isfinite(d[key]).all(), key
            for key in ['object_rot', 'eef_rot', 'eef_relative_rot']:
                r = d[key]
                np.testing.assert_allclose(r.swapaxes(-1, -2) @ r, np.broadcast_to(np.eye(3), r.shape), atol=1e-6)
                np.testing.assert_allclose(np.linalg.det(r), 1., atol=1e-6)
            rel = np.einsum('tni,tij->tnj', d['object_pos']-d['eef_pos'][:, None], d['eef_rot'])
            np.testing.assert_allclose(rel, d['eef_relative_pos'], atol=1e-7)
            rel_rot = np.einsum('tji,tnjk->tnik', d['eef_rot'], d['object_rot'])
            np.testing.assert_allclose(rel_rot, d['eef_relative_rot'], atol=1e-7)
            c = d['contact_matrix']
            assert np.array_equal(c, c.swapaxes(-1, -2))
            assert not np.diagonal(c, axis1=-2, axis2=-1).any()
            assert np.all(d['slip_label'] == -1)
            for camera in ['agentview', 'robot0_eye_in_hand']:
                assert d[camera+'_rgb'].shape == (n+1, 224, 224, 3)
                assert d[camera+'_rgb'].dtype == np.uint8
                assert np.std(d[camera+'_rgb']) > 0
                assert np.all(d[camera+'_depth_m'] > 0)
            current = {k:d[k][0].copy() for k in ['qpos', 'qvel', 'object_pos', 'eef_pos', 'sim_time', 'agentview_rgb']}
            if reference is None:
                reference = current
            else:
                for key in reference:
                    np.testing.assert_allclose(current[key], reference[key], atol=1e-6, rtol=0)
            report['records'].append({'file': record['file'], 'frames':n+1,
                'gripper_contact_frames': int(c[:, :-1, -1].any(axis=1).sum()),
                'success_after_chunk':bool(d['success'][-1])})
    assert len(report['records']) == manifest['args']['candidates']
    report.update(status='PASS', limitations=['scripted candidates, not VLA candidates',
        'slip unknown; no positive contact coverage established',
        'no WAM futures, no training, no ranking evidence',
        'sanity-only bank, not train/validation/test'])
    target = args.bank / 'alignment_validation.json'
    with target.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
