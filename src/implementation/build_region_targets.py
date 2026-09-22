"""Replay fixed train actions and certify RGB alignment before exporting geom masks."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from geometry_supervision_gate import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    bank = a.root / 'bank_official_full_20260917'
    rows = [r for r in json.loads((bank / 'manifest.json').read_text())['records'] if r['split'] == 'train'][:3]
    gate = json.loads((a.root / 'geometry_train_gate_20260917T094500Z/result.json').read_text())
    assert gate['status'] == 'PASS_DATA_ALIGNMENT_ONLY'
    assert gate['bank_manifest_sha256'] == digest(bank / 'manifest.json')
    selected = {r['id']: r for r in gate['records']}
    assert all(selected[r['id']]['split'] == 'train' and selected[r['id']]['group'] == r['group'] for r in rows)
    hashes = {selected[r['id']]['source_sha256'] for r in rows}
    assert len(rows) == 3 and len(hashes) == 1
    src = next(s for s in gate['sources'] if s['sha256'] in hashes)
    path = Path(src['source']); assert digest(path) == src['sha256']
    meta = json.loads(path.with_name('result.json').read_text())
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in ['actions', 'qpos', 'qvel', 'sim_time', 'agentview_rgb', 'robot0_eye_in_hand_rgb']}
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[meta['suite']](); task = suite.get_task(meta['task'])
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
                            camera_heights=256, camera_widths=256, camera_depths=True,
                            camera_segmentations='element', render_gpu_device_id=-1)
    required = sorted({int(r['start'] + offset) for r in rows for offset in [0, 4, 8, 12, 16]})
    frames = {}; witnesses = []
    try:
        np.random.seed(meta['seed']); env.seed(meta['seed']); env.reset()
        obs = env.set_init_state(suite.get_task_init_states(meta['task'])[meta['episode']])
        for _ in range(10):
            obs, *_ = env.step([0.] * 6 + [-1.])
        model = env.sim.model
        object_names = sorted(set(env.env.objects_dict) - set(env.env.fixtures_dict))
        roots = {int(env.env.obj_body_id[n]) for n in object_names if n in env.env.obj_body_id}
        geom_labels = np.zeros(model.ngeom, dtype=np.uint8)
        for gid in range(model.ngeom):
            bid = int(model.geom_bodyid[gid]); ancestry = []
            while bid:
                ancestry.append(bid); bid = int(model.body_parentid[bid])
            names = [model.body_id2name(b) or '' for b in ancestry]
            if any(n.startswith(('robot0_', 'gripper0_')) for n in names):
                geom_labels[gid] = 1
            elif roots.intersection(ancestry):
                geom_labels[gid] = 2
        assert (geom_labels == 1).any() and (geom_labels == 2).any()
        for t in range(max(required) + 1):
            if t in required:
                np.testing.assert_allclose(env.sim.data.qpos, data['qpos'][t], rtol=0, atol=1e-8)
                np.testing.assert_allclose(env.sim.data.qvel, data['qvel'][t], rtol=0, atol=1e-8)
                np.testing.assert_allclose(env.sim.data.time, data['sim_time'][t], rtol=0, atol=1e-8)
                paired = []
                for cam in ['agentview', 'robot0_eye_in_hand']:
                    np.testing.assert_array_equal(obs[cam + '_image'], data[cam + '_rgb'][t])
                    # Use the official element sensor in the SAME observation update,
                    # not a new post-step render (observations may be cached).
                    seg = obs[cam + '_segmentation_element'][..., 0]
                    valid = (seg >= 0) & (seg < model.ngeom)
                    mask = np.zeros((256, 256), dtype=np.uint8)
                    mask[valid] = geom_labels[seg[valid]]
                    paired.append(np.asarray(Image.fromarray(mask[::-1, ::-1]).resize((224, 224), Image.Resampling.NEAREST)))
                    witnesses.append({'frame': t, 'camera': cam, 'rgb_max_error': 0,
                                      'robot_pixels': int((mask == 1).sum()), 'object_pixels': int((mask == 2).sum())})
                frames[t] = np.concatenate(paired, axis=1)
            if t < max(required):
                obs, *_ = env.step(data['actions'][t].tolist())
        report = {'status': 'PASS_REPLAY_ALIGNED_MASKS', 'scope': 'train-only visible geom segmentation targets; not physical error labels',
                  'source_sha256': src['sha256'], 'source_metadata_sha256': digest(path.with_name('result.json')),
                  'bank_sha256': digest(bank / 'manifest.json'), 'code_sha256': digest(Path(__file__)),
                  'segmentation_source': 'robosuite element observation sensor, same update as RGB; no post-step render',
                  'seed': meta['seed'], 'object_names': object_names, 'labels': {'background': 0, 'robot': 1, 'objects': 2},
                  'witnesses': witnesses, 'records': []}
        for r in rows:
            mask = np.stack([frames[r['start'] + off] for off in [0, 4, 8, 12, 16]])
            assert (mask[1:] == 1).any() and (mask[1:] == 2).any()
            dest = a.out / (r['id'] + '_region_target.npz')
            np.savez_compressed(dest, labels=mask)
            report['records'].append({'id': r['id'], 'split': 'train', 'target': dest.name, 'sha256': digest(dest),
                                      'region_fraction': float((mask[1:] > 0).mean())})
        (a.out / 'result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
