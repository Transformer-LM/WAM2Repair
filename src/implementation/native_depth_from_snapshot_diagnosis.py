"""Render an atomic snapshot with its recorded XML, without LIBERO reconstruction.

This is a renderer-output diagnosis only: it never changes the B0 protocol,
threshold, labels, or authorization to collect/train/evaluate.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(a, b):
    absolute = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return {'mae_m': float(np.mean(absolute)), 'p95_abs_m': float(np.percentile(absolute, 95)),
            'max_abs_m': float(np.max(absolute)), 'exact': bool(np.array_equal(a, b)),
            'mismatched_elements': int(np.count_nonzero(np.asarray(a) != np.asarray(b)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'scope': 'recorded-XML native renderer diagnosis only; no B0 approval',
              'code_sha256': sha(__file__), 'source_result_sha256': sha(a.source / 'result.json'),
              'mujoco_version': mujoco.__version__}
    start = time.monotonic()
    try:
        source = json.loads((a.source / 'result.json').read_text())
        assert source['status'] == 'PASS_ATOMIC_REPEATABILITY_ONLY'
        inv = next(x for x in source['inventory'] if x['suite'] == 'libero_object')
        xml = a.source / 'libero_object_model.xml'
        assert xml.is_file() and sha(xml) == inv['model_xml_sha256']
        model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model)
        report['recorded_model_xml_sha256'] = sha(xml)
        renderer = mujoco.Renderer(model, height=256, width=256)
        try:
            for row in [x for x in source['snapshots'] if x['suite'] == 'libero_object']:
                path = a.source / row['file']; assert path.is_file() and sha(path) == row['sha256']
                with np.load(path, allow_pickle=False) as f:
                    state = f['state']; assert len(state) == 1 + model.nq + model.nv
                    data.time = state[0]; data.qpos[:] = state[1:1 + model.nq]
                    data.qvel[:] = state[1 + model.nq:]; mujoco.mj_forward(model, data)
                    np.testing.assert_array_equal(np.asarray(data.time), f['time'])
                    np.testing.assert_allclose(data.geom_xpos, f['geom_xpos'], atol=1e-5, rtol=0)
                    np.testing.assert_allclose(data.geom_xmat, f['geom_xmat'], atol=1e-5, rtol=0)
                    cameras = {}
                    for camera in ('agentview', 'robot0_eye_in_hand'):
                        renderer.update_scene(data, camera=camera)
                        # The installed MuJoCo Python binding selects depth mode on the renderer
                        # rather than via a render(depth=...) keyword.
                        renderer.enable_depth_rendering()
                        try:
                            depth = renderer.render().copy()
                        finally:
                            renderer.disable_depth_rendering()
                        saved = f[camera + '_depth_m']
                        cameras[camera] = {'native_shape': list(depth.shape), 'stored_shape': list(saved.shape),
                                           'native_vs_stored_top_down': summary(depth, saved),
                                           'native_vs_stored_bottom_up': summary(depth[::-1], saved)}
                    report.setdefault('records', []).append({'file': row['file'], 'cameras': cameras})
        finally:
            renderer.close()
        report['status'] = 'COMPLETED_DIAGNOSIS_ONLY'
        report['limitations'] = ['native-vs-stored renderer outputs are not physical ground truth',
                                 'no post-hoc orientation or renderer choice may unlock B0',
                                 'does not authorize D0, training, WAM, or VLA claims']
    except Exception as exc:
        report['status'] = 'ERROR'; report['error'] = repr(exc); raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        (a.out / 'result.json').write_text(json.dumps(report, indent=2))
        print('RESULT', report['status'], flush=True)


if __name__ == '__main__':
    main()
