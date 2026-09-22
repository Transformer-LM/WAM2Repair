"""CPU-only diagnostic: compare robosuite depth with MuJoCo's native renderer.

This records a renderer-path difference only.  It never changes B0's gate,
labels, threshold, or data-collection authorization.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(a, b):
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    absolute = np.abs(delta)
    return {
        'exact': bool(np.array_equal(a, b)),
        'mismatched_elements': int(np.count_nonzero(np.asarray(a) != np.asarray(b))),
        'mean_signed_m': float(np.mean(delta)),
        'mae_m': float(np.mean(absolute)),
        'p95_abs_m': float(np.percentile(absolute, 95)),
        'max_abs_m': float(np.max(absolute)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        'status': 'RUNNING',
        'scope': 'native-vs-robosuite renderer-path diagnosis only; no B0 approval',
        'code_sha256': sha(__file__),
        'source_result_sha256': sha(args.source / 'result.json'),
    }
    started = time.monotonic()
    try:
        import mujoco
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from atomic_snapshot_probe import snapshot

        source = json.loads((args.source / 'result.json').read_text())
        assert source['status'] == 'PASS_ATOMIC_REPEATABILITY_ONLY'
        inventory = next(row for row in source['inventory'] if row['suite'] == 'libero_object')
        suite = benchmark.get_benchmark_dict()['libero_object']()
        task = suite.get_task(inventory['task_id'])
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        assert sha(bddl) == inventory['bddl_sha256']
        source_xml = args.source / 'libero_object_model.xml'
        assert source_xml.is_file() and sha(source_xml) == inventory['model_xml_sha256']
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                 camera_widths=256, camera_depths=True,
                                 render_gpu_device_id=-1)
        try:
            raw_model, raw_data = env.sim.model._model, env.sim.data._data
            reconstructed_model_xml_sha256 = hashlib.sha256(
                env.sim.model.get_xml().encode('utf-8')).hexdigest()
            assert reconstructed_model_xml_sha256 == inventory['model_xml_sha256']
            report['mujoco_version'] = mujoco.__version__
            report['binding'] = {
                'model_type': str(type(raw_model)), 'data_type': str(type(raw_data)),
            }
            report['model_xml_sha256'] = {
                'source_recorded': inventory['model_xml_sha256'],
                'source_file': sha(source_xml),
                'reconstructed': reconstructed_model_xml_sha256,
            }
            renderer = mujoco.Renderer(raw_model, height=256, width=256)
            try:
                for row in [x for x in source['snapshots'] if x['suite'] == 'libero_object']:
                    path = args.source / row['file']
                    assert path.is_file() and sha(path) == row['sha256']
                    with np.load(path, allow_pickle=False) as saved:
                        env.sim.set_state_from_flattened(saved['state'])
                        env.sim.forward()
                        fresh = snapshot(env.sim, 256)
                        for key in ('state', 'time', 'body_xpos', 'body_xmat', 'geom_xpos', 'geom_xmat'):
                            np.testing.assert_array_equal(saved[key], fresh[key], err_msg=key)
                        cameras = {}
                        for camera in ('agentview', 'robot0_eye_in_hand'):
                            renderer.update_scene(raw_data, camera=camera)
                            # The native Python API documents this as metric depth; preserve its
                            # orientation as rendered and compare both orientations diagnostically.
                            native = renderer.render(depth=True).copy()
                            stored = saved[camera + '_depth_m']
                            cameras[camera] = {
                                'native_shape': list(native.shape),
                                'stored_shape': list(stored.shape),
                                'native_vs_stored_top_down': summary(native, stored),
                                'native_vs_stored_bottom_up': summary(native[::-1], stored),
                            }
                        report.setdefault('records', []).append({'file': row['file'], 'cameras': cameras})
            finally:
                renderer.close()
        finally:
            env.close()
        report['status'] = 'COMPLETED_DIAGNOSIS_ONLY'
        report['limitations'] = [
            'does not identify an independent physical ground truth',
            'does not alter the frozen B0 ray/raster gate or threshold',
            'does not approve D0, repair training, WAM, or VLA claims',
        ]
    except Exception as exc:
        report['status'] = 'ERROR'
        report['error'] = repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        (args.out / 'result.json').write_text(json.dumps(report, indent=2))
        print('RESULT', report['status'], flush=True)


if __name__ == '__main__':
    main()
