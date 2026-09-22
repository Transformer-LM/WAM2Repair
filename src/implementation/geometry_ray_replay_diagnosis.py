"""Diagnose a failed ray/raster gate in the original LIBERO environment."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'scope': 'diagnosis of failed object-agentview ray/raster gate; no label approval',
              'code_sha256': sha(__file__), 'source_result_sha256': sha(a.source / 'result.json')}
    start = time.monotonic()
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from atomic_snapshot_probe import snapshot
        from geometry_ray_gate import ray_metrics
        source = json.loads((a.source / 'result.json').read_text())
        inv = next(x for x in source['inventory'] if x['suite'] == 'libero_object')
        suite = benchmark.get_benchmark_dict()['libero_object'](); task = suite.get_task(inv['task_id'])
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        assert sha(bddl) == inv['bddl_sha256']
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                 camera_depths=True, render_gpu_device_id=-1)
        try:
            # robosuite wraps official mujoco structs; mj_ray only accepts the underlying structs.
            raw_model, raw_data = env.sim.model._model, env.sim.data._data
            report['ray_binding'] = {'wrapper_model_type': str(type(env.sim.model)), 'wrapper_data_type': str(type(env.sim.data)),
                                     'mujoco_model_type': str(type(raw_model)), 'mujoco_data_type': str(type(raw_data))}
            assert env.sim.model.ngeom == inv['ngeom']
            assert list(env.sim.model.geom_names) == inv['geom_names']
            assert list(env.sim.model.body_names) == inv['body_names']
            for row in [x for x in source['snapshots'] if x['suite'] == 'libero_object']:
                path = a.source / row['file']; assert path.is_file() and sha(path) == row['sha256']
                with np.load(path, allow_pickle=False) as f:
                    env.sim.set_state_from_flattened(f['state']); env.sim.forward()
                    fresh = snapshot(env.sim, 256)
                    for key in ['state', 'time', 'body_xpos', 'body_xmat', 'geom_xpos', 'geom_xmat']:
                        np.testing.assert_array_equal(f[key], fresh[key], err_msg=key)
                    camera_rows = {}
                    for camera in ['agentview', 'robot0_eye_in_hand']:
                        comparisons = {}
                        for suffix in ['K', 'camera_to_world', 'rgb', 'depth_m', 'seg']:
                            saved, current = f[camera + '_' + suffix], fresh[camera + '_' + suffix]
                            comparisons[suffix] = {'exact': bool(np.array_equal(saved, current)),
                                                   'max_abs': max_abs(saved, current),
                                                   'mismatched_elements': int(np.count_nonzero(np.asarray(saved) != np.asarray(current)))}
                        # State/geometry were checked exact above. Record renderer differences instead of
                        # aborting at the first RGB mismatch, so this remains a diagnosis rather than a gate pass.
                        camera_rows[camera] = {
                            'saved_vs_current': comparisons,
                            'original_environment_ray_metrics': ray_metrics(raw_model, raw_data, f, camera),
                            # Convention probes are diagnostic only. The original gate remains the default
                            # pixel-center=.5, axial-depth formulation and its threshold is not changed here.
                            'ray_convention_probes': {
                                'pixel_corner_axial_depth': ray_metrics(raw_model, raw_data, f, camera, pixel_offset=0., axial_depth=True),
                                'pixel_center_ray_distance': ray_metrics(raw_model, raw_data, f, camera, pixel_offset=.5, axial_depth=False),
                                'pixel_corner_ray_distance': ray_metrics(raw_model, raw_data, f, camera, pixel_offset=0., axial_depth=False),
                            },
                        }
                    report.setdefault('records', []).append({'file': row['file'], 'camera_rows': camera_rows})
                    print(json.dumps({'file': row['file'], 'agentview': camera_rows['agentview']}), flush=True)
        finally:
            env.close()
        report['status'] = 'COMPLETED_DIAGNOSIS_ONLY'
        report['limitations'] = ['cross-process renderer differences are recorded, not waived',
                                 'does not approve B0 projection/collision labels or D0 collection',
                                 'does not evaluate WAM, repair, ranking, or VLA benefit']
    except Exception as exc:
        report['status'] = 'ERROR'; report['error'] = repr(exc); raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        (a.out / 'result.json').write_text(json.dumps(report, indent=2))
        print('RESULT', report['status'], flush=True)


if __name__ == '__main__':
    main()
