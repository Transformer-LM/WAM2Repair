"""Geometry/raster agreement prerequisite, never a repair benchmark."""
import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ray_metrics(m, d, f, camera, wrong_flip=False, pixel_offset=.5, axial_depth=True):
    depth = f[camera + '_depth_m']
    seg = f[camera + '_seg']
    K = f[camera + '_K']; T = f[camera + '_camera_to_world']
    h, w = depth.shape
    mask = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
    rows = []
    for v in range(8, h - 8, 12):
        for u in range(8, w - 8, 12):
            gid = int(seg[v, u, 1])
            # Exclusion is purely label-boundary based, not based on residual.
            if int(seg[v, u, 0]) != int(mujoco.mjtObj.mjOBJ_GEOM) or gid < 0:
                continue
            if not np.all(seg[v-1:v+2, u-1:u+2, 1] == gid):
                continue
            vv = h - 1 - v if wrong_flip else v
            vec_cam = np.linalg.solve(K, np.array([u + pixel_offset, vv + pixel_offset, 1.]))
            vec_cam /= np.linalg.norm(vec_cam)
            vec_world = T[:3, :3] @ vec_cam
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(m, d, T[:3, 3].copy(), vec_world, mask, 1, -1, hit)
            z = float(distance * vec_cam[2]) if distance >= 0 and axial_depth else (float(distance) if distance >= 0 else None)
            rows.append({'u': u, 'v': v, 'render_geom': gid, 'ray_geom': int(hit[0]),
                         'render_depth': float(depth[v, u]), 'ray_depth': z,
                         'error_m': abs(z - float(depth[v, u])) if z is not None else None})
    errors = [r['error_m'] for r in rows if r['error_m'] is not None]
    return {'sample_count': len(rows), 'hit_count': len(errors),
            'geom_match_fraction': sum(r['render_geom'] == r['ray_geom'] for r in rows) / max(1, len(rows)),
            'depth_p95_m': float(np.percentile(errors, 95)) if errors else None,
            'depth_max_m': max(errors) if errors else None, 'samples': rows}


def collision_unit():
    # Analytic unit test ONLY; not a LIBERO mesh validation or physical result.
    rows = []
    for separation in [.25, .2, .15]:
        xml = f'<mujoco><worldbody><geom type="sphere" size=".1"/><body pos="{separation} 0 0"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        value = mujoco.mj_geomDistance(m, d, 0, 1, 1., np.zeros(6))
        expected = separation - .2
        np.testing.assert_allclose(value, expected, atol=1e-10, rtol=0)
        rows.append({'center_distance_m': separation, 'expected_signed_gap_m': expected, 'queried_m': value})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expected-source-status', default='PASS_ATOMIC_REPEATABILITY_ONLY')
    p.add_argument('--expected-init-id', type=int, default=4)
    p.add_argument('--required-depth-backend', default=None)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'scope': 'simulation_only geometry/raster prerequisite',
              'code_sha256': sha(__file__), 'source_result_sha256': sha(a.source / 'result.json'),
              'mujoco_version': mujoco.__version__, 'records': [], 'assets': [],
              'thresholds': {'minimum_rays': 100, 'geom_match_fraction': .95, 'depth_p95_m': .005},
              'not_validated': ['independent landmark pixel error', 'crossview occlusion',
                                'LIBERO mesh signed distance accuracy', 'solver tolerance',
                                'full D0 episodes', 'repair improvement']}
    start = time.monotonic()
    try:
        source = json.loads((a.source / 'result.json').read_text())
        assert source['status'] == a.expected_source_status
        if a.required_depth_backend is not None:
            assert source.get('depth_backend') == a.required_depth_backend
        expected = {
            ('libero_spatial', a.expected_init_id, tick, f'libero_spatial_init{a.expected_init_id}_tick{tick}.npz')
            for tick in range(3)
        } | {
            ('libero_object', a.expected_init_id, tick, f'libero_object_init{a.expected_init_id}_tick{tick}.npz')
            for tick in range(3)
        }
        observed = {(r['suite'], r['init_id'], r['tick'], r['file']) for r in source['snapshots']}
        assert observed == expected and len(source['snapshots']) == len(expected)
        report['source_contract'] = {'expected_status': a.expected_source_status,
                                     'expected_init_id': a.expected_init_id,
                                     'required_depth_backend': a.required_depth_backend}
        report['analytic_sphere_unit_only'] = collision_unit()
        for inventory in source['inventory']:
            if 'model_xml_sha256' not in inventory:
                continue
            suite = inventory['suite']; xml = a.source / (suite + '_model.xml')
            assert sha(xml) == inventory['model_xml_sha256']
            # Existing asset paths are read-only; all hashes stored, no assets modified.
            for node in ET.parse(xml).getroot().findall('./asset/*'):
                if 'file' in node.attrib:
                    asset = Path(node.attrib['file'])
                    assert asset.is_absolute() and asset.is_file(), str(asset)
                    report['assets'].append({'path': str(asset), 'sha256': sha(asset)})
            m = mujoco.MjModel.from_xml_path(str(xml)); d = mujoco.MjData(m)
            for row in source['snapshots']:
                if row['suite'] != suite:
                    continue
                path = a.source / row['file']
                assert Path(row['file']).name == row['file'] and path.is_file()
                assert sha(path) == row['sha256']
                with np.load(path, allow_pickle=False) as f:
                    state = f['state']; assert len(state) == 1 + m.nq + m.nv
                    d.time = state[0]; d.qpos[:] = state[1:1+m.nq]; d.qvel[:] = state[1+m.nq:]
                    mujoco.mj_forward(m, d)
                    # Compiled XML roundtrip may round numeric values: do not assert exact.
                    np.testing.assert_allclose(d.geom_xpos, f['geom_xpos'], atol=1e-5, rtol=0)
                    np.testing.assert_allclose(d.geom_xmat, f['geom_xmat'], atol=1e-5, rtol=0)
                    for camera in ['agentview', 'robot0_eye_in_hand']:
                        metrics = ray_metrics(m, d, f, camera)
                        bad = ray_metrics(m, d, f, camera, wrong_flip=True)
                        passed = (metrics['sample_count'] >= 100 and metrics['hit_count'] == metrics['sample_count']
                                  and metrics['geom_match_fraction'] >= .95 and metrics['depth_p95_m'] <= .005)
                        # The negative control must be rejected, otherwise a vertically flipped
                        # coordinate convention could pass while this check claims sensitivity.
                        wrong_rejected = (bad['hit_count'] != bad['sample_count'] or
                                          bad['geom_match_fraction'] < .95 or
                                          bad['depth_p95_m'] is None or bad['depth_p95_m'] > .005)
                        passed = passed and wrong_rejected
                        report['records'].append({'file': row['file'], 'camera': camera, 'passed': passed,
                                                  'wrong_vertical_flip_rejected': wrong_rejected,
                                                  'correct': metrics, 'wrong_vertical_flip_control': bad})
                        print(json.dumps({'file': row['file'], 'camera': camera, 'passed': passed,
                            'count': metrics['sample_count'], 'match': metrics['geom_match_fraction'],
                            'p95_m': metrics['depth_p95_m'], 'wrong_flip_match': bad['geom_match_fraction']}), flush=True)
        assert len(report['records']) == 12
        report['status'] = 'PASS_RAY_RASTER_ONLY' if all(r['passed'] for r in report['records']) else 'FAIL_RAY_RASTER_GATE'
    except Exception as exc:
        report['status'] = 'ERROR'; report['error'] = repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        (a.out / 'result.json').write_text(json.dumps(report, indent=2))
        print('RESULT', report['status'], flush=True)
    if report['status'] != 'PASS_RAY_RASTER_ONLY':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
