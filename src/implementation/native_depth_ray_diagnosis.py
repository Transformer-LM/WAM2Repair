"""Diagnostic-only native renderer depth/segmentation versus mj_ray check."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from geometry_ray_gate import ray_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {'exact': bool(np.array_equal(a, b)),
            'mismatched_elements': int(np.count_nonzero(a != b)),
            'max_abs': float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))}


def native_images(renderer):
    renderer.enable_depth_rendering()
    try:
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    try:
        segmentation = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    return depth, segmentation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING',
              'scope': 'native renderer versus mj_ray diagnosis only; no B0/D0 approval',
              'code_sha256': sha(__file__), 'source_result_sha256': sha(a.source / 'result.json'),
              'mujoco_version': mujoco.__version__}
    start = time.monotonic()
    try:
        source = json.loads((a.source / 'result.json').read_text())
        assert source['status'] == 'PASS_ATOMIC_REPEATABILITY_ONLY'
        inv = next(x for x in source['inventory'] if x['suite'] == 'libero_object')
        xml = a.source / 'libero_object_model.xml'
        assert xml.is_file() and sha(xml) == inv['model_xml_sha256']
        model, data = mujoco.MjModel.from_xml_path(str(xml)), None
        data = mujoco.MjData(model)
        report['recorded_model_xml_sha256'] = sha(xml)
        renderer = ray_group_renderer = None
        try:
            renderer = mujoco.Renderer(model, height=256, width=256)
            ray_group_renderer = mujoco.Renderer(model, height=256, width=256)
            ray_scene_option = mujoco.MjvOption()
            ray_scene_option.geomgroup[:] = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
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
                        depth, seg = native_images(renderer)
                        ray_group_renderer.update_scene(data, camera=camera, scene_option=ray_scene_option)
                        group_depth, group_seg = native_images(ray_group_renderer)
                        # Native Renderer encodes segmentation as (object id, object type),
                        # whereas robosuite snapshots use (object type, object id). Record the
                        # channel-swapped diagnostic separately; do not replace any gate input.
                        seg_swapped = seg[..., [1, 0]]
                        native = {camera + '_depth_m': depth, camera + '_seg': seg,
                                  camera + '_K': f[camera + '_K'],
                                  camera + '_camera_to_world': f[camera + '_camera_to_world']}
                        flipped = {camera + '_depth_m': depth[::-1], camera + '_seg': seg[::-1],
                                   camera + '_K': f[camera + '_K'],
                                   camera + '_camera_to_world': f[camera + '_camera_to_world']}
                        swapped = {camera + '_depth_m': depth, camera + '_seg': seg_swapped,
                                   camera + '_K': f[camera + '_K'],
                                   camera + '_camera_to_world': f[camera + '_camera_to_world']}
                        swapped_flipped = {camera + '_depth_m': depth[::-1],
                                           camera + '_seg': seg_swapped[::-1],
                                           camera + '_K': f[camera + '_K'],
                                           camera + '_camera_to_world': f[camera + '_camera_to_world']}
                        group_swapped = {camera + '_depth_m': group_depth,
                                         camera + '_seg': group_seg[..., [1, 0]],
                                         camera + '_K': f[camera + '_K'],
                                         camera + '_camera_to_world': f[camera + '_camera_to_world']}
                        group_swapped_flipped = {camera + '_depth_m': group_depth[::-1],
                                                 camera + '_seg': group_seg[..., [1, 0]][::-1],
                                                 camera + '_K': f[camera + '_K'],
                                                 camera + '_camera_to_world': f[camera + '_camera_to_world']}
                        cameras[camera] = {
                            'native_vs_saved_seg_top_down': compare(seg, f[camera + '_seg']),
                            'native_vs_saved_seg_bottom_up': compare(seg[::-1], f[camera + '_seg']),
                            'native_channel_swapped_vs_saved_seg_top_down': compare(seg_swapped, f[camera + '_seg']),
                            'native_channel_swapped_vs_saved_seg_bottom_up': compare(seg_swapped[::-1], f[camera + '_seg']),
                            'native_top_down_ray_metrics': ray_metrics(model, data, native, camera),
                            'native_bottom_up_ray_metrics': ray_metrics(model, data, flipped, camera),
                            'native_channel_swapped_top_down_ray_metrics': ray_metrics(model, data, swapped, camera),
                            'native_channel_swapped_bottom_up_ray_metrics': ray_metrics(model, data, swapped_flipped, camera),
                            'native_ray_geomgroup_vs_saved_seg_top_down': compare(group_seg[..., [1, 0]], f[camera + '_seg']),
                            'native_ray_geomgroup_vs_saved_seg_bottom_up': compare(group_seg[..., [1, 0]][::-1], f[camera + '_seg']),
                            'native_ray_geomgroup_top_down_ray_metrics': ray_metrics(model, data, group_swapped, camera),
                            'native_ray_geomgroup_bottom_up_ray_metrics': ray_metrics(model, data, group_swapped_flipped, camera),
                        }
                    report.setdefault('records', []).append({'file': row['file'], 'cameras': cameras})
        finally:
            if ray_group_renderer is not None:
                ray_group_renderer.close()
            if renderer is not None:
                renderer.close()
        report['status'] = 'COMPLETED_DIAGNOSIS_ONLY'
        report['limitations'] = ['native renderer agreement is not independent physical ground truth',
                                 'does not replace stored-depth B0 or alter its frozen threshold',
                                 'does not authorize D0, repair training, WAM, or VLA claims']
    except Exception as exc:
        report['status'] = 'ERROR'; report['error'] = repr(exc); raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        (a.out / 'result.json').write_text(json.dumps(report, indent=2))
        print('RESULT', report['status'], flush=True)


if __name__ == '__main__':
    main()
