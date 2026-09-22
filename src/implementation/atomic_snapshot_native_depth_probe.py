"""Fresh CPU atomic snapshots with robosuite RGB/seg and native metric depth.

This is a B0 input prerequisite only, not D0 collection or a repair result.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np


RAY_GEOMGROUP = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_segmentation(renderer):
    renderer.enable_segmentation_rendering()
    try:
        return renderer.render().copy()[..., [1, 0]]
    finally:
        renderer.disable_segmentation_rendering()


def snapshot(sim, default_renderer, grouped_renderer, scene_option, size):
    sim.forward()
    out = {'state': sim.get_state().flatten().copy(), 'time': np.asarray(sim.data.time),
           'body_xpos': sim.data.body_xpos.copy(), 'body_xmat': sim.data.body_xmat.copy(),
           'geom_xpos': sim.data.geom_xpos.copy(), 'geom_xmat': sim.data.geom_xmat.copy()}
    for camera in ('agentview', 'robot0_eye_in_hand'):
        rgb, _unused_depth = sim.render(width=size, height=size, camera_name=camera, depth=True)
        robosuite_seg = sim.render(width=size, height=size, camera_name=camera, segmentation=True)[::-1].copy()
        default_renderer.update_scene(sim.data._data, camera=camera)
        default_seg = native_segmentation(default_renderer)
        grouped_renderer.update_scene(sim.data._data, camera=camera, scene_option=scene_option)
        grouped_renderer.enable_depth_rendering()
        try:
            native_depth = grouped_renderer.render().copy()
        finally:
            grouped_renderer.disable_depth_rendering()
        grouped_seg = native_segmentation(grouped_renderer)
        np.testing.assert_array_equal(grouped_seg, robosuite_seg, err_msg=camera + '_grouped_seg_alignment')
        out[camera + '_rgb'] = rgb[::-1].copy()
        # Depth and segmentation below originate from the same grouped native scene.
        out[camera + '_seg'] = grouped_seg
        out[camera + '_depth_m'] = native_depth
        out[camera + '_robosuite_seg'] = robosuite_seg
        out[camera + '_native_default_seg'] = default_seg
        out[camera + '_K'] = __import__('robosuite.utils.camera_utils', fromlist=['get_camera_intrinsic_matrix']).get_camera_intrinsic_matrix(sim, camera, size, size)
        out[camera + '_camera_to_world'] = __import__('robosuite.utils.camera_utils', fromlist=['get_camera_extrinsic_matrix']).get_camera_extrinsic_matrix(sim, camera).copy()
        assert np.isfinite(native_depth).all() and (native_depth > 0).all()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--init-id', type=int, default=6)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'scope': 'fresh B0 native-depth atomic prerequisite only',
              'code_sha256': sha(__file__), 'seed': a.seed, 'init_id': a.init_id,
              'depth_backend': 'mujoco.Renderer metric depth; top-down; MjvOption geomgroup=011000',
              'segmentation_backend': 'robosuite top-down, exact-equal to native swapped segmentation',
              'd0_completed_trajectories': 0}
    start = time.monotonic()
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
        del get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
        for suite_name in ('libero_spatial', 'libero_object'):
            suite = benchmark.get_benchmark_dict()[suite_name](); task = suite.get_task(0)
            states = suite.get_task_init_states(0); assert len(states) > a.init_id
            bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=a.size, camera_widths=a.size,
                                     camera_depths=True, render_gpu_device_id=-1)
            try:
                np.random.seed(a.seed); env.seed(a.seed); env.reset(); env.set_init_state(states[a.init_id])
                for _ in range(10): env.step([0.] * 6 + [-1.])
                raw_model = env.sim.model._model
                renderer = grouped_renderer = None
                try:
                    renderer = mujoco.Renderer(raw_model, height=a.size, width=a.size)
                    grouped_renderer = mujoco.Renderer(raw_model, height=a.size, width=a.size)
                    option = mujoco.MjvOption(); option.geomgroup[:] = RAY_GEOMGROUP
                    xml = a.out / (suite_name + '_model.xml'); xml.write_text(env.sim.model.get_xml())
                    report.setdefault('inventory', []).append({'suite': suite_name, 'task_id': 0, 'name': task.name,
                        'initial_states': len(states), 'bddl_sha256': sha(bddl), 'model_xml_sha256': sha(xml),
                        'ngeom': env.sim.model.ngeom, 'body_names': list(env.sim.model.body_names),
                        'geom_names': list(env.sim.model.geom_names)})
                    for tick in range(3):
                        first = snapshot(env.sim, renderer, grouped_renderer, option, a.size)
                        second = snapshot(env.sim, renderer, grouped_renderer, option, a.size)
                        for key in first: np.testing.assert_array_equal(first[key], second[key], err_msg=key)
                        dest = a.out / f'{suite_name}_init{a.init_id}_tick{tick}.npz'; np.savez_compressed(dest, **first)
                        report.setdefault('snapshots', []).append({'suite': suite_name, 'init_id': a.init_id, 'tick': tick,
                            'file': dest.name, 'sha256': sha(dest), 'repeat_exact': True,
                            'native_seg_equals_robosuite_seg': True})
                        if tick < 2: env.step([0.01, 0., 0., 0., 0., 0., -1.])
                finally:
                    if grouped_renderer is not None:
                        grouped_renderer.close()
                    if renderer is not None:
                        renderer.close()
            finally:
                env.close()
        assert len(report['snapshots']) == 6
        report['status'] = 'PASS_NATIVE_DEPTH_ATOMIC_REPEATABILITY_ONLY'
        report['limitations'] = ['not full D0 trajectories', 'not independent physical ground truth',
                                 'not WAM/repair/VLA evidence']
    except Exception as exc:
        report['status'] = 'ERROR'; report['error'] = repr(exc); raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        (a.out / 'result.json').write_text(json.dumps(report, indent=2))
        print('RESULT', report['status'], flush=True)


if __name__ == '__main__':
    main()
