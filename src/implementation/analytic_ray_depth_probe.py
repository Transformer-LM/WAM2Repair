"""CPU-only D0 prerequisite for analytic ``mj_ray`` depth labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from analytic_ray_depth import dense_ray_depth, require_exact_alignment, RAY_GEOMGROUP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raster_product(env, camera, size):
    from robosuite.utils import camera_utils
    sim = env.sim
    # These are robosuite raster products only. Do not instantiate native
    # mujoco.Renderer: its depth/segmentation buffers are explicitly excluded.
    rgb = sim.render(width=size, height=size, camera_name=camera)
    seg = sim.render(width=size, height=size, camera_name=camera, segmentation=True)
    K = camera_utils.get_camera_intrinsic_matrix(sim, camera, size, size)
    T = camera_utils.get_camera_extrinsic_matrix(sim, camera)
    return rgb, seg, K, T


def view_record(env, camera, size):
    sim = env.sim
    # Repeat the complete factual product, rather than merely repeating rays
    # against one cached segmentation array.
    rgb_first, seg_first_raw, K_first, T_first = raster_product(env, camera, size)
    rgb_second, seg_second_raw, K_second, T_second = raster_product(env, camera, size)
    if not (np.array_equal(rgb_first, rgb_second) and np.array_equal(seg_first_raw, seg_second_raw) and
            np.array_equal(K_first, K_second) and np.array_equal(T_first, T_second)):
        raise ValueError(f"{camera}: repeated factual raster/camera products are not exact")
    # Lock the B0 convention before inspecting metrics: robosuite render output
    # is raw storage orientation, but camera ray coordinates are top-down.
    seg_first = seg_first_raw[::-1].copy()
    seg_second = seg_second_raw[::-1].copy()
    first = dense_ray_depth(sim.model._model, sim.data._data, K_first, T_first, seg_first)
    second = dense_ray_depth(sim.model._model, sim.data._data, K_second, T_second, seg_second)
    require_exact_alignment(first)
    require_exact_alignment(second)
    if not (np.array_equal(first.valid_mask, second.valid_mask) and
            np.array_equal(first.ray_geom_id, second.ray_geom_id) and
            np.array_equal(first.depth_m, second.depth_m, equal_nan=True)):
        raise ValueError(f"{camera}: repeated analytic labels are not exact")
    wrong = dense_ray_depth(sim.model._model, sim.data._data, K_first, T_first, seg_first, wrong_vertical_flip=True)
    wrong_rejected = (wrong.metrics['ray_hit_count'] != wrong.metrics['eligible_pixel_count'] or
                      wrong.metrics['ray_geom_id_agreement_fraction'] < .95)
    if not wrong_rejected:
        raise ValueError(f"{camera}: vertical-flip negative control was not rejected: {wrong.metrics}")
    return {
        "camera": camera, "rgb_sha256": hashlib.sha256(rgb_first.tobytes()).hexdigest(),
        "seg_raw_sha256": hashlib.sha256(seg_first_raw.tobytes()).hexdigest(),
        "seg_top_down_sha256": hashlib.sha256(seg_first.tobytes()).hexdigest(),
        "K": K_first.tolist(), "camera_to_world": T_first.tolist(),
        "correct": first.metrics, "wrong_vertical_flip": wrong.metrics,
        "wrong_vertical_flip_rejected": bool(wrong_rejected),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--suite', default='libero_spatial')
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--episode', type=int, default=6)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--render-gpu', type=int, default=-1)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "RUNNING", "scope": "CPU ray-depth alignment prerequisite only; no collision/contact/WAM/VLA claim",
              "native_renderer_instantiated": False, "ray_geomgroup": RAY_GEOMGROUP.tolist(), "records": [],
              "raster_orientation_contract": "robosuite RGB/seg are raw storage orientation; segmentation is vertically flipped to top-down before K/T mj_ray; analytic depth/mask must be flipped back before raw bank storage",
              "args": vars(a), "code_sha256": sha(__file__), "analytic_module_sha256": sha(Path(__file__).with_name('analytic_ray_depth.py'))}
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        suite = benchmark.get_benchmark_dict()[a.suite]()
        task = suite.get_task(a.task)
        init = suite.get_task_init_states(a.task)[a.episode]
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        report['bddl_sha256'] = sha(bddl)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=a.size, camera_widths=a.size,
                                 camera_depths=True, render_gpu_device_id=a.render_gpu)
        try:
            import mujoco
            report['mujoco_version'] = mujoco.__version__
            report['compiled_model_xml_sha256'] = hashlib.sha256(env.sim.model.get_xml().encode('utf-8')).hexdigest()
            env.seed(a.seed); env.reset(); env.set_init_state(init)
            for tick, action in enumerate(([0.] * 6 + [-1.], [0.02, 0., 0., 0., 0., 0., -1.])):
                if tick:
                    env.step(action)
                for camera in ('agentview', 'robot0_eye_in_hand'):
                    item = view_record(env, camera, a.size); item['tick'] = tick; report['records'].append(item)
            assert len(report['records']) == 4
            report['status'] = 'PASS_ANALYTIC_RAY_DEPTH_ONLY'
        finally:
            env.close()
    except Exception as exc:
        report['status'] = 'ERROR'; report['error'] = repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        (out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({"status": report['status'], "path": str(out), "records": len(report['records'])}), flush=True)
    if report['status'] != 'PASS_ANALYTIC_RAY_DEPTH_ONLY':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
