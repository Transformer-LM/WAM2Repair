"""CPU-only v2 prerequisite; not D0 trajectories or a physical-repair result."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(sim, size):
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map)
    sim.forward()
    state = sim.get_state().flatten().copy()
    out = {'state': state, 'time': np.asarray(sim.data.time),
           'body_xpos': sim.data.body_xpos.copy(), 'body_xmat': sim.data.body_xmat.copy(),
           'geom_xpos': sim.data.geom_xpos.copy(), 'geom_xmat': sim.data.geom_xmat.copy()}
    for cam in ['agentview', 'robot0_eye_in_hand']:
        rgb, depth = sim.render(width=size, height=size, camera_name=cam, depth=True)
        seg = sim.render(width=size, height=size, camera_name=cam, segmentation=True)
        # Raw renderer is bottom-up. All stored images here are top-down, no WAM transform.
        out[cam + '_rgb'] = rgb[::-1].copy()
        out[cam + '_depth_m'] = get_real_depth_map(sim, depth[::-1]).copy()
        out[cam + '_seg'] = seg[::-1].copy()
        out[cam + '_K'] = get_camera_intrinsic_matrix(sim, cam, size, size)
        out[cam + '_camera_to_world'] = get_camera_extrinsic_matrix(sim, cam).copy()
        assert np.isfinite(out[cam + '_depth_m']).all()
        assert (out[cam + '_depth_m'] > 0).all()
    contacts = []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        contacts.append([c.geom1, c.geom2, c.dist, *c.pos, *c.frame])
    out['contacts'] = np.asarray(contacts, dtype=np.float64).reshape(-1, 15)
    np.testing.assert_array_equal(state, sim.get_state().flatten())
    for key in ['body_xpos', 'body_xmat', 'geom_xpos', 'geom_xmat']:
        np.testing.assert_array_equal(out[key], getattr(sim.data, key))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--size', type=int, default=256)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    specs = [('libero_spatial', None),
             ('libero_object', 'pick_up_the_milk_and_place_it_in_the_basket'),
             ('libero_goal', 'put_the_cream_cheese_in_the_bowl'),
             ('libero_goal', 'push_the_plate_to_the_front_of_the_stove')]
    report = {'status': 'RUNNING', 'code_sha256': sha(__file__), 'seed': args.seed,
              'scope': 'CPU development snapshot prerequisite, not full D0 or training',
              'inventory': [], 'snapshots': [], 'd0_completed_trajectories': 0,
              'not_validated': ['independent landmark projection', 'cross-view correspondence',
                                'collision/SDF sign', 'contact substeps', 'WAM input pairing']}
    started = time.monotonic()
    try:
        for suite_name, wanted in specs:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            ids = [i for i in range(suite.n_tasks) if wanted is None and i == 0
                   or wanted is not None and suite.get_task(i).name == wanted]
            assert len(ids) == 1, (suite_name, wanted, ids)
            tid = ids[0]; task = suite.get_task(tid)
            states = suite.get_task_init_states(tid)
            bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
            report['inventory'].append({'suite': suite_name, 'task_id': tid, 'name': task.name,
                                        'initial_states': len(states), 'bddl_sha256': sha(bddl)})
            if suite_name == 'libero_goal':
                continue
            # init 4 is development only, outside legacy episodes 0..3.
            assert len(states) > 4
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=args.size,
                                    camera_widths=args.size, camera_depths=True, render_gpu_device_id=-1)
            try:
                np.random.seed(args.seed); env.seed(args.seed); env.reset()
                env.set_init_state(states[4])
                for _ in range(10):
                    env.step([0.] * 6 + [-1.])
                model = env.sim.model
                assets = args.out / (suite_name + '_model.xml')
                assets.write_text(model.get_xml())
                report['inventory'][-1].update({'model_xml_sha256': sha(assets),
                    'physics_dt': float(model.opt.timestep), 'ngeom': model.ngeom,
                    'body_names': list(model.body_names), 'geom_names': list(model.geom_names),
                    'geometry_assets_verified': False})
                for tick in range(3):
                    first = snapshot(env.sim, args.size)
                    second = snapshot(env.sim, args.size)
                    for key in first:
                        np.testing.assert_array_equal(first[key], second[key], err_msg=key)
                    dest = args.out / f'{suite_name}_init4_tick{tick}.npz'
                    np.savez_compressed(dest, **first)
                    row = {'suite': suite_name, 'init_id': 4, 'tick': tick,
                           'time': float(first['time']), 'file': dest.name,
                           'sha256': sha(dest), 'contact_count': len(first['contacts']),
                           'repeat_exact': True, 'state_unchanged_during_render': True}
                    report['snapshots'].append(row)
                    print(json.dumps(row), flush=True)
                    if tick < 2:
                        env.step([0.01, 0., 0., 0., 0., 0., -1.])
            finally:
                env.close()
        report['status'] = 'PASS_ATOMIC_REPEATABILITY_ONLY'
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error'] = repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        (args.out / 'result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
