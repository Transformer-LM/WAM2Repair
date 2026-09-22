"""Cheap π0.5 contact-coverage screen before expensive dense-depth D0 capture.

It never creates a repair target or a claim-bearing factual bank.  Episodes
with observed gripper contacts are candidates for the separately gated v7
collector; zero-contact episodes are preserved as negative coverage evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from collect_physical_bank import Client, action_sha256, json_safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--suite', default='libero_spatial')
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--episodes', default='0,1,2,3,4,5,6,7')
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--prefix-chunks', type=int, default=1,
                   help='predeclared count of consecutive 10-step fixed-noise policy chunks')
    p.add_argument('--prefix-steps', type=int, default=None,
                   help='optional exact positive prefix length; final 10-step policy chunk is deterministically truncated and recorded')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--noise-seed', type=int, default=777)
    p.add_argument('--resolution', type=int, default=256)
    p.add_argument('--render-gpu', type=int, default=-1)
    a = p.parse_args()
    episode_ids = [int(x) for x in a.episodes.split(',') if x.strip()]
    if not episode_ids or a.horizon != 10 or a.prefix_chunks < 1:
        p.error('contact screen requires one or more consecutive 10-step policy chunks per episode')
    if a.prefix_steps is not None and a.prefix_steps < 1:
        p.error('--prefix-steps must be positive when supplied')
    requested_steps = a.prefix_steps if a.prefix_steps is not None else a.prefix_chunks * a.horizon
    if a.resolution != 256:
        p.error('contact screen is locked to the v7 D0 256px environment camera context')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'status': 'RUNNING', 'scope': 'π0.5 contact-coverage screen only; not D0 bank/training/WAM/VLA evaluation',
              'args': vars(a), 'episode_results': [], 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'prefix_contract': 'each requested chunk is exactly one (10,7) fixed-noise pi0.5 action chunk; when prefix_steps is set the final chunk is deterministically truncated to that exact saved action count. Saved actions may be replayed explicitly (never re-inferred) by D0 to reproduce only the prefix state. D0 candidate calls use their own fresh fixed-noise sequence.'}
    report['env_camera_resolution'] = [256, 256]
    report['policy_preprocessing'] = 'rotate_180_then_resize_with_pad_to_224; identical to collect_physical_bank canonical_policy_image'
    client = None; env = None
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        suite = benchmark.get_benchmark_dict()[a.suite](); task = suite.get_task(a.task)
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        report['bddl_sha256'] = hashlib.sha256(bddl.read_bytes()).hexdigest()
        # Match the v7 D0 model construction exactly. Even though this screen
        # does not consume a depth raster, changing camera_depths changes the
        # compiled model fingerprint and invalidates prefix-state provenance.
        # LIBERO samples fixture placements while constructing the environment.
        # Seed *before* construction so this screen and D0 can share an XML/state
        # provenance anchor across processes.
        np.random.seed(a.seed)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=a.resolution, camera_widths=a.resolution,
                                 camera_depths=True, render_gpu_device_id=a.render_gpu)
        client = Client(a.port, a.noise_seed, True)
        report['policy_server_metadata'] = json_safe(client.metadata)
        report['collect_physical_bank_code_sha256'] = hashlib.sha256(Path(__file__).with_name('collect_physical_bank.py').read_bytes()).hexdigest()
        import mujoco
        report['mujoco_version'] = mujoco.__version__
        report['compiled_model_xml_sha256'] = hashlib.sha256(env.sim.model.get_xml().encode('utf-8')).hexdigest()
        init_states = suite.get_task_init_states(a.task)
        invalid = [episode for episode in episode_ids if episode < 0 or episode >= len(init_states)]
        if invalid:
            raise ValueError(f'episodes {invalid} outside [0,{len(init_states)})')
        for episode in episode_ids:
            env.seed(a.seed); env.reset(); obs = env.set_init_state(init_states[episode])
            for _ in range(10): obs, *_ = env.step([0.] * 6 + [-1.])
            gripper = {env.sim.model.geom_name2id(g) for g in env.robots[0].gripper.contact_geoms}
            # `obj_body_id` also includes task fixtures (for example a stove).
            # `objects_dict` is the movable-object collection; fixtures are
            # deliberately excluded from this label.
            target_names = sorted(env.env.objects_dict)
            target_by_geom = {}
            for name in target_names:
                for geom in env.env.objects_dict[name].contact_geoms:
                    target_by_geom[env.sim.model.geom_name2id(geom)] = name
            contact_steps, any_geometry_steps, object_contacts, prefix_chunks, all_actions = [], [], [], [], []
            chunk_index = 0
            while len(all_actions) < requested_steps:
                chunk = client.infer(obs, task.language)
                if chunk.shape != (10, 7):
                    raise ValueError(f'fixed-noise π0.5 must return one (10,7) chunk, got {chunk.shape}')
                remaining = requested_steps - len(all_actions)
                consumed = chunk[:remaining]
                prefix_chunks.append({'chunk_index': chunk_index, 'action_sha256': action_sha256(chunk),
                                      'policy_inference_ordinal': client.inference_records[-1]['ordinal'],
                                      'returned_actions': chunk.tolist(), 'consumed_actions': consumed.tolist(),
                                      'consumed_count': int(len(consumed))})
                for local_step, action in enumerate(consumed):
                    step = len(all_actions); all_actions.append(action.copy())
                    obs, *_ = env.step(action.tolist())
                    hit_objects = set(); any_geometry = False
                    for c in env.sim.data.contact[:env.sim.data.ncon]:
                        left, right = int(c.geom1), int(c.geom2)
                        if c.dist > 0: continue
                        if (left in gripper) != (right in gripper): any_geometry = True
                        other = right if left in gripper else left if right in gripper else None
                        if other in target_by_geom: hit_objects.add(target_by_geom[other])
                    if any_geometry: any_geometry_steps.append(step)
                    if hit_objects:
                        contact_steps.append(step); object_contacts.append({'step': step, 'target_objects': sorted(hit_objects)})
                chunk_index += 1
            all_actions = np.asarray(all_actions, dtype=np.float32)
            action_file = out / f'episode_{episode:03d}_prefix_actions.npy'
            np.save(action_file, all_actions)
            report['episode_results'].append({'episode': episode, 'n_actions': len(all_actions),
                'action_sha256': action_sha256(all_actions), 'prefix_actions_file': action_file.name,
                'prefix_actions_file_sha256': hashlib.sha256(action_file.read_bytes()).hexdigest(),
                'prefix_chunks': prefix_chunks,
                'target_object_names': target_names, 'gripper_object_contact_steps': contact_steps,
                'gripper_object_contact_frames': len(contact_steps), 'gripper_object_contacts': object_contacts,
                'gripper_any_geometry_steps_diagnostic': any_geometry_steps,
                'zero_object_contact_interpretation': 'not observed in these fixed-noise prefix chunks; not an episode-level negative label',
                'success_after_chunk': bool(env.check_success())})
            print(json.dumps(report['episode_results'][-1]), flush=True)
        report['policy_inference_calls'] = client.inference_records
        report['status'] = 'COMPLETED_CONTACT_COVERAGE_SCREEN'
    except Exception as exc:
        report['status'] = 'ERROR'; report['error'] = repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        (out/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if client is not None: client.ws.close()
        if env is not None: env.close()
        print(json.dumps({'status': report['status'], 'path': str(out)}), flush=True)
    if report['status'] != 'COMPLETED_CONTACT_COVERAGE_SCREEN': raise SystemExit(2)


if __name__ == '__main__':
    main()
