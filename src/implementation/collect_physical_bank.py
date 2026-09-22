"""Small simulator-GT action-conditioned bank. No WAM predictions or robot IO.

Frames are stored in raw simulator orientation. Policy alone rotates both images
180 degrees, matching OpenPI's LIBERO example. Metric depth is computed with
analytic ``mj_ray`` from the same MuJoCo state; no native Renderer buffer is used.
MuJoCo contact != confirmed grasp; slip labels remain unknown.  V8 preserves
an explicit observation mask for contact-pair distances: NaN means that MuJoCo
did not report that owned pair, rather than an assumed zero distance.
"""
import argparse
import hashlib
import json
import pathlib
import time
import numpy as np
from scipy.spatial.transform import Rotation


class Client:
    def __init__(self, port, noise_seed, require_fixed_noise):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        self.msgpack_numpy = msgpack_numpy
        self.ws = connect(f'ws://127.0.0.1:{port}', compression=None,
                          max_size=None, ping_interval=None, open_timeout=60)
        self.metadata = msgpack_numpy.unpackb(self.ws.recv(timeout=60))
        self.noise_rng = np.random.default_rng(noise_seed)
        self.inference_records = []
        self.fixed_noise_enabled = bool(self.metadata.get('wam2repair_fixed_noise', False))
        self.noise_shape = tuple(self.metadata.get('noise_shape', [])) if self.fixed_noise_enabled else None
        if require_fixed_noise and (not self.fixed_noise_enabled or self.noise_shape != (10, 32)):
            raise ValueError('D0 paired collection requires a fixed-noise pi0.5 server with internal shape (10,32)')

    def infer(self, obs, prompt):
        payload = {
            'observation/image': canonical_policy_image(obs['agentview_image']),
            'observation/wrist_image': canonical_policy_image(obs['robot0_eye_in_hand_image']),
            'observation/state': np.concatenate((obs['robot0_eef_pos'],
                Rotation.from_quat(obs['robot0_eef_quat']).as_rotvec(), obs['robot0_gripper_qpos'])).astype(np.float32),
            'prompt': prompt,
        }
        noise = None
        if self.fixed_noise_enabled:
            noise = self.noise_rng.standard_normal(self.noise_shape).astype(np.float32)
            payload['_wam2repair_noise'] = noise
        self.ws.send(self.msgpack_numpy.Packer().pack(payload))
        raw = self.ws.recv(timeout=1200)
        if isinstance(raw, str):
            raise RuntimeError(raw)
        actions = np.asarray(self.msgpack_numpy.unpackb(raw)['actions'], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError(f'Invalid actions: {actions.shape}')
        self.inference_records.append({
            'ordinal': len(self.inference_records),
            'internal_noise_sha256': (hashlib.sha256(noise.tobytes()).hexdigest() if noise is not None else None),
            'policy_actions_sha256': action_sha256(actions), 'policy_actions_shape': list(actions.shape),
        })
        return actions


RAY_GEOMGROUP = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)


def canonical_policy_image(image):
    from openpi_client import image_tools
    return image_tools.resize_with_pad(np.ascontiguousarray(image[::-1, ::-1]), 224, 224)


def action_sha256(actions):
    a = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, bytes):
        return {'bytes_hex': value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def snapshot(env, obs, objects, geom_owner, factual_size):
    from robosuite.utils import camera_utils
    sim = env.sim
    ids = [env.env.obj_body_id[name] for name in objects]
    pos = np.asarray(sim.data.body_xpos)[ids].copy()
    rot = np.asarray(sim.data.body_xmat)[ids].copy().reshape(-1, 3, 3)
    eef_pos = np.asarray(obs['robot0_eef_pos']).copy()
    eef_rot = Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
    contacts = np.zeros((len(objects)+1, len(objects)+1), dtype=bool)
    # Keep the legacy zero-filled field for backward readers, but never use it
    # as signed-distance supervision.  The v8 pair below is the authoritative
    # value/mask representation.
    minimum = np.zeros_like(contacts, dtype=np.float32)
    distance_observed = np.zeros_like(contacts, dtype=bool)
    signed_distance = np.full(contacts.shape, np.nan, dtype=np.float32)
    for c in sim.data.contact[:sim.data.ncon]:
        i, j = geom_owner.get(int(c.geom1)), geom_owner.get(int(c.geom2))
        if i is not None and j is not None and i != j:
            distance_observed[i, j] = distance_observed[j, i] = True
            old = signed_distance[i, j]
            value = np.float32(c.dist)
            signed_distance[i, j] = signed_distance[j, i] = value if np.isnan(old) else min(old, value)
            if c.dist <= 0:
                contacts[i, j] = contacts[j, i] = True
                minimum[i, j] = minimum[j, i] = min(minimum[i, j], float(c.dist))
    row = {
        'object_pos': pos, 'object_rot': rot, 'eef_pos': eef_pos, 'eef_rot': eef_rot,
        'eef_relative_pos': (pos-eef_pos) @ eef_rot,
        'eef_relative_rot': np.einsum('ij,njk->nik', eef_rot.T, rot),
        'contact_matrix': contacts, 'contact_min_distance': minimum,
        'contact_distance_observed': distance_observed,
        'contact_signed_distance_m': signed_distance,
        'gripper_qpos': np.asarray(obs['robot0_gripper_qpos']).copy(),
        'qpos': np.asarray(sim.data.qpos).copy(), 'qvel': np.asarray(sim.data.qvel).copy(),
        'sim_time': np.asarray(sim.data.time),
        'slip_label': np.full(len(objects), -1, dtype=np.int8),
    }
    for cam in ['agentview', 'robot0_eye_in_hand']:
        policy_observation_raw = np.asarray(obs[cam+'_image']).copy()
        policy_rgb = canonical_policy_image(policy_observation_raw)
        robosuite_rgb_raw = sim.render(width=factual_size, height=factual_size,
                                       camera_name=cam)
        # Keep both products: factual RGB is paired with factual seg/depth at a renderer-stable
        # resolution; policy RGB is the exact 224px observation that generated the action.
        row[cam+'_rgb'] = robosuite_rgb_raw.copy()
        row[cam+'_policy_rgb'] = policy_rgb
        row[cam+'_policy_observation_raw_rgb'] = policy_observation_raw
        row[cam+'_factual_rgb_shape'] = np.asarray(robosuite_rgb_raw.shape, dtype=np.int32)
        row[cam+'_policy_rgb_shape'] = np.asarray(policy_rgb.shape, dtype=np.int32)
        row[cam+'_policy_vs_factual_rgb_comparable'] = np.asarray(False)
        robosuite_seg_raw = sim.render(width=factual_size, height=factual_size,
                                       camera_name=cam, segmentation=True)
        K = camera_utils.get_camera_intrinsic_matrix(sim, cam, factual_size, factual_size)
        T = camera_utils.get_camera_extrinsic_matrix(sim, cam)
        # Predeclared B0 convention: raw raster storage is vertically flipped
        # to top-down for K/T ray construction, then labels are flipped back.
        from analytic_ray_depth import dense_ray_depth, require_exact_alignment
        good = dense_ray_depth(sim.model._model, sim.data._data, K, T, robosuite_seg_raw[::-1])
        require_exact_alignment(good)
        bad = dense_ray_depth(sim.model._model, sim.data._data, K, T, robosuite_seg_raw[::-1],
                              wrong_vertical_flip=True)
        rejected = (bad.metrics['ray_hit_count'] != bad.metrics['eligible_pixel_count'] or
                    bad.metrics['ray_geom_id_agreement_fraction'] < .95)
        if not rejected:
            raise ValueError(f'{cam} vertical-flip negative control was not rejected: {bad.metrics}')
        row[cam+'_depth_m'] = good.depth_m[::-1].astype(np.float32)
        row[cam+'_seg'] = robosuite_seg_raw
        row[cam+'_depth_valid_mask'] = good.valid_mask[::-1]
        row[cam+'_ray_geom_id'] = good.ray_geom_id[::-1]
        row[cam+'_ray_eligible_mask'] = good.eligible_mask[::-1]
        row[cam+'_ray_eligible_pixel_count'] = np.asarray(good.metrics['eligible_pixel_count'], dtype=np.int32)
        row[cam+'_ray_hit_count'] = np.asarray(good.metrics['ray_hit_count'], dtype=np.int32)
        row[cam+'_ray_geom_match_fraction'] = np.asarray(good.metrics['ray_geom_id_agreement_fraction'], dtype=np.float64)
        row[cam+'_ray_valid_coverage_fraction'] = np.asarray(good.metrics['valid_coverage_fraction'], dtype=np.float64)
        row[cam+'_ray_gate_passed'] = np.asarray(True)
        row[cam+'_ray_wrong_flip_rejected'] = np.asarray(rejected)
        height, width = row[cam+'_rgb'].shape[:2]
        row[cam+'_K'] = K
        row[cam+'_camera_to_world'] = T
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--suite', default='libero_spatial')
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--prefix-steps', type=int, default=0)
    ap.add_argument('--prefix-actions', type=pathlib.Path, default=None,
                    help='explicit N-by-7 action replay from an audited contact screen; never re-inferred')
    ap.add_argument('--prefix-actions-provenance', type=pathlib.Path, default=None,
                    help='screen result.json that binds --prefix-actions by filename and SHA')
    ap.add_argument('--candidates', type=int, default=2)
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--port', type=int, default=10098)
    ap.add_argument('--noise-seed', type=int, default=123)
    ap.add_argument('--require-fixed-noise', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--render-gpu', type=int, default=-1)
    ap.add_argument('--factual-size', type=int, default=256)
    ap.add_argument('--env-camera-size', type=int, default=256)
    ap.add_argument('--action-source', choices=['openpi', 'scripted-sanity'], default='openpi')
    ap.add_argument('--declared-split', choices=['sanity_only', 'train', 'validation', 'heldout'],
                    default='sanity_only',
                    help='provenance role declared by a pre-existing experiment gate; defaults to excluded sanity-only')
    ap.add_argument('--candidate-manifest', type=pathlib.Path, default=None,
                    help='optional SHA-bound explicit action candidates; disables fresh candidate policy calls only')
    a = ap.parse_args()
    if a.declared_split != 'sanity_only' and a.action_source != 'openpi':
        ap.error('a train/validation/heldout collection requires --action-source openpi')
    if a.horizon < 1 or a.candidates < 1:
        ap.error('horizon and candidates must be positive')
    if a.factual_size != 256:
        ap.error('D0 collector is locked to the B0-validated 256px factual stream; use a separately validated joint resampling bridge for other sizes')
    if a.env_camera_size != 256:
        ap.error('D0 collector is locked to the B0-validated 256px OffScreenRenderEnv camera context')
    if a.prefix_actions is not None:
        if a.prefix_actions_provenance is None:
            ap.error('--prefix-actions requires --prefix-actions-provenance for fail-closed screen binding')
        if a.action_source != 'openpi':
            ap.error('explicit screened prefix replay is only valid for openpi factual collection')
        if not a.prefix_actions.is_file() or not a.prefix_actions_provenance.is_file():
            ap.error('prefix action/provenance paths must be existing regular files')
        external_prefix = np.load(a.prefix_actions, allow_pickle=False)
        if external_prefix.ndim != 2 or external_prefix.shape[1] != 7 or not np.isfinite(external_prefix).all():
            ap.error(f'prefix actions must be finite N-by-7, got {external_prefix.shape}')
        if a.prefix_steps not in (0, len(external_prefix)):
            ap.error('--prefix-steps must be zero or exactly match --prefix-actions length')
        provenance = json.loads(a.prefix_actions_provenance.read_text(encoding='utf-8'))
        if provenance.get('status') != 'COMPLETED_CONTACT_COVERAGE_SCREEN':
            ap.error('prefix provenance must be a completed contact-coverage screen')
        screen_args = provenance.get('args', {})
        if (screen_args.get('suite') != a.suite or screen_args.get('task') != a.task or
                screen_args.get('seed') != a.seed or screen_args.get('resolution') != 256):
            ap.error('screen suite/task/seed/256px camera contract does not match D0 arguments')
        action_hash = action_sha256(external_prefix)
        action_file_sha = hashlib.sha256(a.prefix_actions.read_bytes()).hexdigest()
        matching = [r for r in provenance.get('episode_results', [])
                    if r.get('episode') == a.episode and r.get('prefix_actions_file') == a.prefix_actions.name and
                    r.get('action_sha256') == action_hash and r.get('prefix_actions_file_sha256') == action_file_sha and
                    r.get('n_actions') == len(external_prefix)]
        if len(matching) != 1:
            ap.error('prefix action file/SHA is not bound exactly once by supplied screen provenance')
        a.prefix_steps = int(len(external_prefix))
    else:
        external_prefix = provenance = None
    if a.action_source == 'scripted-sanity' and a.prefix_steps:
        ap.error('scripted sanity does not support policy prefix steps')
    explicit_candidates = None
    explicit_candidate_manifest_sha = None
    if a.candidate_manifest is not None:
        if a.action_source != 'openpi' or not a.candidate_manifest.is_file():
            ap.error('explicit candidate manifest requires an existing openpi collection manifest')
        if external_prefix is None:
            ap.error('explicit candidate manifest requires a screened explicit --prefix-actions binding')
        candidate_manifest = json.loads(a.candidate_manifest.read_text(encoding='utf-8'))
        if candidate_manifest.get('schema') != 'wam2repair-counterfactual-action-candidates-v1' or candidate_manifest.get('status') != 'PREDECLARED_ACTIONS':
            ap.error('invalid explicit candidate manifest schema/status')
        if candidate_manifest.get('prefix_actions_sha256') != action_sha256(external_prefix):
            ap.error('candidate manifest prefix SHA does not bind the replayed screened prefix')
        rows = candidate_manifest.get('candidates')
        if not isinstance(rows, list) or len(rows) != a.candidates:
            ap.error('candidate manifest count does not match --candidates')
        explicit_candidates = []
        for i, row in enumerate(rows):
            x = np.asarray(row.get('actions'), dtype=np.float32)
            if x.shape != (a.horizon, 7) or not np.isfinite(x).all() or row.get('actions_sha256') != action_sha256(x):
                ap.error(f'explicit candidate {i} action contract/SHA failure')
            explicit_candidates.append(x)
        explicit_candidate_manifest_sha = hashlib.sha256(a.candidate_manifest.read_bytes()).hexdigest()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import mujoco
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    suite = benchmark.get_benchmark_dict()[a.suite]()
    task = suite.get_task(a.task)
    init = suite.get_task_init_states(a.task)[a.episode]
    bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    # LIBERO samples fixture placement during construction. This must happen
    # under the declared seed, before `env.seed/reset`, for cross-process
    # screen-prefix provenance to identify the same compiled scene.
    np.random.seed(a.seed)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                            camera_heights=a.env_camera_size, camera_widths=a.env_camera_size, camera_depths=True,
                            render_gpu_device_id=a.render_gpu)
    client = None
    def reset():
        env.seed(a.seed)
        env.reset()
        obs = env.set_init_state(init)
        for _ in range(10):
            obs, *_ = env.step([0.]*6+[-1.])
        return obs
    try:
        if external_prefix is not None:
            compiled_xml_sha = hashlib.sha256(env.sim.model.get_xml().encode('utf-8')).hexdigest()
            if (provenance.get('bddl_sha256') != hashlib.sha256(bddl.read_bytes()).hexdigest() or
                    provenance.get('compiled_model_xml_sha256') != compiled_xml_sha):
                raise ValueError('screen BDDL/compiled-model provenance does not match current D0 environment')
        if a.action_source == 'openpi':
            client = Client(a.port, a.noise_seed, a.require_fixed_noise)
        obs = reset()
        objects = sorted(env.env.obj_body_id)
        owner = {}
        all_objects = {**env.env.objects_dict, **env.env.fixtures_dict}
        for i, name in enumerate(objects):
            for geom in all_objects[name].contact_geoms:
                owner[env.sim.model.geom_name2id(geom)] = i
        for geom in env.robots[0].gripper.contact_geoms:
            owner[env.sim.model.geom_name2id(geom)] = len(objects)
        prefix = []
        if external_prefix is not None:
            prefix = [action.copy() for action in external_prefix]
            for action in prefix:
                obs, *_ = env.step(action.tolist())
        else:
            while len(prefix) < a.prefix_steps:
                for action in client.infer(obs, task.language):
                    if len(prefix) >= a.prefix_steps:
                        break
                    obs, *_ = env.step(action.tolist())
                    prefix.append(action.copy())
        reference = snapshot(env, obs, objects, owner, a.factual_size)
        candidate_policy_call_ordinals = []
        if explicit_candidates is not None:
            candidates = [x.copy() for x in explicit_candidates]
            candidate_policy_call_ordinals = [None] * len(candidates)
        elif client is not None:
            candidates = []
            for _ in range(a.candidates):
                candidates.append(client.infer(obs, task.language)[:a.horizon])
                candidate_policy_call_ordinals.append(client.inference_records[-1]['ordinal'])
        else:
            candidates = []
            for k in range(a.candidates):
                chunk = np.zeros((a.horizon, 7), dtype=np.float32)
                chunk[:, -1] = -1.
                if k:
                    chunk[:, (k-1) % 3] = .1 * (-1 if k % 2 == 0 else 1)
                candidates.append(chunk)
        if any(len(chunk) != a.horizon for chunk in candidates):
            raise ValueError('Policy action horizon shorter than requested horizon')
        policy_metadata = json_safe(client.metadata) if client is not None else None
        analytic_path = pathlib.Path(__file__).with_name('analytic_ray_depth.py')
        manifest = {'schema':'wam2repair-physical-bank-v8-analytic-mj-ray-depth-per-frame-exact-id-gated-contact-distance-mask', 'args':json_safe(vars(a)), 'objects':objects,
                    'contact_entities':objects+['gripper'], 'language':task.language,
                    'labels':'simulator object/eef poses and instantaneous contact; slip unknown (-1)',
                    'prediction_source':'none: simulator factual collection only',
                    'action_source':a.action_source,
                    'eligible_for_d0_claims': bool(a.action_source == 'openpi'),
                    'scripted_sanity_excluded_from_d0_claims': bool(a.action_source != 'openpi'),
                    'policy_server_metadata': policy_metadata,
                    'policy_server_metadata_sha256': (hashlib.sha256(json.dumps(policy_metadata, sort_keys=True,
                        separators=(',', ':')).encode('utf-8')).hexdigest() if policy_metadata is not None else None),
                    'scope':('factual simulator alignment sanity; not WAM/VLA performance evidence'
                             if a.declared_split == 'sanity_only' else
                             'factual simulator target for the explicitly declared repair split; not WAM/VLA performance evidence'),
                    'split':('sanity_only; excluded from train/validation/test'
                             if a.declared_split == 'sanity_only' else a.declared_split),
                    'policy_checkpoint':('pi05_base_pytorch; LIBERO skill quality not established' if client else None),
                    'rotation_convention':'world-from-body 3x3; eef_relative=inv(T_world_eef)*T_world_object',
                    'image_convention':'raw simulator orientation; rotate 180deg only for policy',
                    'rgb_provenance':'*_rgb is factual 256px sim.render RGB paired with factual seg/depth; *_policy_observation_raw_rgb is raw env observation; *_policy_rgb is the exact rotated/resized/padded 224px pi0.5 input. Streams are separate and no pixel equality is claimed.',
                    'downstream_stream_contract':'WAM/geometric repair may use only *_rgb with matching *_depth_m/*_seg. Pi0.5 action-context provenance uses only *_policy_rgb. The streams are not the same pixels and this bank does not authorize an undeclared direct image-substitution interface.',
                    'limitations':['pretrained WAM input resolution is not established by this bank',
                                   'if WAM requires 224px, use only a separately declared and validated deterministic joint RGB/depth/seg resampling bridge',
                                   'do not substitute 224px policy RGB for factual RGB or resample RGB alone'],
                    'fixed_noise_enabled': (client.fixed_noise_enabled if client is not None else False),
                    'fixed_noise_required': bool(a.require_fixed_noise),
                    'fixed_noise_seed': (a.noise_seed if client is not None else None),
                    'fixed_noise_internal_shape': (list(client.noise_shape) if client is not None and client.noise_shape else None),
                    'candidate_policy_call_ordinals': candidate_policy_call_ordinals,
                    'explicit_candidate_manifest_path': (str(a.candidate_manifest) if a.candidate_manifest is not None else None),
                    'explicit_candidate_manifest_sha256': explicit_candidate_manifest_sha,
                    'prefix_action_replay': ({'mode':'screened_explicit_action_replay_not_reinferred',
                        'actions_path':str(a.prefix_actions),
                        'actions_sha256':action_sha256(np.asarray(prefix, dtype=np.float32).reshape(-1,7)),
                        'actions_file_sha256':hashlib.sha256(a.prefix_actions.read_bytes()).hexdigest(),
                        'screen_provenance_path':str(a.prefix_actions_provenance),
                        'screen_provenance_sha256':hashlib.sha256(a.prefix_actions_provenance.read_bytes()).hexdigest(),
                        'screen_episode':matching[0].get('episode'),
                        'reproduces':'explicit prefix action sequence/state only',
                        'does_not_reproduce':'screen continuation candidate/noise draws; D0 candidate calls start a fresh fixed-noise sequence'} if external_prefix is not None else
                        {'mode':'policy_reinferred_prefix', 'prefix_steps':a.prefix_steps}),
                    'depth_backend':'analytic mujoco.mj_ray axial depth; raw storage orientation; top-down K/T ray convention; geomgroup=011000',
                    'factual_stream_resolution':[a.factual_size, a.factual_size],
                    'policy_input_resolution':[224, 224],
                    'policy_preprocessing':'rotate_180_then_resize_with_pad_to_224',
                    'depth_seg_alignment':'robosuite raw segmentation is flipped to the predeclared top-down K/T convention; analytic mj_ray depth is valid only at 3x3-stable selected-group geom pixels with exact ray/seg geom-ID agreement. Depth is NaN elsewhere and depth_valid_mask is mandatory.',
                    'per_frame_ray_gate_contract':{'geomgroup':[0,1,1,0,0,0], 'pixel_offset':.5, 'axial_depth':True, 'minimum_eligible_pixels':100, 'required_ray_hit_fraction':1.0, 'required_geom_id_agreement_fraction':1.0, 'required_valid_coverage_fraction':1.0, 'invalid_depth':'NaN', 'wrong_vertical_flip_must_be_rejected':True},
                    'analytic_ray_depth_code_sha256':hashlib.sha256(analytic_path.read_bytes()).hexdigest(),
                    'bddl_sha256':hashlib.sha256(bddl.read_bytes()).hexdigest(),
                    'compiled_model_xml_sha256':hashlib.sha256(env.sim.model.get_xml().encode('utf-8')).hexdigest(),
                    'mujoco_version':mujoco.__version__,
                    'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                    'records':[]}
        manifest['policy_inference_calls'] = list(client.inference_records) if client is not None else []
        prefix_array = np.asarray(prefix, dtype=np.float32).reshape(-1,7)
        np.save(out/'prefix_actions.npy', prefix_array)
        manifest['prefix_actions_sha256'] = action_sha256(prefix_array)
        for k, chunk in enumerate(candidates):
            obs = reset()
            for action in prefix:
                obs, *_ = env.step(action.tolist())
            before = snapshot(env, obs, objects, owner, a.factual_size)
            reset_error = float(np.max(np.abs(before['qpos']-reference['qpos'])))
            velocity_reset_error = float(np.max(np.abs(before['qvel']-reference['qvel'])))
            if max(reset_error, velocity_reset_error) > 1e-6:
                raise ValueError(f'Unpaired reset: qpos error {reset_error}')
            for key in ('sim_time', 'agentview_rgb', 'robot0_eye_in_hand_rgb',
                        'agentview_policy_rgb', 'robot0_eye_in_hand_policy_rgb',
                        'agentview_K', 'robot0_eye_in_hand_K',
                        'agentview_camera_to_world', 'robot0_eye_in_hand_camera_to_world'):
                np.testing.assert_array_equal(before[key], reference[key], err_msg='prefix_replay_' + key)
            rows, rewards, success = [before], [], [bool(env.check_success())]
            for action in chunk:
                obs, reward, done, _ = env.step(action.tolist())
                rows.append(snapshot(env, obs, objects, owner, a.factual_size))
                rewards.append(float(reward))
                success.append(bool(env.check_success()))
            arrays = {key:np.stack([r[key] for r in rows]) for key in before}
            arrays.update(actions=chunk, rewards=np.asarray(rewards), success=np.asarray(success))
            for key, value in arrays.items():
                if key.endswith('_depth_m'):
                    camera = key[:-len('_depth_m')]
                    valid = arrays[camera + '_depth_valid_mask']
                    if not (np.isfinite(value[valid]).all() and np.isnan(value[~valid]).all()):
                        raise ValueError(f'{key} violates required finite-valid/NaN-invalid depth contract')
                elif key == 'contact_signed_distance_m':
                    observed = arrays['contact_distance_observed']
                    if not (np.isfinite(value[observed]).all() and np.isnan(value[~observed]).all()):
                        raise ValueError('contact signed-distance/mask contract violation')
                elif value.dtype.kind in 'fc' and not np.isfinite(value).all():
                    raise ValueError(f'Nonfinite {key}')
            file = out/f'candidate_{k:02d}.npz'
            np.savez_compressed(file, **arrays)
            manifest['records'].append({'file':file.name, 'n_actions':len(chunk),
                'qpos_reset_error':reset_error, 'success_after_chunk':bool(success[-1]),
                'qvel_reset_error':velocity_reset_error,
                'candidate_actions_sha256': action_sha256(chunk),
                'policy_inference_ordinal': (candidate_policy_call_ordinals[k] if client is not None else None),
                'gripper_contact_frames':int(arrays['contact_matrix'][:,:-1,-1].any(axis=1).sum()),
                'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
            print(json.dumps(manifest['records'][-1]), flush=True)
        manifest['finished_unix'] = time.time()
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps({'status':'COLLECTED', 'path':str(out), 'records':manifest['records']}), flush=True)
    finally:
        if client is not None:
            client.ws.close()
        env.close()


if __name__ == '__main__':
    main()
