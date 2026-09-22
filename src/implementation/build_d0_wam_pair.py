"""Create one action-aligned WAM input/held-out factual-target pair from v7 D0.

The WAM generator may open only ``*_input.npz``.  Simulator future video,
contact and geometry remain in a separate target file, so a generation run
cannot accidentally condition on factual future pixels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wam_rgb(raw: np.ndarray) -> np.ndarray:
    # Contract inherited from the verified FastWAM LIBERO diagnostic: raw
    # simulator storage -> rotate 180 -> bilinear 224, agentview then wrist.
    return np.asarray(Image.fromarray(raw[::-1, ::-1]).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d0', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--candidate', type=int, default=0)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.d0 / 'manifest.json').read_text(encoding='utf-8'))
    accepted_schemas = {
        'wam2repair-physical-bank-v7-analytic-mj-ray-depth-per-frame-exact-id-gated',
        'wam2repair-physical-bank-v8-analytic-mj-ray-depth-per-frame-exact-id-gated-contact-distance-mask',
    }
    if manifest.get('schema') not in accepted_schemas:
        raise ValueError('requires v7 or v8 analytic D0 factual source')
    d0_args = manifest.get('args', {})
    source_episode = {'suite': str(d0_args.get('suite')), 'task': d0_args.get('task'), 'episode': d0_args.get('episode')}
    if source_episode['suite'] == 'None' or not isinstance(source_episode['task'], int) or not isinstance(source_episode['episode'], int):
        raise ValueError('D0 manifest lacks exact suite/task/episode identity')
    records = manifest.get('records', [])
    if a.candidate < 0 or a.candidate >= len(records): raise ValueError('candidate outside D0 manifest')
    record = records[a.candidate]; source = a.d0 / record['file']
    if sha(source) != record['sha256'] or record.get('n_actions') != 8:
        raise ValueError('source candidate file/action-count provenance mismatch')
    with np.load(source, allow_pickle=False) as d:
        actions = d['actions'].astype(np.float32); horizon = len(actions)
        if actions.ndim != 2 or actions.shape[1] != 7 or horizon != 8: raise ValueError(f'requires exact 8-by-7 D0 candidate, got {actions.shape}')
        offsets = np.asarray([0, 2, 4, 6, 8], dtype=np.int32)
        if len(d['sim_time']) != horizon + 1 or offsets[-1] != horizon: raise ValueError('frame/action timing mismatch')
        video = np.stack([np.concatenate([wam_rgb(d[cam + '_rgb'][t]) for cam in ('agentview', 'robot0_eye_in_hand')], axis=1)
                          for t in offsets])
        proprio = np.concatenate([d['eef_pos'][0], Rotation.from_matrix(d['eef_rot'][0]).as_rotvec(), d['gripper_qpos'][0]]).astype(np.float32)
        if hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest() != record.get('candidate_actions_sha256'):
            raise ValueError('candidate internal action array does not match D0 manifest action hash')
        # Physical separation makes broad target-directory globbing impossible
        # for an input-only generator pointed at `wam_input/`.
        input_dir = a.out / 'wam_input'; target_dir = a.out / 'privileged_factual_target'
        input_dir.mkdir(); target_dir.mkdir()
        input_path = input_dir / 'window_0000_input.npz'; target_path = target_dir / 'window_0000_target.npz'
        np.savez_compressed(input_path, current_rgb=video[0], proprio=proprio, actions=actions)
        np.savez_compressed(target_path, video=video, frame_offsets=offsets, times=d['sim_time'][offsets],
                            contact=d['contact_matrix'][offsets], object_pos=d['object_pos'][offsets],
                            object_rot=d['object_rot'][offsets], eef_pos=d['eef_pos'][offsets],
                            eef_rot=d['eef_rot'][offsets], success=d['success'][offsets])
    output = {'schema': 'wam2repair-d0-wam-pair-v1', 'status': 'PACKED_INPUT_TARGET_SEPARATED',
              'scope': 'one factual action-aligned WAM pairing gate; no WAM accuracy, repair, or VLA claim',
              'd0_manifest_sha256': sha(a.d0 / 'manifest.json'), 'd0_candidate_sha256': sha(source),
              'source_d0_episode': source_episode,
              'input': str(input_path.relative_to(a.out)), 'input_allowlist': [str(input_path.relative_to(a.out))],
              'input_sha256': sha(input_path), 'target': str(target_path.relative_to(a.out)), 'target_sha256': sha(target_path),
              'candidate': a.candidate, 'horizon_actions': 8, 'frame_offsets': offsets.tolist(),
              'image_contract': 'raw simulator RGB rotate_180 then PIL bilinear 224; concatenate agentview,wrist to 224x448',
              'generator_permission': 'input-only generator must receive explicit wam_input/window_0000_input.npz allowlist; privileged target is in a separate directory',
              'language': manifest['language'], 'code_sha256': sha(Path(__file__))}
    (a.out / 'manifest.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'status': output['status'], 'out': str(a.out), 'frame_offsets': output['frame_offsets']}), flush=True)


if __name__ == '__main__': main()
