"""Pack executed simulator trajectories into separated input/target video samples."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--starts',default='all')
    p.add_argument('--split',choices=['diagnostic_only','train','validation','test'],default='diagnostic_only')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    source=a.trajectory/'trajectory.npz'; meta=json.loads((a.trajectory/'result.json').read_text())
    manifest={'schema':'w2r-video-paired-v1','source':str(source),'source_sha256':digest(source),
        'policy_checkpoint':meta['checkpoint'],'scope':'aligned factual bank; WAM generation pending',
        'horizon':16,'frame_offsets':[0,4,8,12,16],
        'action_convention':'executed LIBERO delta-eef controller action, gripper unchanged',
        'image_convention':'rotate raw simulator 180 degrees then PIL bilinear resize to 224; concatenate primary then wrist',
        'records':[]}
    with np.load(source,allow_pickle=False) as d:
        rgb_arrays={cam:d[cam+'_rgb'] for cam in ['agentview','robot0_eye_in_hand']}
        assert len(d['sim_time'])==len(d['actions'])+1
        dt=np.diff(d['sim_time']); assert (dt>0).all()
        np.testing.assert_allclose(dt,dt[0],rtol=1e-6,atol=1e-8)
        group=f"{meta['suite']}/task{meta['task']}/episode{meta['episode']}"
        starts=list(range(0,len(d['actions'])-15,16)) if a.starts=='all' else list(map(int,a.starts.split(',')))
        manifest['selection']={'rule':'all complete nonoverlapping 16-action windows' if a.starts=='all' else 'explicit diagnostic indices',
            'starts':starts,'excluded_tail_actions':len(d['actions'])%16,'source_action_mode':meta.get('action_mode','policy')}
        for start in starts:
            if start<0 or start+16>=len(d['sim_time']): raise ValueError('Incomplete horizon')
            idx=start+np.asarray([0,4,8,12,16])
            frames=[]
            for t in idx:
                frames.append(np.concatenate([np.asarray(Image.fromarray(rgb_arrays[cam][t,::-1,::-1]).resize((224,224),Image.Resampling.BILINEAR)) for cam in ['agentview','robot0_eye_in_hand']],axis=1))
            video=np.stack(frames)
            state=np.concatenate([d['eef_pos'][start],Rotation.from_matrix(d['eef_rot'][start]).as_rotvec(),d['gripper_qpos'][start]]).astype(np.float32)
            sample=f'window_{start:04d}'; ip=a.out/(sample+'_input.npz'); tp=a.out/(sample+'_target.npz')
            np.savez_compressed(ip,current_rgb=video[0],proprio=state,actions=d['actions'][start:start+16])
            # Privileged targets physically separated; WAM generator opens only input.npz.
            np.savez_compressed(tp,video=video,times=d['sim_time'][idx],contact=d['contact_matrix'][idx],
                object_pos=d['object_pos'][idx],object_rot=d['object_rot'][idx],eef_pos=d['eef_pos'][idx],eef_rot=d['eef_rot'][idx],
                slip=d['slip_label'][idx],success=d['success'][idx])
            manifest['records'].append({'id':sample,'group':group,'split':a.split,'start':start,
                'input':ip.name,'input_sha256':digest(ip),'target':tp.name,'target_sha256':digest(tp),
                'language':meta['language'],'episode_success':bool(meta['success']),
                'input_keys':['current_rgb','proprio','actions'],
                'frame_times':d['sim_time'][idx].tolist()})
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'status':'PACKED','samples':len(manifest['records']),'frame_offsets':manifest['frame_offsets'],'group':group}),flush=True)


if __name__=='__main__': main()
