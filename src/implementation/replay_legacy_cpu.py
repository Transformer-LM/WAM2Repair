"""Replay legacy saved actions without any policy call; audit endpoint alignment."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    meta=json.loads(a.prepared.with_suffix('.json').read_text())
    report={'scope':'legacy StarVLA action replay, CPU OSMesa; not pi0.5 evidence','records':[]}
    with np.load(a.prepared.with_suffix('.npz'),allow_pickle=False) as source:
        for r in meta['records'][:1]:
            suite=benchmark.get_benchmark_dict()[r['suite']]()
            task=suite.get_task(r['task_id'])
            bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
            env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,camera_depths=True,render_gpu_device_id=-1)
            try:
                env.seed(meta['seed']+r['task_id']); env.reset()
                obs=env.set_init_state(suite.get_task_init_states(r['task_id'])[r['episode']])
                for _ in range(10): obs,*_=env.step([0.]*6+[-1.])
                state=np.concatenate([obs['robot0_eef_pos'],Rotation.from_quat(obs['robot0_eef_quat']).as_rotvec(),obs['robot0_gripper_qpos']])
                key=r['key']; rows=[]
                def capture():
                    return { 'primary':np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),
                        'wrist':np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),
                        'qpos':env.sim.data.qpos.copy(),'qvel':env.sim.data.qvel.copy(),
                        'sim_time':np.asarray(env.sim.data.time),
                        'body_xpos':env.sim.data.body_xpos.copy(), 'body_xmat':env.sim.data.body_xmat.copy()}
                rows.append(capture()); executed=[]
                for raw in source[key+'_action']:
                    action=raw.copy(); action[6]=1.-2.*float(action[6]>.5)
                    obs,*_=env.step(action.tolist()); rows.append(capture()); executed.append(action)
                arrays={k:np.stack([row[k] for row in rows]) for k in rows[0]}
                arrays['executed_actions']=np.asarray(executed)
                np.savez_compressed(a.out/(key+'_replay.npz'),**arrays)
                result={'key':key,'initial_proprio_max_abs_error':float(np.max(np.abs(state-source[key+'_state']))),
                        'n_frames':len(rows),'initial_state_exact_reconstruction':'not certifiable: legacy qpos/qvel not saved'}
                for cam in ['primary','wrist']:
                    for index,suffix in [(0,cam),(-1,'gt_'+cam)]:
                        diff=arrays[cam][index].astype(float)-source[key+'_'+suffix].astype(float)
                        result[suffix+'_mse']=float(np.mean(diff**2))
                result['status']='REPLAYED_UNVERIFIED'  # RGB agreement alone cannot certify simulator state.
                report['records'].append(result)
                print(json.dumps(result),flush=True)
            finally: env.close()
    (a.out/'replay_report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
