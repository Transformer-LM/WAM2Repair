"""Serial CPU simulator clients; reuse a separately verified personal policy server."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('__WAM2REPAIR_ROOT__')
OUT=ROOT/'results/wam2repair/official_grid_20260917'
IMPL=ROOT/'workspace/policy-relevant-imagined-state-repair/implementation'

def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)

def validate(job):
    folder=ROOT/'results/wam2repair'/job['run_id']
    report=json.loads((folder/'result.json').read_text())
    assert report['status']=='EXECUTED'
    for key in ['task','episode','action_mode']: assert report[key]==job[key],key
    assert report['seed']==7 and report['port']==10099
    assert report['checkpoint']==str(ROOT/'models/openpi/official/pi05_libero')
    trajectory=folder/'trajectory.npz'
    assert trajectory.stat().st_size>0
    return {'result':str(folder/'result.json'),'trajectory':str(trajectory),
            'result_sha256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),
            'trajectory_sha256':hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            'success':report['success'],'steps':report['steps'],
            'gripper_contact_frames':report['gripper_contact_frames']}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Advisory lock is held for the lifetime of the scheduler, not a stale-file heuristic.
    import fcntl
    with (OUT/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest_path=OUT/'manifest.json'; state_path=OUT/'queue_state.json'
        if not manifest_path.exists():
            jobs=[]
            for task in [0,1]:
                for episode in [0,1,2,3]:
                    for mode in ['policy','hold_gripper_open']:
                        run=f'official_grid_t{task}_e{episode}_{mode}_20260917'
                        if (task,episode)==(0,0):
                            run='official_pilot_20260917T044133Z' if mode=='policy' else 'official_pilot_20260917T044334Z'
                        jobs.append({'id':f't{task}_e{episode}_{mode}','task':task,'episode':episode,'action_mode':mode,
                            'split':'train' if episode<2 else 'validation' if episode==2 else 'test',
                            'run_id':run,'preexisting':(task,episode)==(0,0)})
            save(manifest_path,{'scope':'predeclared episode-held-out pilot; one CPU client at a time',
                'seed':7,'server_port':10099,'jobs':jobs})
        manifest=json.loads(manifest_path.read_text())
        state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':{}}
        for job in manifest['jobs']:
            old=state['jobs'].get(job['id'],{})
            if old.get('status')=='completed':
                assert validate(job)==old['evidence']; continue
            if old.get('status') in ['running','failed']:
                raise RuntimeError(f"Manual inspection required, do not blindly restart: {job['id']}")
            if job['preexisting']:
                evidence=validate(job)
            else:
                state['jobs'][job['id']]={'status':'running'}; save(state_path,state)
                env=dict(os.environ,W2R_RUN_ID=job['run_id'])
                result=subprocess.run(['bash',str(IMPL/'run_official_pilot.sh'),'--task',str(job['task']),
                    '--episode',str(job['episode']),'--action-mode',job['action_mode']],env=env)
                if result.returncode:
                    state['jobs'][job['id']]={'status':'failed','exit_code':result.returncode}; save(state_path,state)
                    raise RuntimeError(f"Execution failed: {job['id']}")
                try: evidence=validate(job)
                except Exception as error:
                    state['jobs'][job['id']]={'status':'failed','error':repr(error)}; save(state_path,state); raise
            state['jobs'][job['id']]={'status':'completed','evidence':evidence}; save(state_path,state)
            print('COMPLETED',job['id'],json.dumps(evidence),flush=True)
        state['status']='ALL_COMPLETED'; save(state_path,state)

if __name__=='__main__': main()
