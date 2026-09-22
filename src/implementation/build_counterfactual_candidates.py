"""Build SHA-bound controlled candidates from one screened pi0.5 state.

No simulator future is collected here.  It only replays a screen-bound prefix,
obtains one policy chunk, and serializes predeclared deterministic transforms.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

from collect_physical_bank import Client, action_sha256, canonical_policy_image


def file_sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True); p.add_argument('--episode', type=int, required=True)
    p.add_argument('--prefix-actions', type=Path, required=True); p.add_argument('--provenance', type=Path, required=True)
    p.add_argument('--port', type=int, required=True); p.add_argument('--seed', type=int, default=7)
    p.add_argument('--noise-seed', type=int, default=123); p.add_argument('--horizon', type=int, default=8)
    a=p.parse_args()
    if a.horizon != 8 or a.out.exists(): p.error('requires horizon 8 and nonexistent output')
    prov=json.loads(a.provenance.read_text())
    x=np.load(a.prefix_actions,allow_pickle=False).astype(np.float32)
    matches=[r for r in prov.get('episode_results',[]) if r.get('episode')==a.episode and r.get('prefix_actions_file')==a.prefix_actions.name and r.get('prefix_actions_file_sha256')==file_sha(a.prefix_actions) and r.get('action_sha256')==action_sha256(x)]
    if prov.get('status')!='COMPLETED_CONTACT_COVERAGE_SCREEN' or len(matches)!=1 or x.ndim!=2 or x.shape[1]!=7: raise ValueError('screen-bound prefix failure')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()['libero_spatial'](); task=suite.get_task(0)
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    np.random.seed(a.seed); env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,camera_depths=True,render_gpu_device_id=-1)
    client=None
    try:
        if prov.get('bddl_sha256')!=file_sha(bddl) or prov.get('compiled_model_xml_sha256')!=hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest(): raise ValueError('environment provenance failure')
        client=Client(a.port,a.noise_seed,True); env.seed(a.seed); env.reset(); obs=env.set_init_state(suite.get_task_init_states(0)[a.episode])
        for _ in range(10): obs,*_=env.step([0.]*6+[-1.])
        for u in x: obs,*_=env.step(u.tolist())
        base=client.infer(obs,task.language)[:a.horizon].astype(np.float32)
        if base.shape!=(8,7): raise ValueError('policy chunk contract')
        rows=[]
        transforms=[('policy',lambda z:z),('attenuated',lambda z:np.column_stack([z[:,:6]*.5,z[:,6]])),('hold',lambda z:np.column_stack([np.zeros_like(z[:,:6]),z[:,6]])),('reverse',lambda z:np.column_stack([-z[:,:6]*.5,z[:,6]]))]
        for name,fn in transforms:
            y=np.ascontiguousarray(fn(base),dtype=np.float32); rows.append({'name':name,'actions':y.tolist(),'actions_sha256':action_sha256(y)})
        out={'schema':'wam2repair-counterfactual-action-candidates-v1','status':'PREDECLARED_ACTIONS','scope':'controlled offline candidates; no simulator future/WAM/VLA result','episode':a.episode,'horizon':8,'prefix_actions_path':str(a.prefix_actions),'prefix_actions_sha256':action_sha256(x),'prefix_provenance_sha256':file_sha(a.provenance),'policy_chunk_sha256':action_sha256(base),'policy_server_metadata':client.metadata,'transforms':['policy','spatial_x0.5','spatial_zero','spatial_x-0.5'],'candidates':rows}
        a.out.mkdir(parents=True); (a.out/'manifest.json').write_text(json.dumps(out,indent=2)); print(json.dumps({'status':out['status'],'candidate_shas':[r['actions_sha256'] for r in rows]}))
    finally:
        if client: client.ws.close()
        env.close()
if __name__=='__main__': main()
