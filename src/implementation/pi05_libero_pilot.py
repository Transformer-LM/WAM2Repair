"""One frozen-policy simulator rollout; records actual success, never imagined success."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from collect_physical_bank import Client, snapshot


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--max-steps',type=int,default=220)
    p.add_argument('--port',type=int,default=10098)
    p.add_argument('--checkpoint',default='__PI05_CHECKPOINT__')
    p.add_argument('--resolution',type=int,default=256)
    p.add_argument('--task',type=int,default=0); p.add_argument('--episode',type=int,default=0)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--action-mode',choices=['policy','hold_gripper_open'],default='policy')
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()['libero_spatial'](); task=suite.get_task(a.task)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
        camera_heights=a.resolution,camera_widths=a.resolution,camera_depths=True,render_gpu_device_id=-1)
    client=None
    try:
        np.random.seed(a.seed); env.seed(a.seed); env.reset(); obs=env.set_init_state(suite.get_task_init_states(a.task)[a.episode])
        for _ in range(10): obs,*_=env.step([0.]*6+[-1.])
        objects=sorted(env.env.obj_body_id); owner={}; all_objects={**env.env.objects_dict,**env.env.fixtures_dict}
        for i,name in enumerate(objects):
            for geom in all_objects[name].contact_geoms: owner[env.sim.model.geom_name2id(geom)]=i
        for geom in env.robots[0].gripper.contact_geoms: owner[env.sim.model.geom_name2id(geom)]=len(objects)
        client=Client(a.port); rows=[snapshot(env,obs,objects,owner)]; actions=[]; chunks=[]; latency=[]; successes=[bool(env.check_success())]
        while len(actions)<a.max_steps and not successes[-1]:
            start=time.time(); chunk=client.infer(obs,task.language); latency.append(time.time()-start)
            chunks.append({'start_step':len(actions),'actions':chunk.tolist(),'latency_seconds':latency[-1]})
            for action in chunk[:5]:
                action=action.copy()
                if a.action_mode=='hold_gripper_open': action[-1]=-1.
                obs,reward,done,_=env.step(action.tolist()); actions.append(action.copy())
                rows.append(snapshot(env,obs,objects,owner)); successes.append(bool(env.check_success()))
                if successes[-1] or len(actions)>=a.max_steps: break
            print(json.dumps({'steps':len(actions),'success':successes[-1],'last_infer_seconds':latency[-1]}),flush=True)
        arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
        arrays.update(actions=np.asarray(actions),success=np.asarray(successes))
        np.savez_compressed(out/'trajectory.npz',**arrays)
        frames=arrays['agentview_rgb'][::max(1,len(rows)//12),::-1,::-1]
        sheet=Image.new('RGB',(224*4,224*((len(frames)+3)//4)))
        for i,frame in enumerate(frames): sheet.paste(Image.fromarray(frame).resize((224,224)),(224*(i%4),224*(i//4)))
        sheet.save(out/'rollout.png')
        report={'scope':'single-episode pi05_libero config diagnostic; not benchmark',
            'suite':'libero_spatial','task':a.task,'episode':a.episode,'seed':a.seed,'language':task.language,'objects':objects,
            'action_mode':a.action_mode,
            'checkpoint':a.checkpoint,'port':a.port,'render_resolution':a.resolution,
            'max_steps':a.max_steps,'steps':len(actions),'success':successes[-1],
            'gripper_contact_frames':int(arrays['contact_matrix'][:,:-1,-1].any(axis=1).sum()),
            'chunks':chunks,'slip':'unknown (-1)','image_convention':'raw simulator; policy receives 180-degree rotation',
            'status':'EXECUTED','split':'diagnostic_only'}
        (out/'result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='chunks'}),flush=True)
    finally:
        if client: client.ws.close()
        env.close()


if __name__=='__main__': main()
