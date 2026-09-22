"""One-sample full video generation; no claims of aligned future accuracy."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def generate_bank(model,cfg,root,bank,out,steps,provenance):
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    manifest=json.loads((bank/'manifest.json').read_text())
    assert manifest['horizon']==16 and manifest['frame_offsets']==[0,4,8,12,16]
    processor=instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(root/'results/israc/P027C_LIBERO_COMBINED_NORM_STATS.json')))
    records=[]
    provenance.update(status='RUNNING_ALIGNED',bank=str(bank),manifest_sha256=hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),records=records,
        scope='action-aligned factual video prediction; factual targets never opened by generator')
    (out/'progress.json').write_text(json.dumps(provenance,indent=2,default=str))
    for row in manifest['records']:
        ip=bank/row['input']; assert hashlib.sha256(ip.read_bytes()).hexdigest()==row['input_sha256']
        with np.load(ip,allow_pickle=False) as d:
            assert set(d.files)=={'current_rgb','proprio','actions'}
            image=torch.from_numpy(d['current_rgb'].copy()).permute(2,0,1)[None].to('cuda',torch.bfloat16)/127.5-1
            norm=processor.normalizer.forward({'action':{'default':torch.from_numpy(d['actions'].copy())[None]},
                'state':{'default':torch.from_numpy(d['proprio'].copy())[None]}})
        prompt=DEFAULT_PROMPT.format(task=row['language'])
        cached=root/'cache/fastwam/text_embeds/p027c_libero'/(hashlib.sha256(prompt.encode()).hexdigest()+'.t5_len128.wan22ti2v5b.pt')
        text=torch.load(cached,map_location='cpu',weights_only=False)
        context=text['context'].clone(); context[~text['mask'].bool()]=0
        with torch.inference_mode():
            pred=model.infer_joint(prompt=None,input_image=image,num_video_frames=5,action_horizon=16,
                action=norm['action']['default'].to('cuda',torch.bfloat16),
                proprio=norm['state']['default'].to('cuda',torch.bfloat16),context=context,
                context_mask=torch.ones_like(text['mask'],dtype=torch.bool),num_inference_steps=steps,
                seed=7,rand_device='cpu',tiled=False,test_action_with_infer_action=False)
        frames=np.stack([np.asarray(x) for x in pred['video']]); assert frames.shape==(5,224,448,3)
        path=out/(row['id']+'_wam.npz'); np.savez_compressed(path,frames=frames)
        Image.fromarray(np.concatenate(list(frames),axis=0)).save(out/(row['id']+'_wam.png'))
        records.append({'id':row['id'],'prediction':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        (out/'progress.json').write_text(json.dumps(provenance,indent=2,default=str))
        print('GENERATED_ALIGNED',row['id'],flush=True)
    provenance.update(status='GENERATED_ALIGNED',bank=str(bank),manifest_sha256=hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),records=records,
        scope='action-aligned factual video prediction; policy/splits inherited from bank; factual targets never opened by generator')
    (out/'result.json').write_text(json.dumps(provenance,indent=2,default=str))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--steps',type=int,default=20)
    p.add_argument('--action-controls',action='store_true')
    p.add_argument('--bank',type=Path)
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    root=Path('__WAM2REPAIR_ROOT__'); torch.manual_seed(7)
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    if a.bank:
        bank_manifest=json.loads((a.bank/'manifest.json').read_text())
        tasks={row['language'] for row in bank_manifest['records']}
        assert tasks, 'Empty bank'
    else:
        meta=json.loads((root/'results/policy-relevant-imagined-state-repair/e1/prepared_goal03.json').read_text())
        tasks={meta['records'][0]['instruction']}
    for task in tasks:
        prompt=DEFAULT_PROMPT.format(task=task)
        cached=root/'cache/fastwam/text_embeds/p027c_libero'/(hashlib.sha256(prompt.encode()).hexdigest()+'.t5_len128.wan22ti2v5b.pt')
        if not cached.is_file(): raise FileNotFoundError(cached)
    with initialize_config_dir(version_base=None,config_dir=str(root/'workspace/FastWAM/configs')):
        cfg=compose(config_name='sim_libero',overrides=['task=p027c_libero_videoac_learn500',
            'model.skip_dit_load_from_pretrain=false','model.load_text_encoder=false',
            'model.action_dit_pretrained_path=__FASTWAM_ACTION_DIT_CHECKPOINT__'])
    model=instantiate(cfg.model,model_dtype=torch.bfloat16,device='cuda:0').eval()
    assert model.model_paths['video_dit'] != 'SKIPPED_PRETRAIN', model.model_paths
    ckpt=root/'workspace/FastWAM/runs/policy_relevant_imagined_state_repair_5000/checkpoints/weights/step_005000_trainable_only.pt'
    payload=torch.load(ckpt,map_location='cpu',weights_only=False)
    print('CHECKPOINT_KEYS',list(payload),flush=True)
    if 'state_dict' not in payload: raise ValueError('Expected trainable-only state_dict')
    weights=payload['state_dict']; incompat=model.load_state_dict(weights,strict=False)
    assert not incompat.unexpected_keys, incompat.unexpected_keys
    report={'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        'loaded_keys':list(weights),'missing_key_count':len(incompat.missing_keys),'model_paths':model.model_paths,
        'steps':a.steps,'seed':7,'scope':'generation diagnostic; frame/action temporal alignment unverified'}
    if a.bank:
        generate_bank(model,cfg,root,a.bank,out,a.steps,report)
        return
    data=np.load(root/'results/policy-relevant-imagined-state-repair/e1/prepared_goal03.npz')
    meta=json.loads((root/'results/policy-relevant-imagined-state-repair/e1/prepared_goal03.json').read_text())
    txt=torch.load(cached,map_location='cpu',weights_only=False)
    print('TEXT_KEYS',list(txt),flush=True)
    context=txt['context'] if 'context' in txt else txt['embedding']
    mask=txt['mask'].bool(); context=context.clone(); context[~mask]=0
    # Match the existing training dataset's padding convention exactly.
    mask=torch.ones_like(mask)
    rgb=np.concatenate([np.asarray(Image.fromarray(data['task0_'+cam]).resize((224,224),Image.Resampling.BILINEAR)) for cam in ['primary','wrist']],axis=1)
    image=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to('cuda',torch.bfloat16)/127.5-1
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    processor=instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(root/'results/israc/P027C_LIBERO_COMBINED_NORM_STATS.json')))
    norm=processor.normalizer.forward({'action':{'default':torch.from_numpy(data['task0_action'])[None]},'state':{'default':torch.from_numpy(data['task0_state'])[None]}})
    (out/'provenance.json').write_text(json.dumps(report,indent=2,default=str))
    start=time.time()
    with torch.inference_mode():
        result=model.infer_joint(prompt=None,input_image=image,num_video_frames=5,action_horizon=8,
            action=norm['action']['default'].to('cuda',torch.bfloat16),
            proprio=norm['state']['default'].to('cuda',torch.bfloat16),
            context=context,context_mask=mask,num_inference_steps=a.steps,seed=7,
            rand_device='cpu',tiled=False,test_action_with_infer_action=False)
    frames=np.stack([np.asarray(x) for x in result['video']])
    assert np.isfinite(frames).all()
    np.savez_compressed(out/'video.npz',frames=frames,current=rgb)
    Image.fromarray(np.concatenate(list(frames),axis=0)).save(out/'frames.png')
    report.update(elapsed_seconds=time.time()-start,frames_shape=list(frames.shape),status='GENERATED_REQUIRES_VISUAL_REVIEW')
    (out/'result.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,default=str),flush=True)
    if a.action_controls:
        controls={}
        for name,act in [('zero_normalized',torch.zeros_like(norm['action']['default'])),
                         ('reversed_order',norm['action']['default'].flip(1))]:
            with torch.inference_mode():
                alt=model.infer_joint(prompt=None,input_image=image,num_video_frames=5,action_horizon=8,
                    action=act.to('cuda',torch.bfloat16),proprio=norm['state']['default'].to('cuda',torch.bfloat16),
                    context=context,context_mask=mask,num_inference_steps=a.steps,seed=7,
                    rand_device='cpu',tiled=False,test_action_with_infer_action=False)
            alt_frames=np.stack([np.asarray(x) for x in alt['video']])
            np.savez_compressed(out/(name+'.npz'),frames=alt_frames)
            Image.fromarray(np.concatenate(list(alt_frames),axis=0)).save(out/(name+'.png'))
            controls[name]={'future_pixel_mse_vs_original':float(np.mean((alt_frames[1:].astype(float)-frames[1:].astype(float))**2)),
                'input_action_l2_vs_original':float(torch.linalg.vector_norm(act-norm['action']['default']))}
        controls['scope']='sensitivity only; normalized zero is not physical no-op; no factual counterfactual correctness claim'
        (out/'action_controls.json').write_text(json.dumps(controls,indent=2))
        print(json.dumps(controls),flush=True)


if __name__=='__main__': main()
