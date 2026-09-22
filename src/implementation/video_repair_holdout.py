"""Episode-isolated repair/no-WAM pilot. Test is evaluated only after validation selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from video_repair_overfit import UnifiedVideoRepair, tensor_video


def main():
    p=argparse.ArgumentParser(); p.add_argument('--bank',type=Path,required=True); p.add_argument('--wam',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--height',type=int,choices=[112,224],default=112)
    p.add_argument('--sanity-only',action='store_true',help='Train-only three-window capacity gate; no validation/test evaluation')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False); torch.set_num_threads(4)
    manifest=json.loads((a.bank/'manifest.json').read_text()); generated=json.loads((a.wam/'result.json').read_text())
    assert generated['manifest_sha256']==hashlib.sha256((a.bank/'manifest.json').read_bytes()).hexdigest()
    predictions={r['id']:r for r in generated['records']}
    assert generated['status']=='GENERATED_ALIGNED'
    assert len(predictions)==len(generated['records']), 'Duplicate prediction IDs'
    assert set(predictions)=={r['id'] for r in manifest['records']}, 'Incomplete prediction coverage'
    rows=manifest['records']; groups={name:{r['group'] for r in rows if r['split']==name} for name in ['train','validation','test']}
    assert all(groups.values()),groups
    assert not (groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    if a.sanity_only:
        rows=[r for r in rows if r['split']=='train'][:3]
    convert=lambda x:tensor_video(x,size=(a.height,a.height*2))
    raw=[]; current=[]; actions=[]; proprio=[]; target=[]
    for r in rows:
        pr=predictions[r['id']]; paths=[(a.bank/r['input'],r['input_sha256']),(a.bank/r['target'],r['target_sha256']),(a.wam/pr['prediction'],pr['sha256'])]
        for path,digest in paths: assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,str(path)
        with np.load(paths[0][0],allow_pickle=False) as d:
            assert set(d.files)=={'current_rgb','proprio','actions'}
            current.append(convert(d['current_rgb'][None])[:,0]); actions.append(torch.from_numpy(d['actions'].copy()).float()); proprio.append(torch.from_numpy(d['proprio'].copy()).float())
        with np.load(paths[1][0],allow_pickle=False) as d: target.append(convert(d['video']))
        with np.load(paths[2][0],allow_pickle=False) as d: raw.append(convert(d['frames']))
    raw,current,actions,proprio,target=[torch.stack(x).cuda() for x in [raw,current,actions,proprio,target]]
    indices={name:torch.tensor([i for i,r in enumerate(rows) if r['split']==name],device='cuda') for name in groups}
    report={'scope':'episode-held-out image-error pilot; not physical or VLA gain evidence','seed':17,'steps':a.steps,
        'groups':{k:sorted(v) for k,v in groups.items()},'resolution':[a.height,a.height*2],
        'evaluation_status':'exploratory follow-up on previously inspected test episodes; not a fresh confirmatory test',
        'selection':'validation future-pixel MAE, every 50 steps; no test selection',
        'bank_manifest_sha256':hashlib.sha256((a.bank/'manifest.json').read_bytes()).hexdigest(),
        'wam_result_sha256':hashlib.sha256((a.wam/'result.json').read_bytes()).hexdigest(),
        'model_code_sha256':hashlib.sha256((Path(__file__).parent/'video_repair_overfit.py').read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'methods':{}}
    if a.sanity_only:
        torch.manual_seed(17); model=UnifiedVideoRepair().cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        def objective(pred):
            return (pred[:,:,1:]-target[:,:,1:]).abs().mean()+.1*((pred[:,:,1:]-pred[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])).abs().mean()
        with torch.no_grad(): initial=float(objective(model(raw,current,actions,proprio)))
        for step in range(a.steps):
            loss=objective(model(raw,current,actions,proprio))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite sanity loss')
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            if (step+1)%25==0: print(json.dumps({'sanity_step':step+1,'loss':float(loss.detach())}),flush=True)
        with torch.no_grad(): final=float(objective(model(raw,current,actions,proprio)))
        passed=final<.8*initial
        report.update(status='PASS' if passed else 'FAIL',scope='train-only capacity gate, no physical/control/generalization claim',initial_loss=initial,final_loss=final,sample_ids=[r['id'] for r in rows],peak_memory_bytes=torch.cuda.max_memory_allocated())
        (a.out/'result.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)
        raise SystemExit(0 if passed else 2)
    models={}
    for mode in ['repair','no_wam']:
        torch.manual_seed(17); model=UnifiedVideoRepair().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        source=raw if mode=='repair' else current[:,:,None].expand_as(raw)
        generator=torch.Generator(device='cuda').manual_seed(17)
        best=float('inf'); best_state=None; best_step=None
        history=[]
        for step in range(a.steps):
            choose=indices['train'][torch.randint(len(indices['train']),(min(4,len(indices['train'])),),generator=generator,device='cuda')]
            pred=model(source[choose],current[choose],actions[choose],proprio[choose])
            loss=(pred[:,:,1:]-target[choose,:,1:]).abs().mean()+.1*((pred[:,:,1:]-pred[:,:,:-1])-(target[choose,:,1:]-target[choose,:,:-1])).abs().mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            if (step+1)%50==0:
                total=0.; count=0
                with torch.no_grad():
                    for batch in indices['validation'].split(4):
                        val=model(source[batch],current[batch],actions[batch],proprio[batch])
                        total+=float((val[:,:,1:]-target[batch,:,1:]).abs().mean())*len(batch); count+=len(batch)
                value=total/count; history.append({'step':step+1,'validation_mae':value,'training_batch_loss':float(loss.detach())})
                if value<best:
                    best=value; best_step=step+1; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                print(json.dumps({'method':mode,**history[-1]}),flush=True)
        assert best_state is not None
        model.load_state_dict(best_state); model.eval(); models[mode]=model
        torch.save({'model':best_state,'best_step':best_step,'best_val':best},a.out/(mode+'.pt'))
        report['methods'][mode]={'best_step':best_step,'best_val':best,'history':history}
    # Test starts only after both checkpoints have been chosen and saved.
    per_sample=[]
    for batch in indices['test'].split(4):
        cur=current[batch,:,None].expand_as(raw[batch]); gt=target[batch]
        with torch.no_grad():
            pred={'raw':raw[batch],'copy_current':cur,'anchored':.25*raw[batch]+.75*cur,
                'repair':models['repair'](raw[batch],current[batch],actions[batch],proprio[batch]),
                'no_wam':models['no_wam'](cur,current[batch],actions[batch],proprio[batch])}
            identity=models['repair'](gt,current[batch],actions[batch],proprio[batch])
        motion=((gt[:,:,1:]-cur[:,:,1:]).abs().mean(1,keepdim=True)>.03).float()
        for local,index in enumerate(batch.tolist()):
            r=rows[index]; item={'id':r['id'],'group':r['group'],'episode_success':r['episode_success'],
                'action_mode':r.get('source_action_mode','unknown'),'metrics':{}}
            for name,x in pred.items():
                error=(x[local,:,1:]-gt[local,:,1:]).abs()
                temporal=((x[local,:,1:]-x[local,:,:-1])-(gt[local,:,1:]-gt[local,:,:-1])).abs().mean()
                denom=float(motion[local].sum())*3
                item['metrics'][name]={'mae':float(error.mean()),'temporal_mae':float(temporal),
                    'changed_region_mae':float((error*motion[local]).sum())/denom if denom else None}
            item['oracle_clean_identity_mae']=float((identity[local,:,1:]-gt[local,:,1:]).abs().mean())
            item['repair_edit_mae']=float((pred['repair'][local,:,1:]-pred['raw'][local,:,1:]).abs().mean())
            item['repair_minus_raw_mae']=item['metrics']['repair']['mae']-item['metrics']['raw']['mae']
            video_path=a.out/(r['id']+'_predictions.npz')
            np.savez_compressed(video_path,**{name:x[local].permute(1,2,3,0).cpu().numpy() for name,x in pred.items()},
                oracle_clean_repaired=identity[local].permute(1,2,3,0).cpu().numpy())
            item['prediction_file']=video_path.name
            item['prediction_sha256']=hashlib.sha256(video_path.read_bytes()).hexdigest()
            panels=[]
            for t in range(1,5):
                panels.append(np.concatenate([(x[local,:,t].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8) for x in [pred['raw'],pred['repair'],gt]],axis=1))
            Image.fromarray(np.concatenate(panels,axis=0)).save(a.out/(r['id']+'_raw_repair_gt.png'))
            per_sample.append(item)
    grouped=[]
    for group in sorted(groups['test']):
        subset=[r for r in per_sample if r['group']==group]
        grouped.append({'group':group,'windows':len(subset),'methods':{name:float(np.mean([r['metrics'][name]['mae'] for r in subset])) for name in per_sample[0]['metrics']}})
    # Descriptive slices only; these do not certify physical correctness or tune the model.
    slices={}
    for name,subset in [('raw_low_pixel_error',[r for r in per_sample if r['metrics']['raw']['mae']<=.03]),
                        ('actual_failed_episode',[r for r in per_sample if not r['episode_success']])]:
        summaries=[]
        for group in sorted({r['group'] for r in subset}):
            members=[r for r in subset if r['group']==group]
            summaries.append({'group':group,'windows':len(members),
                'repair_minus_raw_mae':float(np.mean([r['repair_minus_raw_mae'] for r in members])),
                'pixel_error_increase_fraction':float(np.mean([r['repair_minus_raw_mae']>0 for r in members]))})
        slices[name]={'windows':len(subset),'episodes':summaries,
            'episode_macro_repair_minus_raw_mae':float(np.mean([r['repair_minus_raw_mae'] for r in summaries])) if summaries else None}
    report.update(status='EVALUATED',test_samples=per_sample,episode_aggregates=grouped,
        preservation_slices=slices,raw_low_pixel_error_threshold=.03,
        episode_macro_mae={name:float(np.mean([g['methods'][name] for g in grouped])) for name in per_sample[0]['metrics']},
        limitations=['one training seed','small number of independent test episodes','pixel metrics do not establish physical correction',
            'oracle identity is a stress test, not deployable input','failed-future semantics require visual review',
            'no closed-loop repair performance claim'])
    (a.out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':'EVALUATED','episode_macro_mae':report['episode_macro_mae']}),flush=True)


if __name__=='__main__': main()
