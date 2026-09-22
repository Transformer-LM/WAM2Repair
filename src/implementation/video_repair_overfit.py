"""Real paired-video single-batch learnability gate; explicitly not held-out evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image


class UnifiedVideoRepair(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition=nn.Sequential(nn.Linear(16*7+8,64),nn.SiLU(),nn.Linear(64,32))
        self.encoder=nn.Conv3d(6,32,3,padding=1)
        self.body=nn.Sequential(nn.SiLU(),nn.Conv3d(32,32,3,padding=1),nn.SiLU(),nn.Conv3d(32,3,3,padding=1))
        nn.init.zeros_(self.body[-1].weight); nn.init.zeros_(self.body[-1].bias)

    def forward(self,raw,current,actions,proprio):
        inputs=torch.cat([raw,current[:,:,None].expand(-1,-1,raw.shape[2],-1,-1)],dim=1)
        condition=self.condition(torch.cat([actions.flatten(1),proprio],dim=1))[:,:,None,None,None]
        return (raw+self.body(self.encoder(inputs)+condition)).clamp(0,1)


def tensor_video(x, size=(112,224)):
    tensor=torch.from_numpy(x.copy()).permute(0,3,1,2).float()/255
    return F.interpolate(tensor,size=size,mode='bilinear',align_corners=False).permute(1,0,2,3)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--bank',type=Path,required=True); p.add_argument('--wam',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--steps',type=int,default=300)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False); torch.manual_seed(17); torch.set_num_threads(4)
    manifest=json.loads((a.bank/'manifest.json').read_text()); generated=json.loads((a.wam/'result.json').read_text())
    assert generated['manifest_sha256']==hashlib.sha256((a.bank/'manifest.json').read_bytes()).hexdigest()
    pred_rows={r['id']:r for r in generated['records']}
    raw=[]; current=[]; actions=[]; proprio=[]; target=[]
    for r in manifest['records']:
        ip=a.bank/r['input']; tp=a.bank/r['target']; pr=pred_rows[r['id']]; wp=a.wam/pr['prediction']
        for path,digest in [(ip,r['input_sha256']),(tp,r['target_sha256']),(wp,pr['sha256'])]:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        with np.load(ip,allow_pickle=False) as d:
            assert set(d.files)=={'current_rgb','proprio','actions'}
            current.append(tensor_video(d['current_rgb'][None])[:,0]); actions.append(torch.from_numpy(d['actions'].copy()).float()); proprio.append(torch.from_numpy(d['proprio'].copy()).float())
        with np.load(wp,allow_pickle=False) as d: raw.append(tensor_video(d['frames']))
        with np.load(tp,allow_pickle=False) as d: target.append(tensor_video(d['video']))
    raw,current,actions,proprio,target=[torch.stack(x).cuda() for x in [raw,current,actions,proprio,target]]
    model=UnifiedVideoRepair().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    # GT determines supervision weights only; never model inputs.
    weight=1+4*((target-raw).abs().mean(1,keepdim=True)>.03).float()
    def loss(pred):
        return ((pred[:,:,1:]-target[:,:,1:]).abs()*weight[:,:,1:]).mean()+.1*((pred[:,:,1:]-pred[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])).abs().mean()
    with torch.no_grad(): initial=float(loss(model(raw,current,actions,proprio)))
    history=[]
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True); pred=model(raw,current,actions,proprio); value=loss(pred)
        if not torch.isfinite(value): raise RuntimeError('Nonfinite loss')
        value.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if step%50==0:
            history.append({'step':step,'loss':float(value.detach())}); print(json.dumps(history[-1]),flush=True)
    with torch.no_grad():
        repaired=model(raw,current,actions,proprio); final=float(loss(repaired))
        metric=lambda x:float((x[:,:,1:]-target[:,:,1:]).abs().mean())
        metrics={'raw_mae':metric(raw),'copy_current_mae':metric(current[:,:,None].expand_as(raw)),
            'anchored_mae':metric(.25*raw+.75*current[:,:,None]),'repair_mae':metric(repaired)}
    status='PASS' if final<.8*initial and metrics['repair_mae']<metrics['raw_mae'] else 'FAIL'
    report={'status':status,'scope':'single-batch overfit on real WAM/real simulator video; no generalization or physical/control claim',
        'seed':17,'steps':a.steps,'samples':len(raw),'episode_groups':sorted({r['group'] for r in manifest['records']}),
        'resolution':[112,224],'initial_loss':initial,'final_loss':final,'metrics':metrics,'history':history,
        'input_contract':['WAM video','current RGB','executed action sequence','current proprio'],
        'missing_evidence':['held-out evaluation','contact-rich trajectories','physical error correction','VLA improvement']}
    torch.save({'model':model.state_dict(),'report':report},a.out/'repair_sanity.pt')
    (a.out/'result.json').write_text(json.dumps(report,indent=2))
    panels=[]
    for sample in range(len(raw)):
        panels.append(np.concatenate([np.concatenate([(x[sample,:,t].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8) for x in [raw,repaired,target]],axis=1) for t in range(1,5)],axis=0))
    for i,panel in enumerate(panels): Image.fromarray(panel).save(a.out/f'sample_{i}_raw_repaired_gt.png')
    print(json.dumps(report),flush=True)
    raise SystemExit(0 if status=='PASS' else 2)


if __name__=='__main__': main()
