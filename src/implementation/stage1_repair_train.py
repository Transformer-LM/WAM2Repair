"""Stage 1: small synthetic corruption bank and repair training."""
from __future__ import annotations
import argparse, json, pathlib, time
import torch
from torch import nn

def sample(n, seed, device):
    g=torch.Generator().manual_seed(seed)
    pos=torch.randn(n,6,generator=g)*.15; rot=torch.randn(n,6,generator=g)*.2
    rel=pos[:,:3]-pos[:,3:]+torch.randn(n,3,generator=g)*.01
    c=(torch.sigmoid(pos[:,0]*4+pos[:,3]*2)>.5).float().unsqueeze(1)
    s=((pos[:,1]-pos[:,4]).abs()>.18).float().unsqueeze(1)
    y=torch.cat([pos,rot,rel,c,s],1); x=y.clone()
    x[:,:6]+=torch.randn(n,6,generator=g)*.08; x[:,6:9]+=torch.tensor([.10,-.06,.04])
    x[:,15]=torch.roll(x[:,15],1); x[:,16]=torch.roll(x[:,16],2)
    return x.to(device),y.to(device)
class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.m=nn.Sequential(nn.Linear(17,96),nn.GELU(),nn.Linear(96,96),nn.GELU(),nn.Linear(96,17))
    def forward(self,x): return self.m(x)
def score(p,y):
    return {"pose_rmse":float(torch.mean((p[:,:6]-y[:,:6])**2).sqrt()),"relative_pose_rmse":float(torch.mean((p[:,6:9]-y[:,6:9])**2).sqrt()),"contact_acc":float(((p[:,15]>.5)==(y[:,15]>.5)).float().mean()),"slip_acc":float(((p[:,16]>.5)==(y[:,16]>.5)).float().mean())}
def loss(p,y): return torch.mean((p[:,:15]-y[:,:15])**2)+nn.functional.binary_cross_entropy_with_logits(p[:,15],y[:,15])+nn.functional.binary_cross_entropy_with_logits(p[:,16],y[:,16])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--steps',type=int,default=800); ap.add_argument('--train',type=int,default=512); ap.add_argument('--val',type=int,default=128); ap.add_argument('--seed',type=int,default=17); a=ap.parse_args()
    torch.manual_seed(a.seed); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); x,y=sample(a.train,a.seed,d); xv,yv=sample(a.val,a.seed+1,d); m=Net().to(d); o=torch.optim.AdamW(m.parameters(),lr=2e-3)
    with torch.no_grad(): raw=score(xv,yv); initial=float(loss(m(xv),yv))
    for _ in range(a.steps):
        z=m(x); l=loss(z,y); o.zero_grad(set_to_none=True); l.backward(); o.step()
    with torch.no_grad(): final=score(m(xv),yv); final_loss=float(loss(m(xv),yv))
    passed=final_loss<initial*.5 and final['pose_rmse']<raw['pose_rmse'] and final['relative_pose_rmse']<raw['relative_pose_rmse'] and final['contact_acc']>=.9 and final['slip_acc']>=.9
    q={'schema':'wam2repair-stage1-v1','status':'PASS' if passed else 'FAIL','claim_scope':'synthetic corruption repair only; no WAM/VLA claim','device':str(d),'seed':a.seed,'train_samples':a.train,'val_samples':a.val,'steps':a.steps,'raw_val':raw,'initial_val_loss':initial,'final_val':final,'final_val_loss':final_loss,'created_unix':time.time()}
    p=pathlib.Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(q,indent=2)); print(json.dumps(q,indent=2)); raise SystemExit(0 if passed else 2)
if __name__=='__main__': main()
