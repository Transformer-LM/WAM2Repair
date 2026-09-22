"""Stage-1 repair v3: residual model, correct raw baseline, deterministic bank."""
import argparse,json,pathlib,time,torch
from torch import nn
def bank(n,seed,d):
 g=torch.Generator().manual_seed(seed); pos=torch.randn(n,6,generator=g)*.15; rot=torch.randn(n,6,generator=g)*.2; rel=pos[:,:3]-pos[:,3:]+torch.randn(n,3,generator=g)*.01; c=(torch.sigmoid(pos[:,0]*4+pos[:,3]*2)>.5).float(); s=((pos[:,1]-pos[:,4]).abs()>.18).float(); y=torch.cat([pos,rot,rel,c[:,None],s[:,None]],1); x=y.clone(); x[:,:6]+=torch.randn(n,6,generator=g)*.06; x[:,6:9]+=torch.tensor([.07,-.04,.03]); x[:,15]=1-x[:,15]; x[:,16]=1-x[:,16]; return x.to(d),y.to(d)
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.h=nn.Sequential(nn.Linear(17,96),nn.GELU(),nn.Linear(96,96),nn.GELU()); self.r=nn.Linear(96,15); self.c=nn.Linear(96,1); self.s=nn.Linear(96,1)
 def forward(self,x): z=self.h(x); return x[:,:15]+self.r(z),self.c(z).squeeze(1),self.s(z).squeeze(1)
def metrics(r,cl,sl,y): return {'pose_rmse':float((r[:,:6]-y[:,:6]).pow(2).mean().sqrt()),'relative_pose_rmse':float((r[:,6:9]-y[:,6:9]).pow(2).mean().sqrt()),'contact_acc':float(((cl>0)==(y[:,15]>.5)).float().mean()),'slip_acc':float(((sl>0)==(y[:,16]>.5)).float().mean())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--steps',type=int,default=2000); ap.add_argument('--train',type=int,default=4096); ap.add_argument('--val',type=int,default=1024); ap.add_argument('--seed',type=int,default=23); a=ap.parse_args(); torch.manual_seed(a.seed); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); x,y=bank(a.train,a.seed,d); xv,yv=bank(a.val,a.seed+1,d); m=Net().to(d); o=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
 def L(xx,yy):
  r,c,s=m(xx); return (r-yy[:,:15]).pow(2).mean()+nn.functional.binary_cross_entropy_with_logits(c,yy[:,15])+nn.functional.binary_cross_entropy_with_logits(s,yy[:,16])
 with torch.no_grad(): raw=metrics(xv,xv[:,15]*2-1,xv[:,16]*2-1,yv); best=float('inf'); bestm=None
 for step in range(a.steps):
  o.zero_grad(set_to_none=True); z=L(x,y); z.backward(); o.step()
  if step%50==0:
   with torch.no_grad(): vl=float(L(xv,yv))
   if vl<best: best=vl; bestm={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 if bestm: m.load_state_dict(bestm)
 with torch.no_grad(): r,c,s=m(xv); final=metrics(r,c,s,yv)
 passed=final['pose_rmse']<raw['pose_rmse'] and final['relative_pose_rmse']<raw['relative_pose_rmse'] and final['contact_acc']>=.9 and final['slip_acc']>=.9
 q={'schema':'wam2repair-stage1-v3','status':'PASS' if passed else 'FAIL','claim_scope':'synthetic corruption repair only; no WAM/VLA claim','device':str(d),'seed':a.seed,'train_samples':a.train,'val_samples':a.val,'steps':a.steps,'raw_val':raw,'best_val_loss':best,'final_val':final,'created_unix':time.time()}; p=pathlib.Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(q,indent=2)); print(json.dumps(q,indent=2)); raise SystemExit(0 if passed else 2)
if __name__=='__main__': main()
