"""Clean Stage-1 synthetic repair reimplementation with separate heads."""
import argparse,json,pathlib,time,torch
from torch import nn
def data(n,seed,d):
 g=torch.Generator().manual_seed(seed); pos=torch.randn(n,6,generator=g)*.15; rot=torch.randn(n,6,generator=g)*.2; rel=pos[:,:3]-pos[:,3:]+torch.randn(n,3,generator=g)*.01; c=(torch.sigmoid(pos[:,0]*4+pos[:,3]*2)>.5).float(); s=((pos[:,1]-pos[:,4]).abs()>.18).float(); y=torch.cat([pos,rot,rel],1); x=y.clone(); x[:,:6]+=torch.randn(n,6,generator=g)*.03; x[:,6:9]+=torch.tensor([.05,-.03,.02]); return x.to(d),y.to(d),c.to(d),s.to(d)
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.h=nn.Sequential(nn.Linear(15,128),nn.GELU(),nn.Linear(128,128),nn.GELU()); self.reg=nn.Linear(128,15); self.c=nn.Linear(128,1); self.s=nn.Linear(128,1)
 def forward(self,x): z=self.h(x); return self.reg(z),self.c(z).squeeze(1),self.s(z).squeeze(1)
def sc(r,c,s,y,yc,ys): return {'pose_rmse':float((r[:,:6]-y[:,:6]).pow(2).mean().sqrt()),'relative_pose_rmse':float((r[:,6:9]-y[:,6:9]).pow(2).mean().sqrt()),'contact_acc':float(((c>0)==(yc>0.5)).float().mean()),'slip_acc':float(((s>0)==(ys>0.5)).float().mean())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--steps',type=int,default=2500); ap.add_argument('--train',type=int,default=2048); ap.add_argument('--val',type=int,default=512); ap.add_argument('--seed',type=int,default=19); a=ap.parse_args(); torch.manual_seed(a.seed); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); x,y,c,s=data(a.train,a.seed,d); xv,yv,cv,sv=data(a.val,a.seed+1,d); m=Net().to(d); o=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=1e-4)
 def L(xx,yy,cc,ss):
  r,cl,sl=m(xx); return (r-yy).pow(2).mean()+5.0*nn.functional.binary_cross_entropy_with_logits(cl,cc)+5.0*nn.functional.binary_cross_entropy_with_logits(sl,ss)
 with torch.no_grad(): r,cl,sl=m(xv); raw=sc(xv,torch.zeros_like(cv),torch.zeros_like(sv),yv,cv,sv); initial=float(L(xv,yv,cv,sv))
 for _ in range(a.steps): o.zero_grad(set_to_none=True); z=L(x,y,c,s); z.backward(); o.step()
 with torch.no_grad(): r,cl,sl=m(xv); final=sc(r,cl,sl,yv,cv,sv); final_loss=float(L(xv,yv,cv,sv))
 passed=final_loss<initial and final['pose_rmse']<raw['pose_rmse'] and final['relative_pose_rmse']<raw['relative_pose_rmse'] and final['contact_acc']>=.9 and final['slip_acc']>=.9; q={'schema':'wam2repair-stage1-v2','status':'PASS' if passed else 'FAIL','claim_scope':'synthetic corruption repair only; no WAM/VLA claim','device':str(d),'seed':a.seed,'train_samples':a.train,'val_samples':a.val,'steps':a.steps,'raw_val':raw,'initial_val_loss':initial,'final_val':final,'final_val_loss':final_loss,'created_unix':time.time()}; p=pathlib.Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(q,indent=2)); print(json.dumps(q,indent=2)); raise SystemExit(0 if passed else 2)
if __name__=='__main__': main()
