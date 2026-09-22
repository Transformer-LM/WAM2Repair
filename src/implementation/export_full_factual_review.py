"""CPU-only display of complete saved factual test trajectories, never model predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

p=argparse.ArgumentParser()
p.add_argument('--grid',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
state=json.loads(a.grid.read_text()); report=[]
for name,job in sorted(state['jobs'].items()):
    if '_e3_' not in name: continue
    ev=job['evidence']; path=Path(ev['trajectory'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==ev['trajectory_sha256']
    with np.load(path,allow_pickle=False) as d:
        times=d['sim_time']; count=len(times)
        assert count==len(d['actions'])+1
        cameras={cam:d[cam+'_rgb'] for cam in ['agentview','robot0_eye_in_hand']}
        success=d['success']
        frames=[]
        for t in range(count):
            pair=np.concatenate([cameras[cam][t,::-1,::-1] for cam in ['agentview','robot0_eye_in_hand']],axis=1)
            canvas=Image.new('RGB',(pair.shape[1],pair.shape[0]+32),'white')
            canvas.paste(Image.fromarray(pair),(0,32))
            draw=ImageDraw.Draw(canvas)
            draw.text((4,2),f'FACTUAL ONLY | {name} | step={t} | t={times[t]-times[0]:.2f}s',fill='black')
            draw.text((4,16),f'simulator success={bool(success[t])}; NOT a WAM/repair prediction',fill='black')
            frames.append(canvas)
        durations=[max(10,int(round(float(dt)*1000/10))*10) for dt in np.diff(times)]
        durations.append(durations[-1])
        dest=a.out/(name+'.gif')
        frames[0].save(dest,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False)
        report.append({'id':name,'source_sha256':ev['trajectory_sha256'],'frames':count,'all_frames_including_terminal':True,'end_time_seconds':float(times[-1]-times[0]),'success':bool(d['success'][-1]),'file':dest.name,'display':'raw 256x256 per view, rotated 180deg, GIF palette quantization; no resampling in time'})
(a.out/'manifest.json').write_text(json.dumps({'scope':'full factual context only, no model comparison or physical correction claim','records':report},indent=2))
print(json.dumps(report),flush=True)
