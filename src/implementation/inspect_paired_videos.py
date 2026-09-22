"""CPU-only raw/copy diagnostics and scientific comparison panels; not physics labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

p=argparse.ArgumentParser(); p.add_argument('--bank',type=Path,required=True); p.add_argument('--wam',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
manifest=json.loads((a.bank/'manifest.json').read_text()); generated=json.loads((a.wam/'result.json').read_text())
assert generated['manifest_sha256']==hashlib.sha256((a.bank/'manifest.json').read_bytes()).hexdigest()
predictions={r['id']:r for r in generated['records']}; rows=[]
for r in manifest['records']:
    prediction=predictions[r['id']]
    for path,digest in [(a.bank/r['target'],r['target_sha256']),(a.wam/prediction['prediction'],prediction['sha256'])]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    with np.load(a.bank/r['target'],allow_pickle=False) as data:
        gt=data['video']; contact=data['contact']; success=data['success']
    with np.load(a.wam/prediction['prediction'],allow_pickle=False) as data: raw=data['frames']
    assert raw.shape==gt.shape
    raw_mae=float(np.abs(raw[1:].astype(float)-gt[1:]).mean()/255)
    copy_mae=float(np.abs(gt[:1].astype(float)-gt[1:]).mean()/255)
    panels=np.concatenate([np.concatenate([raw[t],gt[t]],axis=1) for t in range(len(gt))],axis=0)
    Image.fromarray(panels).save(a.out/(r['id']+'_raw_gt.png'))
    rows.append({'id':r['id'],'group':r['group'],'raw_mae':raw_mae,'copy_current_mae':copy_mae,
        'raw_relative_to_copy':raw_mae/copy_mae-1 if copy_mae else None,
        'gt_contact_sampled_frames':int(contact[:,:-1,-1].any(axis=1).sum()),
        'episode_success':r['episode_success'],'window_success':bool(success.any())})
report={'scope':'training-only raw prediction diagnostic, not repair or physics correction evidence',
    'resolution':[224,448],'panel_columns':['raw_wam','factual_gt'],'rows':rows,
    'mean_raw_mae':float(np.mean([r['raw_mae'] for r in rows])),
    'mean_copy_current_mae':float(np.mean([r['copy_current_mae'] for r in rows]))}
(a.out/'result.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
