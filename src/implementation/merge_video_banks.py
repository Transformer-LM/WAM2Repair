"""Combine explicitly selected factual banks without changing episode split membership."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('banks',type=Path,nargs='+'); a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
result={'schema':'w2r-video-paired-v1','horizon':16,'frame_offsets':[0,4,8,12,16],'records':[],'sources':[]}
membership={}; seen=set()
for bank in a.banks:
    source=bank/'manifest.json'; meta=json.loads(source.read_text())
    assert meta['horizon']==16 and meta['frame_offsets']==result['frame_offsets']
    result['sources'].append({'bank':str(bank),'manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'policy_checkpoint':meta['policy_checkpoint'],'selection':meta['selection']})
    for r in meta['records']:
        row=dict(r); group=row['group']; split=row['split']
        if group in membership: assert membership[group]==split,'Episode split leakage'
        membership[group]=split
        row['id']=bank.name+'_'+r['id']; assert row['id'] not in seen; seen.add(row['id'])
        row['source_action_mode']=meta['selection']['source_action_mode']
        for role in ['input','target']:
            src=bank/r[role]; digest=hashlib.sha256(src.read_bytes()).hexdigest(); assert digest==r[role+'_sha256']
            dst=a.out/(row['id']+'_'+role+'.npz'); shutil.copy2(src,dst); row[role]=dst.name
        result['records'].append(row)
result['group_splits']=membership
(a.out/'manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'samples':len(result['records']),'group_splits':membership}),flush=True)
