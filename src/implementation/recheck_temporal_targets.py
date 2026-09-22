"""Reopen original trajectories and targets, verify actual action/frame correspondence."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

root=Path('__WAM2REPAIR_ROOT__/results/wam2repair')
bank=root/'bank_official_full_20260917'
manifest=json.loads((bank/'manifest.json').read_text())
source_banks={}
for entry in manifest['sources']:
    path=Path(entry['bank']); meta=json.loads((path/'manifest.json').read_text())
    assert hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()==entry['manifest_sha256']
    source_banks[path.name]=(path,meta)
duplicates=defaultdict(list); checked=0
for name,(source_bank,meta) in source_banks.items():
    source=Path(meta['source']); assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
    with np.load(source,allow_pickle=False) as trajectory:
        for row in meta['records']:
            merged=next(r for r in manifest['records'] if r['id']==name+'_'+row['id'])
            start=row['start']; idx=start+np.array([0,4,8,12,16])
            target_path=bank/merged['target']; input_path=bank/merged['input']
            assert hashlib.sha256(target_path.read_bytes()).hexdigest()==merged['target_sha256']
            assert hashlib.sha256(input_path.read_bytes()).hexdigest()==merged['input_sha256']
            duplicates[merged['target_sha256']].append({'id':merged['id'],'split':merged['split']})
            with np.load(target_path,allow_pickle=False) as target, np.load(input_path,allow_pickle=False) as inp:
                np.testing.assert_array_equal(inp['actions'],trajectory['actions'][start:start+16])
                np.testing.assert_array_equal(target['times'],trajectory['sim_time'][idx])
                np.testing.assert_allclose(np.diff(target['times']),.2,rtol=1e-6,atol=1e-8)
                for field,source_field in [('contact','contact_matrix'),('success','success'),('object_pos','object_pos'),('object_rot','object_rot'),('eef_pos','eef_pos'),('eef_rot','eef_rot')]:
                    np.testing.assert_array_equal(target[field],trajectory[source_field][idx])
                for t,j in enumerate(idx):
                    frame=np.concatenate([np.asarray(Image.fromarray(trajectory[cam][j,::-1,::-1]).resize((224,224),Image.Resampling.BILINEAR)) for cam in ['agentview_rgb','robot0_eye_in_hand_rgb']],axis=1)
                    np.testing.assert_array_equal(target['video'][t],frame)
                np.testing.assert_array_equal(inp['current_rgb'],target['video'][0])
            checked+=1
dups=[v for v in duplicates.values() if len(v)>1]
cross=[v for v in dups if len({r['split'] for r in v})>1]
assert not cross,cross
report={'status':'VERIFIED','windows':checked,'manifest_sha256':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),
    'target_duplicate_groups':dups,'cross_split_target_duplicates':cross,
    'checks':['all target frames rebuilt from native trajectory at exact offsets','executed actions exact match','actual times exact match and .2 second frame intervals','contact/success/pose exact match','current input equals factual t0'],
    'scope':'artifact alignment; not WAM prediction correctness or pretrained-data-disjointness'}
(bank/'TEMPORAL_TARGET_VERIFIED.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))
