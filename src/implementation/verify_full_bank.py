"""Independent deterministic checks for the predeclared official pilot bank."""
import hashlib
import json
from pathlib import Path
import re
from collections import Counter,defaultdict

bank=Path('__WAM2REPAIR_ROOT__/results/wam2repair/bank_official_full_20260917')
manifest=json.loads((bank/'manifest.json').read_text())
assert manifest['horizon']==16 and manifest['frame_offsets']==[0,4,8,12,16]
seen=defaultdict(list); groups=defaultdict(set); counts=Counter()
for row in manifest['records']:
    match=re.fullmatch(r'libero_spatial/task([01])/episode([0-3])',row['group']); assert match,row
    task,episode=map(int,match.groups()); expected='train' if episode<2 else 'validation' if episode==2 else 'test'
    assert row['split']==expected,row
    groups[expected].add(row['group']); counts[expected]+=1
    assert row['source_action_mode'] in ['policy','hold_gripper_open']
    for role in ['input','target']:
        path=bank/row[role]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row[role+'_sha256']
    seen[row['input_sha256']].append({'id':row['id'],'split':expected,'group':row['group']})
duplicates=[rows for rows in seen.values() if len(rows)>1]
cross_split=[rows for rows in duplicates if len({r['split'] for r in rows})>1]
assert not cross_split,cross_split
assert dict(counts)=={'train':77,'validation':38,'test':36},counts
assert {k:len(v) for k,v in groups.items()}=={'train':4,'validation':2,'test':2}
assert len(manifest['sources'])==16
for src in manifest['sources']:
    assert src['policy_checkpoint']=='__WAM2REPAIR_ROOT__/models/openpi/official/pi05_libero'
    assert src['selection']['rule']=='all complete nonoverlapping 16-action windows'
report={'status':'VERIFIED','manifest_sha256':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),
    'counts':dict(counts),'groups':{k:sorted(v) for k,v in groups.items()},'duplicate_inputs':duplicates,
    'cross_split_duplicate_inputs':cross_split,'scope':'split, coverage, artifact hashes; policy identity separately evidenced by launch/runtime logs'}
(bank/'SPLIT_VERIFIED.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))
