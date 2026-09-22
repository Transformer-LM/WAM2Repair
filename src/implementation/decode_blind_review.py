"""Decode a frozen reviewer response only after the blinded review is complete."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

project=Path(__file__).resolve().parent.parent
report=project/'AI_BLIND_VIDEO_REVIEW_20260917.md'
key_path=project/'artifacts/gpu_gate_20260917/blind_review_key.json'
key={r['case']:r for r in json.loads(key_path.read_text())}; rows=[]
for line in report.read_text(encoding='utf-8').splitlines():
    if not re.match(r'^\| \d{3} \|',line): continue
    fields=[v.strip() for v in line.split('|')[1:-1]]
    case='case_'+fields[0]; meta=key[case]; preference=fields[1]
    winner=meta[preference[0]] if preference.startswith(('A','B')) else 'tie' if preference=='≈' else 'unknown'
    labels=[v.strip() for v in fields[2].split('/')]; assert all(v in ['N','U'] for v in labels)
    rows.append({'case':case,'id':meta['id'],'group':meta['group'],'episode_success':meta['episode_success'],
        'low_raw_pixel_error':meta['low_raw_pixel_error'],'closer':winner,'low_confidence':'低置信' in preference,
        'unsupported_success':{meta['A']:labels[0],meta['B']:labels[1]},'notes':fields[3]})
assert len(rows)==36 and len({r['case'] for r in rows})==36
slices={}
for name,subset in [('all',rows),('failed_episode',[r for r in rows if not r['episode_success']]),('low_raw_error',[r for r in rows if r['low_raw_pixel_error']])]:
    slices[name]={'windows':len(subset),'closer_counts':dict(Counter(r['closer'] for r in subset)),
        'unsupported_success_labels':{method:dict(Counter(r['unsupported_success'][method] for r in subset)) for method in ['raw','repair']}}
out={'scope':'AI visual diagnostic, same-family provisional; not human annotation or physical certification',
    'review_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'key_sha256':hashlib.sha256(key_path.read_bytes()).hexdigest(),
    'labels':{'N':'No clear evidence observed, NOT proof of preservation','U':'Unknown, NOT a negative example'},
    'slices':slices,'rows':rows}
(project/'artifacts/gpu_gate_20260917/blind_review_decoded.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(slices,ensure_ascii=False))
