"""Export all test windows with anonymized raw/repair columns and a factual reference."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image,ImageDraw

root=Path('__WAM2REPAIR_ROOT__/results/wam2repair')
run=root/'video_holdout_20260917T050816Z'; bank=root/'bank_official_full_20260917'
out=run/'blind_review'; out.mkdir(exist_ok=False)
result=json.loads((run/'result.json').read_text()); manifest=json.loads((bank/'manifest.json').read_text())
targets={r['id']:r for r in manifest['records']}; rng=random.Random(417)
rows=list(result['test_samples']); rng.shuffle(rows); key=[]
for i,row in enumerate(rows):
    name=f'case_{i:03d}'; path=run/row['prediction_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['prediction_sha256']
    target=targets[row['id']]; target_path=bank/target['target']
    assert hashlib.sha256(target_path.read_bytes()).hexdigest()==target['target_sha256']
    with np.load(target_path,allow_pickle=False) as data: gt=data['video']
    order=['raw','repair']; rng.shuffle(order)
    with np.load(path,allow_pickle=False) as data:
        columns=[(data[method]*255).clip(0,255).astype(np.uint8) for method in order]
    # Upsampling is display-only; outputs remain 112x224, GT is resized identically.
    sheet=Image.new('RGB',(672,580),'white'); draw=ImageDraw.Draw(sheet)
    draw.text((5,4),name+' | A (left) | B (middle) | factual reference (right)',fill='black')
    for t in range(5):
        for col,video in enumerate(columns+[gt]):
            frame=Image.fromarray(video[t]).resize((224,112),Image.Resampling.BILINEAR)
            sheet.paste(frame,(col*224,20+t*112))
    sheet.save(out/(name+'.png'))
    key.append({'case':name,'id':row['id'],'A':order[0],'B':order[1],
        'episode_success':row['episode_success'],'low_raw_pixel_error':row['metrics']['raw']['mae']<=.03,
        'group':row['group']})
(run/'blind_review_key.json').write_text(json.dumps(key,indent=2))
(out/'README.txt').write_text('Each case: A, B, factual reference. Rows are times 0/4/8/12/16. Review visual consistency, contact/pose changes and possible invented grasp/success; mark unknown when pixels are insufficient. Methods and outcomes are withheld. AI review is NOT human annotation or geometry-certified physics evaluation.')
print(json.dumps({'cases':len(key),'out':str(out),'key':str(run/'blind_review_key.json')}))
