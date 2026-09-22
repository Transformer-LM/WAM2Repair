"""Read-only verification of weights; emit marker only after every hash matches."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args()
root=a.root.resolve(strict=True)
manifest=json.loads((root/'transfer_sha256.json').read_text())
for row in manifest:
    path=(root/row['path']).resolve(strict=True)
    if not path.is_relative_to(root): raise ValueError('Path escaped checkpoint directory')
    assert path.stat().st_size==row['bytes'],str(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    assert h.hexdigest()==row['sha256'],str(path)
    print('VERIFIED',row['path'],flush=True)
(root/'TRANSFER_VERIFIED.json').write_text(json.dumps({'status':'VERIFIED','objects':len(manifest)}))
