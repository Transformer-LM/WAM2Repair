import hashlib
import json
from pathlib import Path
import numpy as np

root = Path('__WAM2REPAIR_ROOT__')
src = root / 'results/wam2repair/pi05_object_contact_prefix4_seeded_screen_ep40_47_20260918T156000Z_2305001/episode_044_prefix_actions.npy'
out = root / 'results/wam2repair/e44_precontact_prefix32_20260918T165000Z_2307500.npy'
report = root / 'results/wam2repair/e44_precontact_prefix32_20260918T165000Z_2307500.json'
if out.exists() or report.exists():
    raise SystemExit('refuse overwrite')
x = np.load(src, allow_pickle=False)
if x.shape != (40, 7):
    raise SystemExit(f'bad source shape {x.shape}')
y = np.ascontiguousarray(x[:32], dtype=np.float32)
np.save(out, y)
sha = hashlib.sha256(out.read_bytes()).hexdigest()
report.write_text(json.dumps({'schema':'wam2repair-fixed-prefix-truncation-v1', 'source':str(src), 'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(), 'source_shape':list(x.shape), 'selected_steps':32, 'output':str(out), 'output_sha256':sha, 'reason':'predeclared e44 pre-contact decision state'}, indent=2))
print(json.dumps({'output':str(out),'sha256':sha}))
