import json
import os
import subprocess
from pathlib import Path

root = Path('__WAM2REPAIR_ROOT__')
pair = root / 'results/wam2repair/pilot_v2_rank_e42_c00_20260918T161000Z_2307100/manifest.json'
out = root / 'results/wam2repair/pilot_v2_wam_rank_e42_c00_20260918T162000Z_2307200'
meta = json.loads(pair.read_text())
if out.exists():
    raise SystemExit(f'refuse pre-existing output: {out}')
env = os.environ.copy()
env.update({
    'CUDA_VISIBLE_DEVICES': '3',
    'PYTHONPATH': f"{root}/workspace/FastWAM:{root}/workspace/policy-relevant-imagined-state-repair/implementation",
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled',
    'DIFFSYNTH_MODEL_BASE_PATH': str(root / 'models/fastwam'), 'DIFFSYNTH_SKIP_DOWNLOAD': 'true',
    'XDG_CACHE_HOME': str(root / 'cache/wam2repair-fastwam'), 'TMPDIR': str(root / 'tmp/wam2repair-fastwam'),
})
cmd = [str(root / 'conda/envs/fastwam-py311/bin/python'), '-u',
       str(root / 'workspace/policy-relevant-imagined-state-repair/implementation/generate_wam_allowlisted.py'),
       '--input', str(pair.parent / meta['input']), '--input-sha256', meta['input_sha256'],
       '--language', meta['language'], '--out', str(out), '--steps', '2', '--seed', '7']
subprocess.run(cmd, env=env, check=True)
