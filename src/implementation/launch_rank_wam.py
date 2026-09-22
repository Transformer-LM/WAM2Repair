import json
import os
import subprocess
import sys
from pathlib import Path

root = Path('__WAM2REPAIR_ROOT__')
if len(sys.argv) != 3:
    raise SystemExit('usage: launcher PAIR_MANIFEST OUTPUT_DIR')
pair, out = map(Path, sys.argv[1:])
if not pair.is_absolute() or not out.is_absolute() or not str(pair).startswith(str(root)) or not str(out).startswith(str(root)):
    raise SystemExit('paths must be absolute personal-directory paths')
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
