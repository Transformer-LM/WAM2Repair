"""Package the fixed completed grid; refuses partial queues or altered source artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('__WAM2REPAIR_ROOT__/results/wam2repair')
IMPL=Path(__file__).resolve().parent

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    queue=ROOT/'official_grid_20260917'
    manifest=json.loads((queue/'manifest.json').read_text())
    state=json.loads((queue/'queue_state.json').read_text())
    assert state.get('status')=='ALL_COMPLETED', 'Wait for all predeclared trajectories'
    banks=[]
    for job in manifest['jobs']:
        completed=state['jobs'][job['id']]; assert completed['status']=='completed'
        evidence=completed['evidence']
        folder=ROOT/job['run_id']
        assert digest(folder/'result.json')==evidence['result_sha256']
        assert digest(folder/'trajectory.npz')==evidence['trajectory_sha256']
        bank=ROOT/f"bank_official_t{job['task']}_e{job['episode']}_{job['action_mode']}_20260917"
        if bank.exists():
            meta=json.loads((bank/'manifest.json').read_text())
            assert meta['source_sha256']==evidence['trajectory_sha256']
            assert meta['selection']['rule']=='all complete nonoverlapping 16-action windows'
            assert all(row['split']==job['split'] for row in meta['records'])
        else:
            subprocess.run([sys.executable,str(IMPL/'build_aligned_video_bank.py'),'--trajectory',str(folder),
                '--out',str(bank),'--split',job['split']],check=True)
        banks.append(str(bank))
    subprocess.run([sys.executable,str(IMPL/'merge_video_banks.py'),'--out',str(ROOT/'bank_official_full_20260917'),*banks],check=True)

if __name__=='__main__': main()
