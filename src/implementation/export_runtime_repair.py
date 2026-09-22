"""Export a deployable repair-only artifact; factual/raw arrays never leave the source NPZ."""
import argparse, hashlib, json, traceback
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--repair-result',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
sidecar=a.out.with_suffix('.json'); assert not a.out.exists() and not sidecar.exists()
try:
    result=json.loads(a.repair_result.read_text())
    assert result['status']=='COMPLETED_TINY_HELDOUT_PILOT' and result['groups']['test']==['libero_spatial/task0/episode6']
    assert sha(a.source)==result['heldout_prediction']['sha256']
    with np.load(a.source,allow_pickle=False) as d:
        x=d['repair'].copy(); assert x.dtype.kind=='f' and x.shape==(5,112,224,3) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1
    temporary=a.out.with_name(a.out.name+'.tmp')
    with temporary.open('wb') as f: np.savez_compressed(f,repair=x)
    temporary.replace(a.out)
    meta={'status':'RUNTIME_REPAIR_ONLY','runtime_file':a.out.name,'runtime_sha256':sha(a.out),'source_predictions_sha256':sha(a.source),'repair_result_sha256':sha(a.repair_result),'test_wam_result_sha256':result['hashes']['test_wam'],'test_groups':result['groups']['test'],'keys':['repair'],'no_factual_or_raw_arrays':True}
    temporary_sidecar=sidecar.with_name(sidecar.name+'.tmp'); temporary_sidecar.write_text(json.dumps(meta,indent=2)); temporary_sidecar.replace(sidecar)
    print(json.dumps(meta))
except Exception:
    failure={'status':'FAILED_RUNTIME_REPAIR_EXPORT','traceback':traceback.format_exc()}
    temporary_sidecar=sidecar.with_name(sidecar.name+'.tmp'); temporary_sidecar.write_text(json.dumps(failure,indent=2)); temporary_sidecar.replace(sidecar)
    raise
