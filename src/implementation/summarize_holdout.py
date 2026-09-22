"""Post-evaluation descriptive statistics; no checkpoint selection or retraining."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('result',type=Path); p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); result=json.loads(a.result.read_text()); rows=result['test_samples']
groups=sorted({r['group'] for r in rows})
def aggregate(values):
    values=np.asarray(values,dtype=float)
    means=[float(values[list(indices)].mean()) for indices in itertools.product(range(len(values)),repeat=len(values))]
    return {'mean':float(values.mean()),'episode_values':values.tolist(),
        'episode_sd':float(values.std(ddof=1)) if len(values)>1 else None,
        'exploratory_exact_cluster_bootstrap_95_percentile':np.quantile(means,[.025,.975],method='inverted_cdf').tolist(),
        'independent_groups':len(values)}
methods={}
for name in rows[0]['metrics']:
    methods[name]={}
    for metric in ['mae','temporal_mae','changed_region_mae']:
        vals=[]
        for group in groups:
            samples=[r['metrics'][name][metric] for r in rows if r['group']==group and r['metrics'][name][metric] is not None]
            if samples: vals.append(np.mean(samples))
        methods[name][metric]=aggregate(vals)
deltas={}
for baseline in ['raw','copy_current','anchored','no_wam']:
    values=[np.mean([r['metrics']['repair']['mae']-r['metrics'][baseline]['mae'] for r in rows if r['group']==group]) for group in groups]
    deltas[baseline]=aggregate(values)
    deltas[baseline]['relative_mae_reduction']=1-methods['repair']['mae']['mean']/methods[baseline]['mae']['mean']
slices={}
for name,subset in [('all',rows),('raw_low_pixel_error',[r for r in rows if r['metrics']['raw']['mae']<=.03]),
                    ('failed_episode',[r for r in rows if not r['episode_success']])]:
    group_rows=[]
    for group in groups:
        members=[r for r in subset if r['group']==group]
        if members:
            group_rows.append({'group':group,'windows':len(members),**{key:float(np.mean([r[key] for r in members]))
                for key in ['repair_edit_mae','repair_minus_raw_mae','oracle_clean_identity_mae']},
                'error_increase_fraction':float(np.mean([r['repair_minus_raw_mae']>0 for r in members]))})
    slices[name]={'windows':len(subset),'episodes':group_rows,
        'episode_macro':{key:aggregate([r[key] for r in group_rows]) for key in ['repair_edit_mae','repair_minus_raw_mae','oracle_clean_identity_mae','error_increase_fraction']} if group_rows else None}
report={'source_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'methods':methods,'repair_minus_baseline':deltas,
    'preservation_slices':slices,'scope':'simulation_only; post-evaluation descriptive statistics',
    'limitations':['Only two independent test episode groups: bootstrap intervals are exploratory and severely under-resolved, not evidence of robust significance.',
        'One training seed; windows within an episode are not independent.',
        'Pixel-low-error does not certify physically correct raw futures.',
        'Oracle-clean identity probes true future inputs only as a privileged stress test.',
        'Failure semantics and physical correction cannot be established from these metrics.']}
a.out.write_text(json.dumps(report,indent=2)); print(json.dumps({'deltas':deltas,'slice_windows':{k:v['windows'] for k,v in slices.items()}}))
