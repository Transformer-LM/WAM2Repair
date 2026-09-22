"""CPU-only inventory of saved WAM endpoints; never infer physical labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resized(x):
    return np.asarray(Image.fromarray(x).resize((224, 224), Image.Resampling.BILINEAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    report = {'scope':'legacy StarVLA action distribution; endpoint diagnostics only',
              'records':[], 'failures':[], 'sources':[]}
    prepared = sorted(a.root.glob('prepared_*.json'), key=lambda p:len(p.stem), reverse=True)
    for wm in sorted(a.root.glob('wam_*.json')):
        meta = json.loads(wm.read_text())
        if meta.get('schema') != 'state-repair-e1-wam-output-v1':
            continue
        stem = wm.stem.removeprefix('wam_')
        matches = [p for p in prepared if stem == p.stem.removeprefix('prepared_') or stem.startswith(p.stem.removeprefix('prepared_')+'_')]
        try:
            if not matches:
                raise ValueError('no matching prepared metadata')
            pm = matches[0]
            source = json.loads(pm.read_text())
            source_rows = {r['key']:r for r in source['records']}
            dp, wp = pm.with_suffix('.npz'), wm.with_suffix('.npz')
            report['sources'].append({'wam':str(wm), 'prepared':str(pm),
                'wam_npz_sha256':sha(wp), 'prepared_npz_sha256':sha(dp),
                'checkpoint_recorded':meta.get('wam_checkpoint'),
                'checkpoint_exists_now':Path(meta.get('wam_checkpoint','/nonexistent')).is_file()})
            with np.load(dp, allow_pickle=False) as data, np.load(wp, allow_pickle=False) as pred:
                for r in meta['records']:
                    key = r['key']
                    s = source_rows[key]
                    for field in ['suite','task_id','episode','instruction','action_shape']:
                        assert r[field] == s[field], field
                    action = data[key+'_action']
                    assert list(action.shape) == s['action_shape']
                    assert np.isfinite(action).all()
                    canvas = Image.new('RGB', (224*3, 2*250+32), 'white')
                    draw = ImageDraw.Draw(canvas)
                    draw.text((5,5), wm.stem+' / '+key, fill='black')
                    metrics = {}
                    for i, cam in enumerate(['primary','wrist']):
                        arrays = [data[key+'_'+cam], pred[key+'_pred_'+cam], data[key+'_gt_'+cam]]
                        for x in arrays:
                            assert x.dtype == np.uint8 and x.ndim == 3 and x.shape[-1] == 3
                        cur, hat, gt = [resized(x) for x in arrays]
                        for j, (label, x) in enumerate(zip(['current','WAM endpoint','simulator endpoint'], [cur,hat,gt])):
                            draw.text((j*224+4,i*250+32),cam+' '+label,fill='black')
                            canvas.paste(Image.fromarray(x),(j*224,i*250+54))
                        mse = float(np.mean((hat.astype(float)-gt.astype(float))**2))
                        metrics[cam] = {'mse':mse,'psnr':float(10*np.log10(255**2/max(mse,1e-12))),
                            'copy_current_mse':float(np.mean((cur.astype(float)-gt.astype(float))**2)),
                            'prediction_std':float(hat.std())}
                    sid = wm.stem+'_'+key
                    canvas.save(a.out/(sid+'.png'))
                    report['records'].append({'id':sid,'prepared_file':pm.name,'key':key,
                        'suite':s['suite'],'task_id':s['task_id'],'episode':s['episode'],
                        'seed':source['seed'], 'action_sha256':hashlib.sha256(action.tobytes()).hexdigest(),
                        'group':f"{s['suite']}/task{s['task_id']}/episode{s['episode']}",
                        'metrics':metrics,'physical_error_label':'UNREVIEWED',
                        'temporal_alignment':'UNVERIFIED: endpoints only, no saved frame timestamps',
                        'split':'diagnostic_only','sheet':sid+'.png'})
        except Exception as exc:
            report['failures'].append({'file':str(wm),'error':repr(exc)})
    report['n_endpoint_pairs'] = len(report['records'])
    report['n_source_samples'] = len({(r['prepared_file'],r['key']) for r in report['records']})
    report['n_episode_groups'] = len({r['group'] for r in report['records']})
    report['limitations'] = ['saved endpoints are not full videos',
        'no saved simulator qpos/qvel or per-frame contacts in legacy inputs',
        'same task/episode across seeds must not be counted as independent layouts',
        'no automatic penetration or contact ground-truth labels',
        'legacy action normalization and frame-to-action timing require source verification',
        'not pi0.5 evaluation or repair training data admission']
    (a.out/'inventory.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','sources']},indent=2))


if __name__ == '__main__':
    main()
