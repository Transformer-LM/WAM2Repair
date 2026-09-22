"""Print the numeric ray/raster diagnosis fields without dumping samples."""
import argparse
import json
from pathlib import Path


def compact(metrics):
    keys = ('sample_count', 'hit_count', 'geom_match_fraction', 'depth_p95_m',
            'depth_max_m')
    return {key: metrics.get(key) for key in keys if key in metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    out = {
        'status': report.get('status'),
        'scope': report.get('scope'),
        'limitations': report.get('limitations', []),
        'records': [],
    }
    for record in report.get('records', []):
        row = {'file': record.get('file'), 'cameras': {}}
        for camera, values in record.get('camera_rows', {}).items():
            saved = values.get('saved_vs_current', {})
            row['cameras'][camera] = {
                'saved_vs_current': {
                    key: {
                        'exact': item.get('exact'),
                        'max_abs': item.get('max_abs'),
                        'mismatched_elements': item.get('mismatched_elements'),
                    }
                    for key, item in saved.items()
                },
                'default_gate_convention': compact(values.get('original_environment_ray_metrics', {})),
                'diagnostic_convention_probes': {
                    name: compact(metrics)
                    for name, metrics in values.get('ray_convention_probes', {}).items()
                },
            }
        out['records'].append(row)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
