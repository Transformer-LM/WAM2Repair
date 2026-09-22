"""Verify external assets referenced by the recorded atomic object XML.

This is provenance isolation only.  It neither validates physical geometry nor
changes B0/D0 status.
"""
import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--asset-manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'scope': 'external-asset provenance check only; no B0 approval',
              'code_sha256': sha(__file__), 'source_result_sha256': sha(a.source / 'result.json'),
              'asset_manifest_sha256': sha(a.asset_manifest)}
    started = time.monotonic()
    try:
        source = json.loads((a.source / 'result.json').read_text())
        assert source['status'] == 'PASS_ATOMIC_REPEATABILITY_ONLY'
        manifest = json.loads(a.asset_manifest.read_text())
        assert manifest['source_result_sha256'] == report['source_result_sha256']
        expected = {}
        for item in manifest['assets']:
            path, digest = item['path'], item['sha256']
            assert path not in expected or expected[path] == digest, path
            expected[path] = digest
        report['manifest_source_binding_sha256'] = manifest['source_result_sha256']
        xml = a.source / 'libero_object_model.xml'
        inv = next(x for x in source['inventory'] if x['suite'] == 'libero_object')
        assert xml.is_file() and sha(xml) == inv['model_xml_sha256']
        paths = sorted({node.attrib['file'] for node in ET.parse(xml).getroot().findall('./asset/*')
                        if 'file' in node.attrib})
        assert paths
        rows = []
        for name in paths:
            path = Path(name)
            assert path.is_absolute(), name
            row = {'path': name, 'exists': path.is_file(), 'expected_sha256': expected.get(name)}
            if row['exists']:
                row['actual_sha256'] = sha(path)
            row['manifest_known'] = row['expected_sha256'] is not None
            row['matches'] = bool(row['manifest_known'] and row['exists'] and
                                  row['actual_sha256'] == row['expected_sha256'])
            rows.append(row)
        report['assets'] = rows
        report['asset_count'] = len(rows)
        report['all_assets_match_snapshot_era_manifest'] = all(row['matches'] for row in rows)
        report['status'] = 'PASS_ASSET_PROVENANCE_ONLY' if report['all_assets_match_snapshot_era_manifest'] else 'FAIL_ASSET_PROVENANCE_ONLY'
        report['limitations'] = ['a matching file hash does not validate renderer version or physical labels',
                                 'does not alter frozen B0 ray/raster criteria or permit D0',
                                 'does not authorize repair, WAM, or VLA claims']
    except Exception as exc:
        report['status'] = 'ERROR'; report['error'] = repr(exc); raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        (a.out / 'result.json').write_text(json.dumps(report, indent=2))
        print('RESULT', report['status'], flush=True)
    if report['status'] != 'PASS_ASSET_PROVENANCE_ONLY':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
