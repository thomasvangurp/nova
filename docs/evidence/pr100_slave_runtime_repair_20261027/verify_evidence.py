"""Offline evidence integrity checks; never imports or executes archived helpers."""
from collections import Counter
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sanitized(value):
    if isinstance(value, dict):
        forbidden = {'environment', 'environ', 'authorization', 'password',
                     'access_token', 'refresh_token', 'client_secret', 'private_key',
                     'cookie', 'set-cookie', 'sequence', 'sequence_pool'}
        assert not (set(str(k).lower() for k in value) & forbidden)
        for key, child in value.items():
            if key in ('stdout', 'stderr', 'stdout_tail', 'stderr_tail', 'raw_stdout', 'raw_stderr'):
                assert type(child) is int  # Byte counts are retained, streams are not.
            check_sanitized(child)
    elif isinstance(value, list):
        for child in value:
            check_sanitized(child)
    elif isinstance(value, str):
        assert not any(marker in value for marker in (
            'BEGIN PRIVATE KEY', 'BEGIN RSA PRIVATE KEY', 'ghp_', 'glpat-',
            'X-Amz-Signature=', 'access_token='))


def main():
    bundle = read('bundle-manifest.json')
    expected_files = set(bundle['files']) | {'bundle-manifest.json'}
    actual_files = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    assert actual_files == expected_files
    for name, expected in bundle['files'].items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
        assert digest(path) == expected['sha256'] and path.stat().st_size == expected['bytes']
        if path.suffix == '.json':
            check_sanitized(json.loads(path.read_text()))
        elif path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)

    receipt = read('qualification-receipt.json')
    measured = read('measurements.json')
    approved = read('qualification-approved.json')
    identity = receipt['runtime_identity']
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                separators=(',', ':')).encode()).hexdigest()
    assert identity_sha == receipt['runtime_identity_sha256'] == measured['runtime_identity_sha256']
    assert identity_sha == '6eaada274269a1e60b642240326d851830cb7dda262238ee2b9b8a3e6dc44e18'
    assert all(d['status'] == 'passed' for d in (receipt, measured, approved))
    approval = approved['root_runtime_difference_approval']
    assert approval['approved'] is True and approval['runtime_identity_sha256'] == identity_sha
    assert receipt['returncode'] == 0 and receipt['completed_profiles'] == 32
    assert receipt['runtime_inconclusive_count'] == 0
    assert receipt['affinity'] == list(range(32)) and receipt['cpu_affinity_verified'] is True
    assert receipt['cleanup']['remaining_descendants'] == 0
    assert receipt['cleanup']['fixed_point_reached'] is True
    assert receipt['publication_enabled'] is False and receipt['authoritative'] is False
    records = measured['records']
    assert len(records) == len({r['sequence_hash'] for r in records}) == 32
    assert all(r['numeric_metrics_complete'] and r['provenance_complete'] and r['runtime_complete']
               for r in records)
    assert Counter(r['observation']['outcome'] for r in records) == {'rejected': 32}
    assert measured['sources_unchanged'] is True
    for key in ('scientific_sources', 'model_weights'):
        assert len(identity[key]) == 4 and identity[key] == receipt['reference_runtime_identity'][key]

    bindings = {
        'manifest.json': identity['manifest_sha256'],
        'metadata-supplement.json': identity['metadata_manifest_sha256'],
        'mkl-supplement-family-e.json': identity['supplemental_manifest_sha256'],
        'reviewed-runtime-allowlist.json': receipt['reference_runtime_identity']['manifest_sha256'],
    }
    for name, expected in bindings.items():
        assert digest(ROOT / 'manifests' / name) == expected
    assert len(read('manifests/manifest.json')['files']) == 5369
    assert len(read('manifests/metadata-supplement.json')) == 166
    supplement = read('manifests/mkl-supplement-family-e.json')['files']
    assert len(supplement) == 15
    assert {entry['target']: entry['sha256'] for entry in supplement.values()} == identity['supplemental_cpu_libraries']
    for name, evidence in read('helper-provenance.json')['files'].items():
        assert evidence['byte_identical'] and digest(ROOT / name) == evidence['sha256']

    audit = read('submission-yield-audit.json')
    assert audit['registered_crew']['count'] == 13
    for epoch, fresh, repeats, pool, commits in [('25965', 1396, 19, 3, 3), ('25966', 1167, 59, 18, 13)]:
        row = audit['epochs'][epoch]
        assert row['completed_measurements']['fresh_evaluations'] == fresh
        assert row['completed_measurements']['repeat_evaluations'] == repeats
        assert sum(row['native_measurement_gate_counts'].values()) == fresh + repeats
        assert row['published_pool_count'] == row['final_native_count'] == pool
        assert row['published_not_in_final_native_count'] == row['final_native_not_published_count'] == 0
        events = audit['consumer_receipts'][epoch]
        plan = [e for e in events if e['event'] == 'plan'][-1]
        done = [e for e in events if e['event'] == 'done'][-1]
        assert plan['crew'] == 13 and plan['committing'] == commits
        assert done['committed'] == commits and done['failed'] == 0
        assert len({e['uid'] for e in events if e['event'] == 'commit' and e['rc'] == 0}) == commits
    print(json.dumps({'status': 'verified', 'bundle_files': len(expected_files),
                      'numeric_profiles': 32, 'runtime_identity_sha256': identity_sha,
                      'epoch_uid_commit_counts': {'25965': 3, '25966': 13}}))


if __name__ == '__main__':
    main()
