"""Numeric post-release checks; never emit raw sequences, process args or logs."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/root')


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def score_stats(rows):
    scores = [float(row['selection']['score']) for row in rows
              if isinstance(row, dict) and isinstance(row.get('selection'), dict)
              and isinstance(row['selection'].get('score'), (int, float))]
    return {'count': len(scores), 'mean': sum(scores) / len(scores) if scores else None,
            'minimum': min(scores, default=None), 'maximum': max(scores, default=None)}


def main():
    state = read(ROOT / 'artifacts/chain_state.json')
    epoch = int(state['epoch'])
    release = read(ROOT / 'artifacts/submission_ledger_release_20261025.json')
    status = read(ROOT / 'artifacts/submission_ledger_activation_20261025.json')
    audited = set()
    for origin in [25914, 25915]:
        audited.update(read(ROOT / f'artifacts/published_retirement_audit_20261025/epoch{origin}_unused.json')['sequences'])
    proof = read(ROOT / 'artifacts/pi_submission_outcomes.json')
    artifact_path = ROOT / 'artifacts/nanobody_result.json'
    artifact = read(artifact_path)
    artifact_sequences = set(artifact.get('sequence_pool', []))
    artifact_sequences.update(row['sequence'] for row in artifact.get('submissions', []) if row.get('sequence'))
    report = {
        'verified_at': time.time(), 'epoch': epoch, 'offset': state['block'] % state['epoch_length'],
        'release': release['release'], 'activation': status,
        'installed_files_match': all((ROOT / name).is_file() and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                                     for name, digest in release['runtime_files'].items()),
        'pi_proof_age_seconds': time.time() - proof['generated_at'],
        'artifact': {'epoch': artifact.get('epoch'), 'stage': artifact.get('publication_stage'),
                     'mtime': artifact_path.stat().st_mtime if artifact_path.exists() else None,
                     'audited_recovered_count': len(audited & artifact_sequences),
                     'local_scores': score_stats(artifact.get('submissions', [])),
                     'observations': [r.get('observation_count') for r in artifact.get('submissions', [])]},
        'exclusion_receipts': {}, 'rankings': {},
    }
    for name in ['producer/submission_exclusions', 'submission_exclusions']:
        report['exclusion_receipts'][name] = read(ROOT / f'live/epoch{epoch}/{name}.receipt.json')
    for name in ['provisional_rankings.json', 'rankings.json']:
        path = ROOT / f'live/epoch{epoch}' / name
        rows = read(path).get('candidates', [])
        report['rankings'][name] = {'mtime': path.stat().st_mtime if path.exists() else None,
                                   'count': len(rows),
                                   'audited_recovered_count': sum(r.get('sequence') in audited for r in rows)}
    path = ROOT / 'artifacts/submission_ledger_verification_20261025.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
