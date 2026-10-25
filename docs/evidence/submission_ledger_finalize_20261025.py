"""Independently verify a running release after a bounded readiness timeout."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/root')
ARTIFACTS = ROOT / 'artifacts'
STAGE = ROOT / 'releases/submission-ledger-76ea3a0'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.verified-next')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    receipt_path = ARTIFACTS / 'submission_ledger_release_20261025.json'
    status_path = ARTIFACTS / 'submission_ledger_activation_20261025.json'
    receipt = read(receipt_path)
    assert receipt['release'] == '76ea3a057362607f5fc7ef0932650b58586313dc'
    for name, digest in receipt['runtime_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for name, digest in receipt['previous_files'].items():
        backup = STAGE / 'before' / name
        assert (hashlib.sha256(backup.read_bytes()).hexdigest() if backup.exists() else None) == digest
    installed_at = max((ROOT / name).stat().st_ctime for name in receipt['runtime_files'])
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                if line.startswith('btime '))
    processes = {}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')
            kind = ('producer' if 'nanobody_opt_v2.run_live_producer' in args else
                    'publisher' if any(a.endswith('/publish_current_nanobody.py') for a in args) else None)
            if not kind:
                continue
            assert kind not in processes
            assert '--adaptive-confirmation-depth' in args and '--pi-handoff' in args
            if kind == 'producer':
                assert args[args.index('--epoch-candidate-budget') + 1] == '4800'
            stat = (path.parent / 'stat').read_text().rsplit(')', 1)[1].split()
            started = boot + int(stat[19]) / os.sysconf('SC_CLK_TCK')
            assert started >= installed_at - 1
            assert stat[0] not in {'Z', 'T', 't'}
            processes[kind] = {'pid': int(path.parent.name), 'started_at': started}
        except (OSError, UnicodeError):
            continue
    assert set(processes) == {'producer', 'publisher'}
    state = read(ARTIFACTS / 'chain_state.json')
    epoch = state['epoch']
    assert epoch == 25920
    epoch_root = ROOT / f'live/epoch{epoch}'
    budget_path = epoch_root / 'producer/budget.json'
    budget = read(budget_path)
    assert budget_path.stat().st_mtime >= processes['producer']['started_at']
    assert budget['planning_minimum_observations'] in {10, 20}
    assert budget['effective_handoff_reserve_blocks'] == 76
    for name, kind in [('producer/submission_exclusions.receipt.json', 'producer'),
                       ('submission_exclusions.receipt.json', 'publisher')]:
        path = epoch_root / name
        assert read(path)['epoch'] == epoch and path.stat().st_mtime >= processes[kind]['started_at']
    proof = read(ARTIFACTS / 'pi_submission_outcomes.json')
    assert 0 <= time.time() - proof['generated_at'] <= 900
    audited = set()
    for origin in [25914, 25915]:
        audited.update(read(ARTIFACTS / f'published_retirement_audit_20261025/epoch{origin}_unused.json')['sequences'])
    assert len(audited) == 30
    ranked = {row['sequence'] for row in read(epoch_root / 'provisional_rankings.json')['candidates']}
    assert audited <= ranked
    artifact_path = ARTIFACTS / 'nanobody_result.json'
    artifact = read(artifact_path)
    assert artifact['epoch'] == epoch and artifact_path.stat().st_mtime >= processes['publisher']['started_at']
    rows = artifact['submissions']
    assert len(rows) == 13 and all(row['observation_count'] >= 20 for row in rows)
    recovered = sum(row['sequence'] in audited for row in rows)
    assert recovered > 0
    scores = [row['selection']['score'] for row in rows]
    preserved = {}
    observations = 0
    for path in sorted((epoch_root / 'producer').glob('*/race/screen/tranche_*/exact_all.json')):
        if path.stat().st_mtime <= installed_at:
            observations += int(read(path)['n'])
            preserved[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observations >= 2100
    earlier = read(ARTIFACTS / 'pipeline_completion_release_20261025.json')
    for name, digest in earlier['drained_discovery']['preserved_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    receipt.update(
        activation='active', activation_epoch=epoch, first_full_epoch=epoch + 1,
        independently_verified_at=time.time(), initial_verification=read(status_path),
        initial_verification_reason='120-second readiness timeout; resumed budget appeared afterward',
        processes=processes, preserved_completed_observations=observations,
        preserved_files=preserved, recovered_audited_candidates=30,
        recovered_in_published_fleet=recovered,
        published_local_mean=sum(scores) / len(scores),
        publication_observations=[row['observation_count'] for row in rows],
        activation_source_sha256=hashlib.sha256((ARTIFACTS / 'submission_ledger_activation_20261025.py').read_bytes()).hexdigest(),
        independent_verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        live_budget={k: budget[k] for k in ['planning_minimum_observations', 'pending_new_evaluations',
            'replication_reserve_blocks', 'screen_cutoff_block', 'load_seconds',
            'gpu_seconds_per_candidate', 'effective_handoff_reserve_blocks']})
    save(receipt_path, receipt)
    summary = {k: receipt[k] for k in ['activation', 'release', 'activation_epoch', 'first_full_epoch',
               'recovered_audited_candidates', 'recovered_in_published_fleet', 'published_local_mean',
               'preserved_completed_observations', 'independently_verified_at']}
    save(status_path, summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
