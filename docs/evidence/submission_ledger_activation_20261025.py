"""Master-owned release activation; drain the current discovery batch first."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path('/root')
STAGE = ROOT / 'releases/submission-ledger-76ea3a0'
RECEIPT = ROOT / 'artifacts/submission_ledger_release_20261025.json'
STATUS = ROOT / 'artifacts/submission_ledger_activation_20261025.json'
PAUSED = set()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def processes():
    table = {}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')
            stat = (path.parent / 'stat').read_text().rsplit(')', 1)[1].split()
            table[int(path.parent.name)] = {'args': args, 'state': stat[0], 'parent': int(stat[1])}
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return table


def children(table, parents):
    selected = set(parents)
    while True:
        added = {pid for pid, item in table.items() if item['parent'] in selected}
        if added <= selected:
            return selected - set(parents)
        selected |= added


def resume():
    for pid in list(PAUSED):
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        PAUSED.discard(pid)


def pause(pid):
    os.kill(pid, signal.SIGSTOP)
    PAUSED.add(pid)


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, timeout=30)


def state():
    value = json.loads((ROOT / 'artifacts/chain_state.json').read_text())
    epoch, length = int(value['epoch']), int(value['epoch_length'])
    return epoch, int(value['block']) % length, length, value['nanobody_target']


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    receipt = json.loads(RECEIPT.read_text())
    for name, digest in receipt['previous_files'].items():
        assert (hashlib.sha256((ROOT / name).read_bytes()).hexdigest() if (ROOT / name).is_file() else None) == digest
    for name, digest in receipt['runtime_files'].items():
        assert hashlib.sha256((STAGE / name).read_bytes()).hexdigest() == digest
    pane = run('tmux', 'list-panes', '-t', 'nova-nanobody-publisher', '-F', '#{pane_start_command}')
    assert pane.returncode == 0
    launch = pane.stdout.strip()
    for _ in range(3):
        outer = shlex.split(launch)
        if len(outer) != 1 or outer[0] == launch:
            break
        launch = outer[0]
    tokens = shlex.split(launch)
    assert 'publish-current-nanobody' in tokens and '/usr/local/bin/pixi' in tokens
    for token in tokens:
        if '=' in token and not token.startswith('--') and token != '2>&1':
            assert token in {'NOVA_FULL_ARCHIVE_FIELD=0', 'NOVA_LOCAL_SQLITE=1'}
    assert not any(x in launch for x in ('$(', '`', 'Authorization', 'PASSWORD', 'TOKEN', 'SECRET'))
    proof = json.loads((ROOT / 'artifacts/pi_submission_outcomes.json').read_text())
    assert 0 <= time.time() - proof['generated_at'] < 900
    drained = None
    while True:
        epoch, offset, length, target = state()
        table = processes()
        producers = [pid for pid, item in table.items()
                     if 'nanobody_opt_v2.run_live_producer' in item['args']]
        races = [pid for pid, item in table.items()
                 if 'nanobody_opt_v2.sequential_exact_race' in item['args']]
        assert len(producers) == 1
        old_producer = producers[0]
        if len(races) == 1 and offset < length - 136:
            pid = races[0]
            args = table[pid]['args']
            race_epoch = int(args[args.index('--epoch') + 1])
            rounds = int(args[args.index('--replication-rounds') + 1])
            if race_epoch == epoch and rounds == 0:
                output = Path(args[args.index('--output-dir') + 1]).resolve()
                assert output.is_relative_to(ROOT / 'live' / f'epoch{epoch}' / 'producer')
                # Stop only the scheduler. Its scorer children continue, and
                # the producer remains awake to drain inherited stdout pipes.
                pause(pid)
                started = time.monotonic()
                while True:
                    current_epoch, current_offset, _, _ = state()
                    assert current_epoch == epoch and current_offset < length - 76
                    assert time.monotonic() - started < 900
                    current = processes()
                    assert pid in current and current[pid]['state'] in {'T', 't'}
                    busy = [child for child in children(current, {pid})
                            if current[child]['state'] != 'Z']
                    inputs = list((output / 'screen').glob('tranche_*/candidates.json'))
                    aggregates = [p.parent / 'exact_all.json' for p in inputs]
                    complete = all(p.is_file() for p in aggregates)
                    write(STATUS, {'status': 'draining_discovery_batch', 'epoch': epoch,
                                   'offset': current_offset, 'active_children': len(busy),
                                   'dispatched_tranches': len(inputs),
                                   'completed_tranches': sum(p.is_file() for p in aggregates)})
                    if not busy and complete:
                        from nanobody_opt.rank_exact import _observations
                        observed = _observations(aggregates, target)
                        drained = {'epoch': epoch, 'iteration': output.parent.name,
                                   'completed_tranches': len(aggregates),
                                   'completed_observations': len(observed),
                                   'preserved_files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                       for p in aggregates}}
                        break
                    time.sleep(2)
                pause(old_producer)
                break
        if not races and (offset < length - 76 or offset >= length - 39):
            pause(old_producer)
            current = processes()
            if any('nanobody_opt_v2.sequential_exact_race' in x['args'] for x in current.values()):
                resume()
            else:
                break
        write(STATUS, {'status': 'waiting_for_safe_restart', 'epoch': epoch, 'offset': offset})
        time.sleep(2)

    import importlib.util
    spec = importlib.util.spec_from_file_location('release_publication_ledger', STAGE / 'nanobody_opt/publication_ledger.py')
    publication_ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publication_ledger)
    ledger = ROOT / 'artifacts/nanobody_published_sequences.txt'
    raw_before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    recovered = []
    for origin in [25914, 25915]:
        payload = json.loads((ROOT / f'artifacts/published_retirement_audit_20261025/epoch{origin}_unused.json').read_text())
        sequences = payload['sequences']
        assert all(isinstance(sequence, str) for sequence in sequences)
        recovered.extend(sequences)
        publication_ledger.record(ledger, epoch=origin, target=target, sequences=sequences)
    effective = publication_ledger.effective(ledger, epoch=epoch, target=target,
        history=ROOT / 'artifacts/nanobody_history', outcomes=ROOT / 'artifacts/pi_submission_outcomes.json',
        output=ROOT / 'artifacts/recovered_submission_exclusions.txt')
    exclusions = set(effective.read_text().splitlines())
    assert len(recovered) == 30 and not set(recovered) & exclusions
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == raw_before
    activated_at = time.time()
    for name, digest in receipt['previous_files'].items():
        assert (hashlib.sha256((ROOT / name).read_bytes()).hexdigest() if (ROOT / name).is_file() else None) == digest
        backup = STAGE / 'before' / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, backup)
        temporary = (ROOT / name).with_suffix('.pipeline-new')
        shutil.copy2(STAGE / name, temporary)
        temporary.replace(ROOT / name)
    for name, digest in receipt['runtime_files'].items():
        assert (hashlib.sha256((ROOT / name).read_bytes()).hexdigest() if (ROOT / name).is_file() else None) == digest
    old_publishers = {pid for pid, item in processes().items()
                      if any(arg.endswith('/publish_current_nanobody.py') for arg in item['args'])}
    assert len(old_publishers) == 1
    parents = {old_producer} | old_publishers
    selected = parents | children(processes(), parents)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in selected:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(1)
    PAUSED.clear()
    run('tmux', 'kill-session', '-t', 'nova-nanobody-producer')
    run('tmux', 'kill-session', '-t', 'nova-nanobody-publisher')
    pub_launch = run('tmux', 'new-session', '-d', '-s', 'nova-nanobody-publisher', launch)
    assert pub_launch.returncode == 0 or run('tmux', 'has-session', '-t', 'nova-nanobody-publisher').returncode == 0
    launched = run('tmux', 'new-session', '-d', '-s', 'nova-nanobody-producer', '/root/producer_relaunch.sh')
    assert launched.returncode == 0 or run('tmux', 'has-session', '-t', 'nova-nanobody-producer').returncode == 0
    for _ in range(120):
        table = processes()
        running = [pid for pid, item in table.items() if 'nanobody_opt_v2.run_live_producer' in item['args']]
        publishers = [pid for pid, item in table.items() if any(arg.endswith('/publish_current_nanobody.py') for arg in item['args'])]
        if len(running) == 1 and running[0] != old_producer and len(publishers) == 1 and publishers[0] not in old_publishers:
            args = table[running[0]]['args']
            assert all(flag in args for flag in ['--adaptive-confirmation-depth', '--pi-handoff'])
            assert args[args.index('--epoch-candidate-budget') + 1] == '4800'
            budget_path = ROOT / 'live' / f'epoch{epoch}' / 'producer/budget.json'
            budget = json.loads(budget_path.read_text()) if budget_path.exists() else {}
            if (budget_path.exists() and budget_path.stat().st_mtime >= activated_at
                    and budget.get('planning_minimum_observations') in {10, 20}
                    and (ROOT / f'live/epoch{epoch}/producer/submission_exclusions.receipt.json').exists()
                    and (ROOT / f'live/epoch{epoch}/submission_exclusions.receipt.json').exists()):
                break
        time.sleep(1)
    else:
        raise RuntimeError('new producer budget was not verified')
    if drained:
        assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                   for name, digest in drained['preserved_files'].items())
    receipt.update(activation='active', activation_epoch=epoch, first_full_epoch=epoch + 1,
                   producer_pid=running[0], old_producer_pid=old_producer,
                   publisher_pid=publishers[0], recovered_audited_candidates=len(recovered),
                   original_ledger_sha256=raw_before,
                   activation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   drained_discovery=drained,
                   live_budget={key: budget[key] for key in ['planning_minimum_observations',
                       'pending_new_evaluations', 'replication_reserve_blocks', 'screen_cutoff_block',
                       'load_seconds', 'gpu_seconds_per_candidate', 'effective_handoff_reserve_blocks']})
    write(RECEIPT, receipt)
    write(STATUS, {key: receipt[key] for key in ['activation', 'release', 'activation_epoch',
                                               'first_full_epoch', 'producer_pid', 'publisher_pid', 'recovered_audited_candidates', 'live_budget']})


if __name__ == '__main__':
    def terminate(signum, frame):
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        main()
    except BaseException as error:
        write(STATUS, {'status': 'failed', 'error_type': type(error).__name__})
        raise SystemExit(1)
    finally:
        resume()
