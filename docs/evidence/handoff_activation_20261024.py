"""One-time master-owned activation after the current race, never a Mac timer."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

ROOT = Path('/root')
STAGE = ROOT / 'releases/handoff-87cd623'
RECEIPT = ROOT / 'artifacts/handoff_release_20261024.json'
STATUS = ROOT / 'artifacts/handoff_activation_20261024.json'


def write(value):
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATUS)


def processes():
    result = {}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')
            parent = int((path.parent/'stat').read_text().rsplit(')', 1)[1].split()[1])
            result[int(path.parent.name)] = (parent, args)
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return result


def stop_tree(pids):
    tree = processes()
    selected = set(pids)
    while True:
        children = {pid for pid, (parent, _) in tree.items() if parent in selected}
        if children <= selected:
            break
        selected |= children
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in selected:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(1)


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, timeout=30)


def main():
    installed = json.loads(RECEIPT.read_text())
    for name, digest in installed['installed_files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    phase = 'waiting_for_current_race'
    while True:
        state = json.loads((ROOT/'artifacts/chain_state.json').read_text())
        epoch, length = int(state['epoch']), int(state['epoch_length'])
        offset = int(state['block']) % length
        table = processes()
        races = [pid for pid, (_, args) in table.items() if any(
            x in {'nanobody_opt_v2.sequential_exact_race', 'nanobody_opt.sequential_exact_race'} for x in args)]
        finished = (ROOT/f'live/epoch{epoch}/producer/full_scan_confirmation.json').exists()
        # Do not interrupt an old publisher in its final delivery window.
        clear_publication = offset < length-76 or offset >= length-41+2
        safe = not races and clear_publication and (finished or offset >= length-25)
        write({'status': phase, 'epoch': epoch, 'offset': offset,
               'confirmation_complete': finished, 'active_races': len(races),
               'release': installed['release']})
        if safe:
            break
        time.sleep(5)

    publisher = run('tmux', 'list-panes', '-t', 'nova-nanobody-publisher', '-F', '#{pane_start_command}')
    assert publisher.returncode == 0
    launch = publisher.stdout.strip()
    tokens = shlex.split(launch)
    assert 'publish-current-nanobody' in tokens and '/usr/local/bin/pixi' in tokens
    # This launch contains only ordinary flags, paths and the public field-mode
    # switch. Never persist or print command/environment credentials.
    for token in tokens:
        if '=' in token and not token.startswith('--') and token != '2>&1':
            assert token in {'NOVA_FULL_ARCHIVE_FIELD=0', 'NOVA_LOCAL_SQLITE=1'}
    assert not any(x in launch for x in ('$(', '`', 'Authorization', 'PASSWORD', 'TOKEN', 'SECRET'))
    phase = 'activating'
    write({'status': phase, 'epoch': epoch, 'release': installed['release']})

    argv_path = ROOT/'artifacts/producer_argv.txt'
    argv = argv_path.read_text().splitlines()
    assert '--epoch-candidate-budget' in argv and argv[argv.index('--epoch-candidate-budget')+1] == '4800'
    if '--pi-handoff' not in argv:
        argv.append('--pi-handoff')
    temporary = argv_path.with_suffix('.handoff-new')
    temporary.write_text('\n'.join(argv)+'\n')
    temporary.replace(argv_path)
    manifest = ROOT/'pixi.toml'
    lines = manifest.read_text().splitlines()
    matching = [i for i, line in enumerate(lines) if line.startswith('publish-current-nanobody =')]
    assert len(matching) == 1
    index = matching[0]
    assert lines[index].endswith('"') and 'python tools/publish_current_nanobody.py' in lines[index]
    if '--pi-handoff' not in lines[index]:
        lines[index] = lines[index][:-1] + ' --pi-handoff"'
    temporary = manifest.with_suffix('.handoff-new')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(manifest)

    table = processes()
    producer_pids = [pid for pid, (_, args) in table.items()
                     if any(x in {'nanobody_opt_v2.run_live_producer', 'nanobody_opt.run_live_producer'} for x in args)]
    publisher_pids = [pid for pid, (_, args) in table.items()
                     if any(x.endswith('/publish_current_nanobody.py') for x in args)]
    obsolete_jobs = [pid for pid, (_, args) in table.items()
                     if '/root/artifacts/eligibility_refresh_e649ceb.py' in args]
    stop_tree(producer_pids + publisher_pids + obsolete_jobs)
    for name in ['nova-nanobody-producer', 'nova-nanobody-publisher']:
        run('tmux', 'kill-session', '-t', name)
    for name, command in [('nova-nanobody-publisher', launch),
                          ('nova-nanobody-producer', '/root/producer_relaunch.sh')]:
        started = run('tmux', 'new-session', '-d', '-s', name, command)
        # The existing watchdog can win the restart race; verify its loaded
        # flags below rather than starting a second copy.
        assert started.returncode == 0 or run('tmux', 'has-session', '-t', name).returncode == 0
    for _ in range(30):
        table = processes()
        producers = [pid for pid, (_, args) in table.items() if '--pi-handoff' in args and
                     any(x in {'nanobody_opt_v2.run_live_producer', 'nanobody_opt.run_live_producer'} for x in args)]
        publishers = [pid for pid, (_, args) in table.items() if '--pi-handoff' in args and
                      any(x.endswith('/publish_current_nanobody.py') for x in args)]
        if producers and publishers:
            break
        time.sleep(1)
    assert producers and publishers
    installed.update(activation='active', activation_epoch=epoch,
                     first_full_epoch=epoch+1, producer_pids=producers, publisher_pids=publishers,
                     producer_argv_sha256=hashlib.sha256(argv_path.read_bytes()).hexdigest(),
                     pixi_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    RECEIPT.write_text(json.dumps(installed, indent=2)+'\n')
    write({'status': 'active', 'epoch': epoch, 'first_full_epoch': epoch+1,
           'release': installed['release'], 'producer_pids': producers, 'publisher_pids': publishers})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write({'status': 'failed', 'error_type': type(error).__name__})
        raise SystemExit(1)
