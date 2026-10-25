#!/usr/bin/env python3
"""Promote verified source at a quiet epoch boundary; restore code on failure.

No infrastructure provisioning, model replacement or dependency update occurs in
this transaction. The old processes remain running if preflight fails. Durable
state is never rewound. All child output and launch arguments stay private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import fcntl
import uuid
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.promotion_gate import read, require, verify_host, digest
from scripts.release_manifest import safe_path, verify
from scripts.release_state import capture, quarantine_derived


@contextmanager
def cleanup_on_disconnect():
    """SSH disconnect/cancellation must enter the same rollback/finally path."""
    received = False
    def interrupted(_signum, _frame):
        nonlocal received
        if not received:
            received = True
            raise InterruptedError('promotion_interrupted')
        # Repeated cancellation must not interrupt restoration halfway through.
    previous = {sig: signal.signal(sig, interrupted) for sig in
                [signal.SIGHUP, signal.SIGTERM, signal.SIGINT]}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def ignore_cancellation():
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in
                [signal.SIGHUP, signal.SIGTERM, signal.SIGINT]}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.promotion-next')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def processes():
    table = {}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')
            stat = (path.parent / 'stat').read_text().rsplit(')', 1)[1].split()
            table[int(path.parent.name)] = (int(stat[1]), args)
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return table


def owners(table):
    result = {}
    for pid, (_, args) in table.items():
        kind = ('producer' if 'nanobody_opt_v2.run_live_producer' in args else
                'publisher' if any(a.endswith('/publish_current_nanobody.py') for a in args) else None)
        if kind:
            require(kind not in result, 'duplicate_runtime_owner')
            result[kind] = pid
    require(set(result) == {'producer', 'publisher'}, 'missing_runtime_owner')
    return result


def descendants(table, parents):
    selected = set(parents)
    while True:
        more = {pid for pid, (parent, _) in table.items() if parent in selected}
        if more <= selected:
            return selected - set(parents)
        selected |= more


def command(*argv):
    return subprocess.run(argv, capture_output=True, text=True, timeout=30)


def publisher_launch():
    result = command('tmux', 'list-panes', '-t', 'nova-nanobody-publisher', '-F', '#{pane_start_command}')
    require(result.returncode == 0, 'publisher_launcher_unavailable')
    launch = result.stdout.strip()
    for _ in range(3):
        tokens = shlex.split(launch)
        if len(tokens) != 1 or tokens[0] == launch:
            break
        launch = tokens[0]
    tokens = shlex.split(launch)
    require('publish-current-nanobody' in tokens and '/usr/local/bin/pixi' in tokens,
            'unsupported_publisher_launcher')
    require(not any(x in launch for x in ('$(', '`', 'TOKEN', 'PASSWORD', 'SECRET', 'Authorization')),
            'unsafe_publisher_launcher')
    for token in tokens:
        if '=' in token and not token.startswith('--') and token != '2>&1':
            require(token in {'NOVA_LOCAL_SQLITE=1', 'NOVA_FULL_ARCHIVE_FIELD=0'}, 'unknown_launcher_environment')
    return launch


def stop(pids):
    selected = set(pids) | descendants(processes(), set(pids))
    for sig in [signal.SIGTERM, signal.SIGKILL]:
        for pid in selected:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(1)
    for name in ['nova-nanobody-producer', 'nova-nanobody-publisher']:
        command('tmux', 'kill-session', '-t', name)


def launch(publisher):
    for name, argv in [('nova-nanobody-producer', '/root/producer_relaunch.sh'),
                       ('nova-nanobody-publisher', publisher)]:
        result = command('tmux', 'new-session', '-d', '-s', name, argv)
        require(result.returncode == 0, 'runtime_restart_failed')


def install(source, root, manifest, *, deadline=None):
    for name in manifest['runtime_files']:
        require(deadline is None or time.monotonic() < deadline, 'installation_window_ended')
        destination = safe_path(root, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix(destination.suffix + '.promotion-new')
        shutil.copy2(safe_path(source, name), pending)
        pending.replace(destination)
    pending = root / 'build_stamp.promotion-new'
    shutil.copy2(source / 'build_stamp.json', pending)
    pending.replace(root / 'build_stamp.json')


def healthy(root, *, epoch, activated, previous_pids):
    current = owners(processes())
    require(not set(current.values()) & set(previous_pids), 'old_runtime_still_running')
    artifact = read(root / 'artifacts/nanobody_result.json')
    require(artifact['epoch'] == epoch and artifact['updated_at_unix'] >= activated,
            'no_new_publication')
    rows = artifact['submissions']
    require(len(rows) >= 13 and len({row['sequence'] for row in rows}) == len(rows)
            and not artifact.get('unassigned_uids') and all(
                row['checkpoint']['exact']['observation_count'] >= 10 for row in rows), 'live_fleet_degraded')
    # The promotion trial supplies quality acceptance; this checks actual startup
    # and delivery readiness, not a fabricated prediction of validator rewards.
    return {'processes': current, 'artifact_sha256': digest(root / 'artifacts/nanobody_result.json'),
            'entries': len(rows)}


def readiness_remaining(root, epoch):
    """Use the real publication window; never invent a short canary deadline."""
    from nanobody_opt.submission_timing import plan, BLOCK_SECONDS
    state = read(root / 'artifacts/chain_state.json')
    require(state['epoch'] == epoch and 0 <= time.time() - state['updated_at_unix'] <= 90,
            'readiness_clock_changed_or_stale')
    cutoff = state['epoch_length'] - plan()['publication_reserve_blocks']
    return max(0, (cutoff - state['block'] % state['epoch_length']) * BLOCK_SECONDS)


def promote(stage, bundle, root, *, host, target, readiness_seconds=None):
    stage, bundle, root = (Path(p).resolve() for p in [stage, bundle, root])
    candidate = read(stage / 'build_stamp.json')
    verify(stage, candidate, expected_sha=candidate['git_sha'])
    from scripts.deploy_tested_release import deployment_plan
    deployment_plan(stage / 'build_stamp.json', candidate['git_sha'], host, True, bundle, target)
    qualification = verify_host(bundle, runtime_root=root, candidate_sha=candidate['git_sha'], host=host, target=target)
    baseline = read(root / 'build_stamp.json')
    require(candidate['pixi_lock_hash'] == baseline['pixi_lock_hash'], 'dependency_change_requires_image_rollout')
    require({k: v for k, v in candidate['files'].items() if k.startswith('external_tools/')} ==
            {k: v for k, v in baseline['files'].items() if k.startswith('external_tools/')},
            'scorer_change_requires_image_rollout')
    publisher = publisher_launch()
    previous = owners(processes())
    paused = []
    copied = False
    attempt = stage / ('promotion-attempt-' + uuid.uuid4().hex)
    backup = attempt / 'code-before'
    receipt_path = root / 'artifacts/release_promotion.json'
    old_receipt = read(receipt_path) if receipt_path.is_file() else None
    # Slow forensic copies happen with the old owners RUNNING. A large SQLite
    # archive must never freeze an epoch while waiting for its backup.
    backup.mkdir(parents=True)
    for name in baseline['runtime_files']:
        path = safe_path(backup, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(safe_path(root, name), path)
    shutil.copy2(root / 'build_stamp.json', backup / 'build_stamp.json')
    argv = root / 'artifacts/producer_argv.txt'
    shutil.copy2(argv, backup / 'producer_argv.txt')
    capture(root, attempt / 'state-before-promotion', epoch=qualification['activation_epoch'],
            baseline_sha=baseline['git_sha'], candidate_sha=candidate['git_sha'])
    verify_host(bundle, runtime_root=root, candidate_sha=candidate['git_sha'], host=host, target=target)
    try:
        for pid in previous.values():
            paused.append(pid)
            os.kill(pid, signal.SIGSTOP)
        require(not descendants(processes(), set(previous.values())), 'runtime_has_inflight_children')
        verify_host(bundle, runtime_root=root, candidate_sha=candidate['git_sha'], host=host, target=target)
        copied = True
        install(stage, root, candidate, deadline=time.monotonic() + 30)
        shutil.copy2(stage / 'deploy/producer/producer_argv.txt', argv)
        verify(root, candidate, expected_sha=candidate['git_sha'], runtime_only=True)
        stop(previous.values())
        paused.clear()
        activated = time.time()
        launch(publisher)
        success = None
        allowed = readiness_remaining(root, qualification['activation_epoch'])
        deadline = time.monotonic() + min(allowed, readiness_seconds if readiness_seconds is not None else allowed)
        while time.monotonic() < deadline:
            require(readiness_remaining(root, qualification['activation_epoch']) > 0,
                    'publication_window_ended')
            try:
                success = healthy(root, epoch=qualification['activation_epoch'], activated=activated,
                                  previous_pids=previous.values())
                break
            except (OSError, ValueError, KeyError, TypeError):
                time.sleep(2)
        require(success is not None, 'new_release_not_ready')
        receipt = {**qualification, **success, 'status': 'active', 'activated_at': activated,
                   'manifest_sha256': digest(root / 'build_stamp.json'),
                   'qualification_sha256': digest(bundle / 'promotion.json')}
        save(receipt_path, receipt)
        return receipt
    except BaseException:
        with ignore_cancellation():
            if copied:
                # Preserve NEW reservations and measurements produced by the failed
                # release. Only code and its launch contract go backwards.
                table = processes()
                live = [pid for pid, (_, args) in table.items() if
                        'nanobody_opt_v2.run_live_producer' in args or
                        any(a.endswith('/publish_current_nanobody.py') for a in args)]
                stop(live)
                paused.clear()
                install(backup, root, baseline)
                for name in set(candidate['runtime_files']) - set(baseline['runtime_files']):
                    safe_path(root, name).unlink(missing_ok=True)
                shutil.copy2(backup / 'producer_argv.txt', root / 'artifacts/producer_argv.txt')
                quarantine_derived(root, attempt / 'failed-release-derived', epoch=qualification['activation_epoch'])
                verify(root, baseline, expected_sha=baseline['git_sha'], runtime_only=True, require_tests=False)
                if old_receipt:
                    save(receipt_path, old_receipt)
                else:
                    receipt_path.unlink(missing_ok=True)
                launch(publisher)
                save(attempt / 'promotion_failure.json', {'status': 'rolled_back', 'candidate_sha': candidate['git_sha'],
                    'baseline_sha': baseline['git_sha'], 'durable_state_restored': False, 'at': time.time()})
        raise
    finally:
        for pid in paused:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, default=Path('/root'))
    parser.add_argument('--host', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    try:
        require(args.runtime_root.resolve() == Path('/root'), 'production_root_required')
        lock_path = args.runtime_root / 'artifacts/release_promotion.lock'
        with cleanup_on_disconnect(), lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = promote(args.stage, args.bundle, args.runtime_root, host=args.host, target=args.target)
        print(json.dumps({k: result[k] for k in ['status', 'candidate_sha', 'baseline_sha', 'activation_epoch']}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print(json.dumps({'status': 'promotion_refused_or_rolled_back'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
