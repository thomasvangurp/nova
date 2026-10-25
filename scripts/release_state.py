#!/usr/bin/env python3
"""Snapshot durable state and quarantine only explicitly derived cache files.

Snapshots are forensic evidence, never a restore source. Rewinding submission
history, raw measurements or NEW reservations could create duplicate submissions
or overspend. Rollback trials run the old code against the NEW durable state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.promotion_gate import digest, require, sha
from scripts.release_manifest import safe_path

DURABLE = ('artifacts/nanobody_published_sequences.txt',
    'artifacts/nanobody_published_sequences.publication_events.json',
    'artifacts/pi_submission_outcomes.json', 'artifacts/nanobody_mutations.sqlite')
DERIVED = ('artifacts/nanobody_publisher_state.json',)


def protected_paths(root, epoch):
    paths = [root / name for name in DURABLE]
    for e in [epoch - 1, epoch, epoch + 1]:
        paths.extend([root / f'live/epoch{e}/producer/fresh_screen_budget.json',
                      root / f'live/epoch{e}/committed_uids.json'])
    paths += sorted((root / 'artifacts/nanobody_history').glob('epoch*.json'))
    return [p for p in paths if p.is_file()]


def capture(root, output, *, epoch, baseline_sha, candidate_sha):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(type(epoch) is int and epoch > 0 and sha(baseline_sha, 40) and sha(candidate_sha, 40), 'state_scope')
    require(not output.exists() and not root.is_relative_to(output), 'snapshot_destination_exists')
    require((root / DURABLE[0]).is_file() and (root / DURABLE[3]).is_file(), 'missing_durable_state')
    output.mkdir(parents=True, mode=0o700)
    files = {}
    for path in protected_paths(root, epoch):
        name = path.relative_to(root).as_posix()
        safe_path(root, name)
        destination = safe_path(output, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.sqlite':
            # SQLite backup includes committed WAL data; a raw file copy does not.
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as source:
                with sqlite3.connect(destination) as target:
                    source.backup(target)
                    require(target.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'snapshot_sqlite_invalid')
        else:
            shutil.copyfile(path, destination)
        destination.chmod(0o600)
        files[name] = digest(destination)
    receipt = {'schema': 'sn68-state-snapshot-v1', 'baseline_sha': baseline_sha,
        'candidate_sha': candidate_sha, 'epoch': epoch, 'created_at': time.time(),
        'files': files, 'restores_durable_state': False,
        'policy': 'preserve-durable-invalidate-derived-v1'}
    (output / 'state_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def quarantine_derived(root, output, *, epoch):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(type(epoch) is int and epoch > 0, 'state_scope')
    require(not output.exists() and not root.is_relative_to(output), 'quarantine_destination_exists')
    # This operation must be inside a controlled, quiescent code rollback.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            require(not any(x in args for x in [b'nanobody_opt_v2.run_live_producer',
                    b'nanobody_opt_v2.sequential_exact_race']) and not any(
                    x.endswith(b'/publish_current_nanobody.py') for x in args), 'runtime_not_quiescent')
        except OSError:
            continue
    before = {str(p): digest(p) for p in protected_paths(root, epoch)}
    names = [*DERIVED]
    for suffix in ['comparison_precheck.json', 'comparison_candidate_scope.txt',
                   'producer/full_scan_confirmation.json', 'producer/confirmation_attempt.json',
                   'producer/cached_confirmation/comparison_precheck.json']:
        names.append(f'live/epoch{epoch}/{suffix}')
    output.mkdir(parents=True, mode=0o700)
    moved = []
    for name in names:
        path = safe_path(root, name)
        if path.is_file():
            destination = safe_path(output, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
            moved.append(name)
    require(all(Path(p).is_file() and digest(p) == value for p, value in before.items()), 'durable_state_changed')
    return {'policy': 'preserve-durable-invalidate-derived-v1', 'quarantined': moved,
            'protected_files': len(before), 'restores_durable_state': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['capture', 'quarantine-derived'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--baseline-sha')
    parser.add_argument('--candidate-sha')
    args = parser.parse_args()
    try:
        result = (capture(args.root, args.output, epoch=args.epoch, baseline_sha=args.baseline_sha,
                          candidate_sha=args.candidate_sha) if args.action == 'capture' else
                  quarantine_derived(args.root, args.output, epoch=args.epoch))
        print(json.dumps({'status': 'ok', 'action': args.action,
                          'files': len(result.get('files', result.get('quarantined', [])))}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        print(json.dumps({'status': 'state_operation_refused'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
