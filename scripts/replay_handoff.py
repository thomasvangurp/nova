#!/usr/bin/env python3
"""Replay recorded publication artifacts through extracted installed Pi code.

This qualifies release-controls only. It does not simulate GPU measurements or
assert an improvement in biological performance. Application source equivalence
is checked independently on the host before and after control installation.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.promotion_gate import digest, read, require, referenced, sha


def allocation_digest(rows):
    return hashlib.sha256(json.dumps(sorted((int(k), str(v)) for k, v in rows.items()),
                                     separators=(',', ':')).encode()).hexdigest()


def replay(fixtures):
    fixtures = Path(fixtures)
    fixture = read(fixtures / 'fixture.json')
    require(fixture['schema'] == 'sn68-recorded-handoff-v1', 'handoff_fixture_schema')
    from scripts.release_manifest import safe_path
    consumer = safe_path(fixtures, fixture['consumer']['path'])
    require(digest(consumer) == fixture['consumer']['sha256']
            and sha(fixture['installed_consumer_sha256']), 'consumer_source_mismatch')
    spec = importlib.util.spec_from_file_location('recorded_pi_consumer', consumer)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    epochs = set()
    results = []
    for case in fixture['cases']:
        artifact = referenced(fixtures, case['artifact'])
        require(artifact['epoch'] == case['epoch'] and artifact['target'] == fixture['target'], 'recorded_scope')
        # Historical replay uses the archived publication clock. It never
        # changes the artifact timestamp or calls the Pi's live submission loop.
        module.time = SimpleNamespace(time=lambda: case['replay_time'])
        crew = [(uid, '') for uid in case['uids']]
        require(len(crew) >= 1 and len(set(case['uids'])) == len(crew), 'invalid_recorded_crew')
        before = digest(fixtures / case['artifact']['path'])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            allocated = module.sequences_by_uid(artifact, case['epoch'], allow_stale=False, crew=crew)
        require(set(allocated) == set(case['uids']) and len(set(allocated.values())) == len(crew),
                'recorded_fleet_shortfall')
        actual = allocation_digest(allocated)
        require(actual == case['expected_allocation_sha256'], 'recorded_allocation_changed')
        require(digest(fixtures / case['artifact']['path']) == before, 'replay_mutated_input')
        # The same real artifact must be rejected in a later epoch. This is an
        # explicit injected failure, not a claim it happened in production.
        refused = False
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                module.sequences_by_uid(artifact, case['epoch'] + 1, allow_stale=False, crew=crew)
        except ValueError:
            refused = True
        require(refused, 'wrong_epoch_accepted')
        epochs.add(case['epoch'])
        results.append({'epoch': case['epoch'], 'entries': len(allocated), 'allocation_sha256': actual,
                        'wrong_epoch_refused': True, 'artifact_sha256': before})
    require(len(epochs) >= 3, 'insufficient_recorded_epochs')
    return {'schema': 'sn68-control-replay-v1', 'status': 'passed',
            'fixture_sha256': digest(fixtures / 'fixture.json'),
            'consumer_sha256': fixture['consumer']['sha256'],
            'installed_consumer_sha256': fixture['installed_consumer_sha256'],
            'cases': results, 'submission_attempts': 0, 'scoring_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = replay(args.fixtures)
        with args.output.open('x') as handle:
            handle.write(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'status': 'replay_passed', 'epochs': len(result['cases'])}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        print(json.dumps({'status': 'replay_refused'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
