#!/usr/bin/env python3
"""Read-only artifact allocation through the Pi's installed consumer.

Never invokes main, pull, commit, wallets, chain clients, or the submission loop.
The supplied artifact is a separate trial file, never the Pi's live input.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.promotion_gate import digest, read, require, sha


def collect(artifact, consumer_root, *, revision, target, epoch, uids, publication_mtime=None):
    artifact, consumer_root = Path(artifact).resolve(), Path(consumer_root).resolve()
    require(consumer_root == Path('/opt/nova-submit') and platform.system() == 'Linux'
            and platform.machine() in {'aarch64', 'arm64'}, 'receipt_requires_pi')
    require(sha(revision, 40) and len(uids) >= 1 and len(uids) == len(set(uids))
            and all(type(uid) is int and uid >= 0 for uid in uids), 'invalid_trial_crew')
    require(not artifact.is_relative_to(consumer_root), 'trial_must_not_use_live_input')
    before = digest(artifact)
    payload = read(artifact)
    require(payload['target'] == target and payload['epoch'] == epoch, 'trial_artifact_scope')
    # The installed consumer supports legacy publications without an embedded
    # clock. For those, use the producer-observed mtime of the ORIGINAL file,
    # never the fresh mtime of this copied trial.
    published = payload.get('updated_at_unix', publication_mtime)
    require(isinstance(published, (int, float)) and not isinstance(published, bool)
            and math.isfinite(published), 'missing_publication_clock')
    require(0 <= time.time() - published <= 900, 'stale_trial_artifact')
    consumer = consumer_root / 'pi_submit.py'
    consumer_hash = digest(consumer)
    spec = importlib.util.spec_from_file_location('sn68_installed_pi_consumer', consumer)
    module = importlib.util.module_from_spec(spec)
    # Consumer diagnostics stay private, even on an exception.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
        allocated = module.sequences_by_uid(payload, epoch, allow_stale=False,
                                             crew=[(uid, '') for uid in uids])
    require(set(allocated) == set(uids) and len(set(allocated.values())) == len(uids),
            'pi_consumer_shortfall')
    require(digest(artifact) == before and digest(consumer) == consumer_hash, 'pi_trial_mutated_input')
    return {'schema': 'sn68-pi-shadow-v1', 'candidate_sha': revision, 'target': target,
        'epoch': epoch, 'host': 'pi', 'read_only': True, 'submission_attempts': 0,
        'accepted_entries': len(allocated), 'rejected_entries': 0, 'artifact_sha256': before,
        'consumer_sha256': consumer_hash, 'verified_at': time.time()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--consumer-root', type=Path, default=Path('/opt/nova-submit'))
    parser.add_argument('--candidate-sha', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--uids', type=int, nargs='+', required=True)
    parser.add_argument('--publication-mtime', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), 'receipt_already_exists')
        receipt = collect(args.artifact, args.consumer_root, revision=args.candidate_sha,
                          target=args.target, epoch=args.epoch, uids=args.uids,
                          publication_mtime=args.publication_mtime)
        with args.output.open('x') as handle:
            handle.write(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'status': 'pi_trial_passed', 'accepted_entries': receipt['accepted_entries']}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        print(json.dumps({'status': 'pi_trial_refused'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
