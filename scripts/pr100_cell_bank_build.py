#!/usr/bin/env python3
"""Fill the PR100 challenge-indexed bank, bucket by bucket, off the epoch clock.

The builder drives the **existing** worker phases through the same ``job.json``
contract the live controller uses (docs/PR100_LIVE_CONTRACT.md), so nothing about
annotation, the native repeat-and-margin gate, novelty or scoring is reimplemented
here. It only decides *which bucket next*, *how much compute may be spent* and
*what gets written into the bank*.

No epoch hash is required: a bucket's challenge is produced by
``pr100_cell_bank.synthetic_challenge``, which searches block hashes until
``rules.challenge`` draws that (germline, CDR3 length) bucket. The result is a
genuine challenge object -- the worker phases verify it on entry exactly as they
do live -- but the hash is **not** a chain hash.

What still needs the real epoch-start hash, at use time only:

* ``rules.verified_fleet``, ``pr100_live_trial_export.build`` and the Pi commit,
  all of which recompute the challenge from the live hash;
* ``pr100_evidence``, which is why the bank stores the annotation and the raw
  measurements and lets ``CellBank.select_for_epoch`` rebuild the evidence around
  the live challenge instead of storing a build-time evidence blob.

Budgets are enforced before a bucket starts, never mid-bucket, so an interrupted
run resumes from its progress file without losing a measured candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt import pr100_cell_bank as bank_module  # noqa: E402
from nanobody_opt import rules  # noqa: E402

PHASES = ('sequence', 'native', 'score')


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def progress_path(run_root):
    return Path(run_root) / 'progress.json'


def read_progress(run_root):
    value = load_json(progress_path(run_root))
    if not isinstance(value, dict) or value.get('schema_version') != bank_module.SCHEMA_VERSION:
        return {'schema_version': bank_module.SCHEMA_VERSION, 'buckets': {},
                'spent': {'core_hours': 0.0, 'gpu_hours': 0.0}}
    value.setdefault('buckets', {})
    value.setdefault('spent', {'core_hours': 0.0, 'gpu_hours': 0.0})
    return value


def write_progress(run_root, progress):
    bank_module._atomic_json(progress_path(run_root), progress)


def plan_buckets(bank, config, args, *, now=None):
    """Priority-ordered buckets plus the cost model, truncated by the budgets."""
    status = bank.bucket_status(args.target, config=config,
                                per_bucket_target=args.per_bucket_target,
                                runtime_identity=None, now=now)
    rows = bank_module.build_priority(config, status, per_bucket_target=args.per_bucket_target)
    if args.buckets:
        wanted = set(args.buckets)
        rows = [r for r in rows if r['bucket_id'] in wanted]
    plan, core_hours, gpu_hours = [], 0.0, 0.0
    for row in rows:
        cost = bank_module.budget_estimate([row], repeats=args.repeats)
        if args.max_buckets is not None and len(plan) >= args.max_buckets:
            break
        if args.core_hour_budget is not None and core_hours + cost['native_core_hours'] > args.core_hour_budget:
            break
        if args.gpu_hour_budget is not None and gpu_hours + cost['gpu_hours'] > args.gpu_hour_budget:
            break
        core_hours += cost['native_core_hours']
        gpu_hours += cost['gpu_hours']
        plan.append({**row, 'estimate': cost})
    return {'target': args.target, 'status': {k: v for k, v in status.items() if k != 'missing'},
            'buckets': plan, 'total': bank_module.budget_estimate(plan, repeats=args.repeats),
            'planned_core_hours': core_hours, 'planned_gpu_hours': gpu_hours}


def bucket_jobs(args, challenge, directory, archive_sha256, archive_fetched_at):
    """One job.json per phase, keyed exactly as the live contract specifies."""
    jobs = {}
    common = {'challenge': challenge, 'archive_sha256': archive_sha256,
              'archive_fetched_at': archive_fetched_at}
    jobs['sequence'] = {**common, 'phase': 'sequence', 'cpu_slots': args.sequence_cpu_slots,
                        'proposals': args.proposals, 'reference_slots': args.reference_slots,
                        'references': args.references}
    jobs['native'] = {**common, 'phase': 'native', 'cpu_slots': args.native_cpu_slots,
                      'native_tools': args.native_tools, 'workers': args.workers,
                      'max_workers': args.max_workers, 'repeats': args.repeats,
                      'allow_marginal': True,
                      'candidate_budget_seconds': args.candidate_budget_seconds}
    if args.native_phase_budget_seconds:
        jobs['native']['phase_budget_seconds'] = args.native_phase_budget_seconds
    if args.gpus:
        jobs['score'] = {**common, 'phase': 'score', 'cpu_slots': max(2, len(args.gpus)),
                         'gpus': list(args.gpus), 'release': args.release,
                         'source_sha': args.source_sha, 'scorer': args.scorer,
                         'scorer_settings': args.scorer_settings,
                         'gpu_lock_directory': args.gpu_lock_directory,
                         'output_directory': str(Path(directory) / 'score' / 'out'),
                         'wall_seconds': args.score_wall_seconds}
    return jobs


def worker_command(args, job_path):
    return [args.python, '-B', '-m', 'nanobody_opt.pr100_live_worker', '--job', str(job_path)]


def default_runner(args, phase, job_path, directory):
    """Run one worker phase locally; the caller owns host placement and idleness."""
    log = Path(directory) / f'{phase}.log'
    with open(log, 'ab') as handle:
        completed = subprocess.run(worker_command(args, job_path), cwd=args.release,
                                   stdout=handle, stderr=handle, stdin=subprocess.DEVNULL,
                                   check=False)
    return {'returncode': completed.returncode,
            'result': load_json(Path(directory) / 'result.json')}


def ingest_native(bank, args, challenge, result, *, now=None):
    """Write every native passer (tier A and tier B) into the bank."""
    stored = []
    runtime = bank_module.measurement_runtime(
        annotation_identity=(result.get('candidates') or [{}])[0]
        .get('pr100_annotation', {}).get('annotation_identity'),
        native_runtime=args.native_runtime_identity, host=args.host_identity,
        measured_at_unix=now)
    provenance = {'source_sha': args.source_sha, 'builder': 'pr100_cell_bank_build',
                  'synthetic_challenge_hash': challenge['epoch_start_block_hash'],
                  'builder_epoch': challenge['epoch'], 'host': args.host_identity,
                  'created_at_unix': time.time() if now is None else float(now)}
    archive = result.get('archive_sha256')
    view = None
    if args.archive:
        view = bank_module.archive_view(args.archive)
    for row in result.get('candidates') or []:
        novelty = dict(row.get('novelty') or {})
        if not novelty:
            continue  # novelty_checked false: not storable as a watermarked record
        novelty.update(archive_sha256=archive, archive_rows=result.get('archive_rows'),
                       checked_at_unix=result.get('archive_checked_at_unix'))
        if view is not None and view['sha256'] == archive:
            novelty['archive_prefix_sha256'] = bank_module.prefix_digest(
                view['sequences'], result.get('archive_rows') or view['rows'])
        record = bank_module.build_record(
            target=challenge['target'], v_gene=challenge['allowed_v'],
            cdr3_length=challenge['allowed_cdr3_length'], sequence=row['sequence'],
            annotation=row['pr100_annotation'], measurements=row['measurements'],
            runtime=runtime, novelty=novelty, provenance=provenance, tier=row.get('tier'),
            repeats=row.get('repeats'), now=now)
        bank.add(record)
        stored.append(record['sequence_hash'])
    return stored


def ingest_scores(bank, args, challenge, native_result, score_directory, *, now=None):
    """Attach score observations to the records just stored for this bucket."""
    out = Path(score_directory)
    report = load_json(out / 'report.json')
    settings_raw = (out / 'scorer_settings.json').read_bytes() if (out / 'scorer_settings.json').is_file() else None
    if not report or not settings_raw or not report.get('registered_scorer_sha256'):
        return []
    settings = json.loads(settings_raw)
    settings_file_sha256 = hashlib.sha256(settings_raw).hexdigest()
    metrics_by_sequence = {}
    for group in report.get('groups', []):
        if group.get('valid') is not True:
            continue
        value = load_json(out / str(group.get('file')))
        if not value or value.get('target') != challenge['target']:
            continue
        metrics_by_sequence.update(value.get('per_nanobody_components', {}).get('0', {}))
    bucket = bank_module.challenge_bucket(challenge)
    updated = []
    for record in bank.records(challenge['target'], bucket):
        metrics = (metrics_by_sequence.get(record['sequence']) or {}).get(challenge['target'])
        if not metrics:
            continue
        observation = bank_module.score_observation(
            target=challenge['target'], metrics=metrics,
            scorer_sha256=report['registered_scorer_sha256'], settings=settings,
            settings_file_sha256=settings_file_sha256, observed_at_unix=now)
        bank.add({**record, 'scores': list(record.get('scores') or []) + [observation],
                  'updated_at_unix': time.time() if now is None else float(now)})
        updated.append(record['sequence_hash'])
    return updated


def run_bucket(bank, args, row, run_root, *, runner=default_runner, now=None):
    """Drive sequence -> native -> score for one bucket; resumable per phase."""
    config = rules.pinned_config(args.epoch)
    challenge = bank_module.synthetic_challenge(config, args.epoch, row['v_gene'],
                                                row['cdr3_length'],
                                                epoch_length=args.epoch_length)
    directory = Path(run_root) / row['bucket_id'].replace(':', '__cdr3')
    archive_sha256, archive_fetched_at = args.archive_sha256, time.time()
    jobs = bucket_jobs(args, challenge, directory, archive_sha256, archive_fetched_at)
    outcome = {'bucket_id': row['bucket_id'], 'challenge_hash': challenge['epoch_start_block_hash'],
               'phases': {}, 'stored': [], 'scored': []}
    native_result = None
    for phase in PHASES:
        if phase not in jobs:
            continue
        phase_directory = directory / phase
        phase_directory.mkdir(parents=True, exist_ok=True)
        existing = load_json(phase_directory / 'result.json')
        if existing is not None and existing.get('status') in ('complete', 'partial'):
            outcome['phases'][phase] = 'resumed'
            result = existing
        else:
            job = dict(jobs[phase])
            if phase == 'native':
                previous = load_json(directory / 'sequence' / 'result.json')
                if not previous or not previous.get('candidates'):
                    outcome['phases'][phase] = 'no_input'
                    break
                bank_module._atomic_json(phase_directory / 'input.json', previous)
            if phase == 'score':
                if not native_result or not native_result.get('candidates'):
                    outcome['phases'][phase] = 'no_input'
                    break
                bank_module._atomic_json(
                    phase_directory / 'input.json',
                    {'sequences': [r['sequence'] for r in native_result['candidates']]})
            if args.archive and phase in ('sequence', 'native'):
                destination = phase_directory / 'archive.csv'
                if not destination.is_file():
                    destination.write_bytes(Path(args.archive).read_bytes())
            bank_module._atomic_json(phase_directory / 'job.json', job)
            reply = runner(args, phase, phase_directory / 'job.json', phase_directory)
            outcome['phases'][phase] = ('ok' if reply.get('returncode') == 0 else 'failed')
            result = reply.get('result')
            if result is None:
                break
        if phase == 'native':
            native_result = result
            outcome['stored'] = ingest_native(bank, args, challenge, result, now=now)
        if phase == 'score':
            outcome['scored'] = ingest_scores(bank, args, challenge, native_result,
                                              jobs['score']['output_directory'], now=now)
    return outcome


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bank', required=True, type=Path, help='bank root directory')
    parser.add_argument('--run-root', required=True, type=Path, help='job + progress directory')
    parser.add_argument('--release', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--target', default=rules.TARGET)
    parser.add_argument('--epoch', type=int, default=rules.ACTIVATION_EPOCH,
                        help='pinned-config epoch for the synthetic challenge (no chain read)')
    parser.add_argument('--epoch-length', type=int, default=361)
    parser.add_argument('--per-bucket-target', type=int, default=bank_module.DEFAULT_BUCKET_TARGET)
    parser.add_argument('--buckets', nargs='*', default=None,
                        help='explicit bucket ids ("IGHV3-23:18"); default is every short bucket')
    parser.add_argument('--max-buckets', type=int, default=None)
    parser.add_argument('--core-hour-budget', type=float, default=None)
    parser.add_argument('--gpu-hour-budget', type=float, default=None)
    parser.add_argument('--wall-seconds', type=float, default=None)
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--max-workers', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--proposals', type=int, default=600)
    parser.add_argument('--reference-slots', type=int, default=16)
    parser.add_argument('--candidate-budget-seconds', type=float, default=300.0)
    parser.add_argument('--native-phase-budget-seconds', type=float, default=None)
    parser.add_argument('--sequence-cpu-slots', type=int, default=4)
    parser.add_argument('--native-cpu-slots', type=int, default=8)
    parser.add_argument('--gpus', nargs='*', type=int, default=[],
                        help='idle gpu ids; empty means a CPU-only (native) build')
    parser.add_argument('--score-wall-seconds', type=float, default=1200.0)
    parser.add_argument('--gpu-lock-directory', default='/tmp/pr100-gpu-locks')
    parser.add_argument('--scorer', default='registered')
    parser.add_argument('--scorer-settings', default=None)
    parser.add_argument('--references', default=None)
    parser.add_argument('--native-tools', default=None)
    parser.add_argument('--archive', type=Path, default=None,
                        help='archive csv snapshot; required to store a novelty watermark')
    parser.add_argument('--source-sha', default=None)
    parser.add_argument('--native-runtime-identity', default=None)
    parser.add_argument('--host-identity', default=None)
    parser.add_argument('--dry-run', action='store_true',
                        help='print the plan and the cost model; run no phase, write no record')
    args = parser.parse_args(argv)
    args.archive_sha256 = (hashlib.sha256(Path(args.archive).read_bytes()).hexdigest()
                           if args.archive else None)
    for name in ('workers', 'max_workers', 'repeats', 'proposals', 'reference_slots',
                 'per_bucket_target', 'epoch_length'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be a positive integer')
    if not args.dry_run:
        missing = [n for n in ('source_sha', 'native_tools', 'references', 'archive')
                   if getattr(args, n) is None]
        if missing:
            parser.error('a real build needs: ' + ', '.join('--' + n.replace('_', '-')
                                                            for n in missing))
        if args.gpus and args.scorer_settings is None:
            parser.error('scoring needs --scorer-settings')
    return args


def main(argv=None, *, runner=default_runner, now=None):
    args = parse_args(argv)
    config = rules.pinned_config(args.epoch)
    bank = bank_module.CellBank(args.bank)
    plan = plan_buckets(bank, config, args, now=now)
    args.run_root.mkdir(parents=True, exist_ok=True)
    bank_module._atomic_json(args.run_root / 'plan.json', plan)
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'buckets': [r['bucket_id'] for r in plan['buckets']],
                          'planned_core_hours': round(plan['planned_core_hours'], 2),
                          'planned_gpu_hours': round(plan['planned_gpu_hours'], 3),
                          'coverage_probability': plan['status'].get('probability_any'),
                          'estimate': plan['total']}, indent=2, sort_keys=True))
        return 0
    bank.ensure()
    progress = read_progress(args.run_root)
    started = time.monotonic()
    for row in plan['buckets']:
        if progress['buckets'].get(row['bucket_id'], {}).get('status') == 'done':
            continue
        if args.wall_seconds is not None and time.monotonic() - started > args.wall_seconds:
            progress['stopped_on'] = 'wall_budget'
            break
        outcome = run_bucket(bank, args, row, args.run_root, runner=runner, now=now)
        estimate = row['estimate']
        progress['spent']['core_hours'] += estimate['native_core_hours']
        progress['spent']['gpu_hours'] += estimate['gpu_hours']
        progress['buckets'][row['bucket_id']] = {'status': 'done', **outcome}
        write_progress(args.run_root, progress)
    progress['finished_at_unix'] = time.time()
    write_progress(args.run_root, progress)
    print(json.dumps({'buckets': len(progress['buckets']), 'spent': progress['spent'],
                      'stopped_on': progress.get('stopped_on')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
