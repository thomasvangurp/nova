"""Measured baseline phases for the Pi-owned PR100 epoch controller."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
from nanobody_opt import rules, pr100_filter, pr100_live_selection as selection

DEFAULT_WORKERS = 48
DEFAULT_MAX_WORKERS = 64
DEFAULT_CANDIDATE_BUDGET = 300.0
DEFAULT_PROPOSALS = 600
DEFAULT_REFERENCE_SLOTS = 16
WORKER_RSS_BYTES = 1.4 * 1024 ** 3


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def available_cores():
    getter = getattr(os, 'sched_getaffinity', None)
    return len(getter(0)) if getter else (os.cpu_count() or 1)


def available_memory_bytes():
    try:
        page = os.sysconf('SC_PAGE_SIZE')
        return os.sysconf('SC_AVPHYS_PAGES') * page
    except (OSError, ValueError, AttributeError):
        return None


def worker_count(job, *, cores=None, memory=None):
    """Concurrency for native profiling: 1 thread per job, memory-bounded.

    Contention was measured at 32 and is unmeasured above it, so the default is
    deliberately conservative rather than the largest value the box could fit.
    """
    cap = job.get('max_workers', DEFAULT_MAX_WORKERS)
    requested = job.get('workers', DEFAULT_WORKERS)
    for name, value in (('max_workers', cap), ('workers', requested)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + ' must be a positive integer')
    cores = available_cores() if cores is None else cores
    memory = available_memory_bytes() if memory is None else memory
    limit = min(cap, requested, max(1, cores))
    if memory:
        limit = min(limit, max(1, int(memory // WORKER_RSS_BYTES)))
    return max(1, limit)


def candidate_budget(job):
    budget = job.get('candidate_budget_seconds', DEFAULT_CANDIDATE_BUDGET)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
        raise ValueError('candidate_budget_seconds must be a positive number')
    return float(budget)


def measure_batch(sequences, workers, *, clock=time.monotonic, develop=None,
                  on_result=None, on_result_error=None):
    """Profile a batch and report wall measured from slot acquisition.

    The upstream ProfileExecution timeout starts before the semaphore is
    acquired, so a queued candidate can burn its whole budget waiting. We never
    enqueue more profiles than can start at once, and the wall we attribute to a
    candidate begins here, when its slot is already held.
    """
    develop = develop or pr100_filter.develop
    acquired = clock()
    callbacks = ({'on_result': on_result, 'on_result_error': on_result_error}
                 if on_result is not None else {})
    values = asyncio.run(develop(list(sequences), workers, **callbacks))
    return values, acquired, clock() - acquired


def sequence_phase(job, directory):
    from abnumber import Chain
    import pandas as pd
    from nanobody_opt import gates, pr100_proposal_pilot as proposals
    from nanobody_opt import check_validator_novelty as novelty
    expected = job['challenge']
    config = rules.pinned_config(expected['epoch'])
    pr100_filter.validate_annotation_runtime()
    archive = directory / 'archive.csv'
    if file_digest(archive) != job['archive_sha256']:
        raise ValueError('archive snapshot changed')
    bank = json.loads(Path(job['references']).read_text())['rows']
    cell = f"{expected['allowed_v']}:{expected['allowed_cdr3_length']}"
    slots = job.get('reference_slots', DEFAULT_REFERENCE_SLOTS)
    proposals_wanted = job.get('proposals', DEFAULT_PROPOSALS)
    for name, value in (('reference_slots', slots), ('proposals', proposals_wanted)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + ' must be a positive integer')
    refs = sorted((r for r in bank if cell in r['cells']),
                  key=lambda r: rules.digest([expected['epoch'], r['sequence_hash']]))[:slots]
    panels, _ = pr100_filter.helper('germline_identity')._anarci_panel('human')
    rows = [{**r, 'source': 'existing_native_reference'} for r in refs]
    for ordinal in range(proposals_wanted):
        reference_share = proposals_wanted // 2
        ref = refs[ordinal % len(refs)] if refs and ordinal < reference_share else None
        cdr3 = (Chain(ref['sequence'], scheme='imgt', cdr_definition='imgt',
                      use_anarcii=False).cdr3_seq if ref else None)
        row = proposals.propose(panels, expected['allowed_v'], expected['allowed_cdr3_length'],
                                seed=expected['epoch'], ordinal=ordinal, reference_cdr3=cdr3)
        rows.append({**row, 'source': 'fresh_baseline_proposal',
                     'reference_hash': ref['sequence_hash'] if ref else None})
    unique = {r['sequence']: r for r in rows}
    eligible, reasons = [], {}
    for sequence, row in unique.items():
        annotation = {**pr100_filter._annotate_uncached(sequence),
                      'annotation_identity': pr100_filter.annotation_identity()}
        reason = ('basic_sequence' if gates.check_sequence(sequence, config['nanobody_requirements'])
                  else rules.annotation_reason(annotation, config['nanobody_requirements'],
                                               sequence=sequence, epoch_challenge=expected))
        reasons[reason or 'passed'] = reasons.get(reason or 'passed', 0) + 1
        if not reason:
            eligible.append({**row, 'sequence_hash': rules.sequence_hash(sequence),
                             'pr100_annotation': annotation})
    frame = pd.read_csv(archive)
    # Novelty is deliberately NOT run here: the archive search now runs after
    # native developability, so it is spent only on native passers.
    result = {'challenge': expected, 'target': expected['target'], 'epoch': expected['epoch'],
              'archive_sha256': job['archive_sha256'], 'archive_checked_at_unix': job['archive_fetched_at'],
              'archive_rows': len(frame), 'existing_references': len(refs),
              'proposals': proposals_wanted, 'annotation_reasons': reasons,
              'candidates': eligible, 'novelty_checked': False,
              'annotated_total': len(eligible), 'status': 'complete'}
    pr100_filter.atomic(directory / 'result.json', result)
    return result


def check_novelty(rows, archive, *, novelty=None):
    """Archive novelty, run only on candidates that already passed native."""
    import pandas as pd
    if novelty is None:
        from nanobody_opt import check_validator_novelty as novelty
    frame = pd.read_csv(archive)
    novelty._init_worker(frame, set(frame.sequence.dropna().astype(str).str.strip().str.upper()),
                         all_archive=True)
    kept = []
    for row in rows:
        found = novelty._search_one(row)
        if found['validator_novel']:
            kept.append({**row, 'novelty': {k: found[k] for k in (
                'validator_novel', 'exact_historical', 'coarse_candidates', 'maximum_identity')}})
    return kept, len(frame)


def native_phase(job, directory, *, clock=time.monotonic, measure=None, novelty=None):
    from nanobody_opt import pr100_native_journal as journal_module
    incoming = json.loads((directory / 'input.json').read_text())
    if incoming['challenge'] != job['challenge']:
        raise ValueError('native input challenge mismatch')
    tools = Path(job['native_tools'])
    pr100_filter.helper('aggrescan3d').tool_paths = lambda: (tools / 'aggrescan3d-rs', tools / 'freesasa')
    workers = worker_count(job)
    budget = candidate_budget(job)
    repeats = job.get('repeats', selection.DEFAULT_REPEATS)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    margin = selection.margin_settings(job.get('margin'))
    allow_marginal = job.get('allow_marginal', True)
    if not isinstance(allow_marginal, bool):
        raise ValueError('allow_marginal must be a boolean')
    config = rules.pinned_config(job['challenge']['epoch'])
    requirements = config['nanobody_requirements']
    rows = incoming['candidates']
    if len({r['sequence'] for r in rows}) != len(rows):
        raise ValueError('duplicate sequence in native input')
    result = {k: incoming[k] for k in ('challenge', 'target', 'epoch', 'archive_sha256',
                                       'archive_checked_at_unix')}
    result.update(status='running', input_count=len(rows), completed=0,
                  not_started=len(rows), not_completed=len(rows), in_flight=0,
                  candidates=[], rejection_counts={}, cache_hits=0, workers=workers,
                  candidate_budget_seconds=budget, repeats=repeats, allow_marginal=allow_marginal,
                  margin={**margin, 'threshold_at_k': selection.margin_threshold(repeats, margin)},
                  novelty_checked=False, native_classification_complete=False,
                  observation_errors=0,
                  measurement_outcomes={'passed': 0, 'rejected': 0, 'inconclusive': 0})

    def record(reason):
        result['rejection_counts'][reason] = result['rejection_counts'].get(reason, 0) + 1

    def flush():
        result['observation_journal'] = journal.progress()
        pr100_filter.atomic(directory / 'result.json', result)

    phase_budget = job.get('phase_budget_seconds')
    if phase_budget is not None and (isinstance(phase_budget, bool)
                                     or not isinstance(phase_budget, (int, float))
                                     or phase_budget <= 0):
        raise ValueError('phase_budget_seconds must be a positive number')
    # Without an explicit phase budget the controller owns the wall clock; we do
    # not invent a deadline that would mark candidates not_run for no reason.
    deadline = None if phase_budget is None else clock() + float(phase_budget)

    def out_of_time():
        return deadline is not None and clock() + budget > deadline

    measurements = {}
    started = set()
    journal = journal_module.AsyncJournal(directory, job, requirements)
    result['observation_journal'] = journal.progress()
    flush()

    def observation_error(_error_type):
        # Diagnostics cannot convert a valid scientific result to failure.
        result['observation_errors'] += 1

    def measure_and_record(batch):
        by_sequence = {row['sequence']: row for row in batch}
        expected = {key: len(measurements.get(key, [])) + 1 for key in by_sequence}
        started.update(by_sequence)
        result['not_started'] = len(rows) - len(started)
        result['in_flight'] += len(batch)

        def completed(key, value):
            if len(measurements.get(key, [])) >= expected[key]:
                return
            measurements.setdefault(key, []).append(value)
            result['completed'] += 1
            result['not_completed'] = len(rows) - len(measurements)
            result['in_flight'] -= 1
            try:
                observed = journal_module.observation(value, requirements, key)
                result['measurement_outcomes'][observed['outcome']] += 1
                progress = {k: result[k] for k in ('status', 'input_count', 'completed',
                    'not_started', 'not_completed', 'in_flight', 'native_classification_complete',
                    'observation_errors')}
                progress['measurement_outcomes'] = dict(result['measurement_outcomes'])
                journal.save(by_sequence[key], expected[key], observed, progress=progress)
            except Exception as error:
                observation_error(type(error).__name__)

        if measure is None:
            values, acquired, elapsed = measure_batch(
                [r['sequence'] for r in batch], workers, clock=clock,
                on_result=completed, on_result_error=observation_error)
        else:
            values, acquired, elapsed = measure([r['sequence'] for r in batch], workers, clock=clock)
        for row, value in zip(batch, values, strict=True):
            completed(row['sequence'], value)
        return acquired, elapsed

    def collect_measurements():
        # Pass 1: one measurement each. Never enqueue more profiles than can start,
        # so a candidate's budget is never consumed waiting for a cpu slot.
        for offset in range(0, len(rows), workers):
            batch = rows[offset:offset + workers]
            if out_of_time():
                break
            acquired, elapsed = measure_and_record(batch)
            result['last_batch_seconds'] = elapsed
            flush()

        survivors = []
        for row in rows:
            key = row['sequence']
            if key not in measurements:
                continue
            first = selection.decide(measurements[key], requirements, sequence=key,
                                     repeats=repeats, margin=margin)
            # A failing first measurement is a real rejection; do not pay for repeats.
            if first['tier'] is None and first['reason'] not in ('insufficient_repeats', 'not_run'):
                record(first['reason'])
            else:
                survivors.append(row)
        # Pass 2: k-1 further independent measurements for first-pass survivors only.
        for _ in range(max(0, repeats - 1)):
            pending = [r for r in survivors if len(measurements[r['sequence']]) < repeats]
            if not pending:
                break
            for offset in range(0, len(pending), workers):
                batch = pending[offset:offset + workers]
                if out_of_time():
                    break
                _acquired, elapsed = measure_and_record(batch)
                result['last_batch_seconds'] = elapsed
                flush()

        return survivors

    try:
        survivors = collect_measurements()
    finally:
        journal.close(timeout=1.0)
        result['observation_journal'] = journal.progress()
    verdicts = {}

    passers = []
    for row in survivors:
        key = row['sequence']
        verdict = selection.decide(measurements[key], requirements, sequence=key,
                                   repeats=repeats, margin=margin)
        verdicts[key] = verdict
        if verdict['tier'] is None:
            record(verdict['reason'] or 'rejected')
            continue
        record('passed' if verdict['tier'] == 'A' else 'margin_missed')
        evidence = {'challenge': job['challenge'], 'filter_sha': rules.FILTER_SHA,
                    'annotation': row['pr100_annotation'],
                    'developability': verdict['measurements'][0]}
        rules.validate_evidence(evidence, key, job['challenge'], config)
        passers.append({**row, 'layer2_passed': True, 'pr100_evidence': evidence,
                        'tier': verdict['tier'], 'margin_safe': verdict['margin_safe'],
                        'measurements': verdict['measurements'],
                        'psh_values': verdict['psh_values'], 'psh_mean': verdict['psh_mean'],
                        'psh_sd': verdict['psh_sd'], 'repeats': verdict['repeats'],
                        'margin_threshold': verdict['margin_threshold']})
    result['native_passed'] = len(passers)
    result['native_classification_complete'] = True
    flush()

    # Funnel order: annotation -> native developability -> archive novelty.
    archive = directory / 'archive.csv'
    if passers and archive.is_file():
        if file_digest(archive) != job['archive_sha256']:
            raise ValueError('archive snapshot changed')
        passers, archive_rows = check_novelty(passers, archive, novelty=novelty)
        result.update(novelty_checked=True, archive_rows=archive_rows)
    result['candidates'] = selection.order_candidates(passers, allow_marginal=allow_marginal)
    result['tier_counts'] = {tier: sum(1 for r in result['candidates'] if r['tier'] == tier)
                             for tier in ('A', 'B')}
    result['not_run'] = len(rows) - len(started)
    if result['not_run']:
        # Never a failure: these candidates never acquired a cpu slot in budget.
        result['rejection_counts']['not_run'] = result['not_run']
    result['status'] = 'complete'
    flush()
    return result


SCORE_SHARD_LIMIT = 128  # pr100_gpu_probe accepts 1..128 distinct sequences per run
SCORE_ENV_KEYS = ('PATH', 'HOME', 'USER', 'LANG', 'LD_LIBRARY_PATH', 'PYTHONPATH',
                  'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
_SCORE_FILE = __import__('re').compile(r'gpu-[0-9]+/length[0-9]+/exact_all\.json')


def score_jobs(job, directory, sequences):
    """One cooperative GPU reservation job per GPU, each running pr100_gpu_probe (fresh, 1 pass)."""
    import sys
    from nanobody_opt.pr100_live_gpu import ENV_KEYS
    out = Path(job['output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    release = Path(job['release'])
    wall = float(job['wall_seconds'])
    # run_reserved accepts a TTL of 10..1800 s; the probe must finish inside the TTL.
    ttl = max(10.0, min(1800.0, wall - 20.0))
    probe_timeout = int(max(60, min(3600, ttl - 30)))
    shards = selection.shard([{'sequence': s} for s in sequences], list(job['gpus']))
    if any(len(rows) > SCORE_SHARD_LIMIT for rows in shards.values()):
        raise ValueError('score shard exceeds probe limit; use bounded screening waves')
    environment = {k: os.environ[k] for k in SCORE_ENV_KEYS if k in os.environ and k in ENV_KEYS}
    jobs = []
    for gpu in job['gpus']:
        rows = shards[gpu]
        if not rows:
            continue
        shard_input = out / f'shard-gpu-{gpu}.json'
        pr100_filter.atomic(shard_input, {'sequences': [r['sequence'] for r in rows]})
        command = [sys.executable, '-B', str(release / 'scripts/pr100_gpu_probe.py'),
                   '--release', str(release), '--expected-sha', job['source_sha'],
                   '--input', str(shard_input), '--output', str(out / f'gpu-{gpu}'),
                   '--settings', job['scorer_settings'], '--gpu', str(gpu), '--passes', '1',
                   '--mode', 'fresh', '--timeout', str(probe_timeout)]
        jobs.append({'gpu': gpu, 'ttl_seconds': ttl, 'command': command, 'env': environment,
                     'cwd': str(release), 'lock_directory': job['gpu_lock_directory'],
                     'scorer': job['scorer'], 'receipt': str(out / f'gpu-receipt-{gpu}.json'),
                     'candidate_count': len(rows)})
        if 'deadline_unix' in job:
            jobs[-1]['deadline_unix'] = job['deadline_unix']
    return jobs


def collect_scores(out, jobs):
    """Merge whatever per-GPU probe groups finished into score/report.json (partial allowed)."""
    groups, shards, scorer_hashes, settings = [], [], set(), None
    for gpu_job in jobs:
        gpu = gpu_job['gpu']
        entry = {'gpu': gpu, 'candidate_count': gpu_job['candidate_count']}
        try:
            entry['lease'] = json.loads(Path(gpu_job['receipt']).read_text()).get('status')
        except (OSError, ValueError):
            entry['lease'] = None
        try:
            report = json.loads((out / f'gpu-{gpu}' / 'report.json').read_text())
        except (OSError, ValueError):
            report = None
        entry['probe_status'] = report.get('status') if report else None
        if report:
            for group in report.get('groups', []):
                if group.get('valid') is not True:
                    continue
                name = f"gpu-{gpu}/length{int(group['length'])}/exact_all.json"
                if not (out / name).is_file():
                    continue
                groups.append({'gpu': gpu, 'length': int(group['length']), 'valid': True,
                               'candidates': group.get('candidates'), 'file': name})
                scorer_hashes.add(report['registered_scorer_sha256'])
                raw = (out / f'gpu-{gpu}' / 'scorer_settings.json').read_bytes()
                if settings is not None and raw != settings:
                    raise ValueError('scorer settings differ between gpu shards')
                settings = raw
        shards.append(entry)
    if len(scorer_hashes) > 1:
        raise ValueError('registered scorer differs between gpu shards')
    if settings is not None:
        (out / 'scorer_settings.json').write_bytes(settings)
    scored = sum(g['candidates'] or 0 for g in groups)
    complete = bool(jobs) and all(s['probe_status'] == 'complete' for s in shards)
    report = {'schema': 'pr100-live-score-v1', 'groups': groups, 'shards': shards,
              'registered_scorer_sha256': next(iter(scorer_hashes), None),
              'gpus_attempted': len(jobs),
              'gpus_finished': sum(1 for s in shards if s['probe_status'] == 'complete'),
              'scored_candidates': scored, 'status': 'complete' if complete else 'partial'}
    pr100_filter.atomic(out / 'report.json', report)
    return report


def score_phase(job, directory, *, launch=None, clock=time.monotonic, sleep=time.sleep):
    """Shard every candidate across ``gpus``; each GPU runs in its own cooperative lease.

    The per-GPU runners are separate sessions (the lease requires a session leader), so
    the supervisor's process-group kill cannot reach them: SIGTERM is forwarded here and
    every runner also carries its own TTL inside this phase's wall budget.
    """
    import signal
    import subprocess
    import sys
    sequences = json.loads((directory / 'input.json').read_text())['sequences']
    if len(set(sequences)) != len(sequences):
        raise ValueError('duplicate sequence in score input')
    if not sequences:
        raise ValueError('no candidates to score')
    if len(sequences) > SCORE_SHARD_LIMIT * len(job['gpus']):
        from nanobody_opt.pr100_wide_screen import screen_phase
        return screen_phase(job, directory, launch=launch, clock=clock, sleep=sleep)
    jobs = score_jobs(job, directory, sequences)
    out = Path(job['output_directory'])
    launch = launch or (lambda path, log: subprocess.Popen(
        [sys.executable, '-B', '-m', 'nanobody_opt.pr100_live_gpu', '--job', str(path)],
        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True))
    stopping = []
    previous = signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    runners = []
    try:
        for gpu_job in jobs:
            path = out / f"gpu-job-{gpu_job['gpu']}.json"
            pr100_filter.atomic(path, {k: v for k, v in gpu_job.items() if k != 'candidate_count'})
            log = open(out / f"gpu-runner-{gpu_job['gpu']}.log", 'ab')
            runners.append((launch(path, log), log))
        deadline = clock() + float(job['wall_seconds'])
        while any(p.poll() is None for p, _ in runners):
            if stopping or clock() >= deadline:
                for p, _ in runners:
                    if p.poll() is None:
                        p.send_signal(signal.SIGTERM)
                end = clock() + 6
                while clock() < end and any(p.poll() is None for p, _ in runners):
                    sleep(0.1)
                break
            sleep(0.5)
    finally:
        signal.signal(signal.SIGTERM, previous)
        for _, log in runners:
            log.close()
    report = collect_scores(out, jobs)
    scored = {}
    for group in report['groups']:
        value = json.loads((out / group['file']).read_text())
        scored.update(value.get('per_nanobody_components', {}).get('0', {}))
    result = {'challenge': job['challenge'], 'epoch': job['challenge']['epoch'],
              'status': report['status'], 'input_count': len(sequences),
              'scored_candidates': len(scored), 'gpus_attempted': report['gpus_attempted'],
              'gpus_finished': report['gpus_finished'],
              'candidates': [s for s in sequences if s in scored]}
    pr100_filter.atomic(directory / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    directory = args.job.parent
    expected = job['challenge']
    config = rules.pinned_config(expected['epoch'])
    if expected != rules.challenge(config, expected['epoch'], expected['epoch_start_block_hash'], expected['epoch_length']):
        raise ValueError('wrong current challenge')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:job['cpu_slots']])
    started = time.monotonic()
    try:
        from nanobody_opt.pr100_wide_screen import screen_phase
        result = {'sequence': sequence_phase, 'native': native_phase,
                  'score': score_phase, 'screen': screen_phase}[job['phase']](job, directory)
        # A partial score (some GPUs finished) is still a usable result for the controller.
        status = 'complete' if result['status'] in ('complete', 'partial') else result['status']
        receipt = {'status': status, 'result_status': result['status'],
                   'candidate_count': len(result['candidates']),
                   'not_run': result.get('not_run', 0),
                   'tier_counts': result.get('tier_counts', {})}
    except Exception as error:
        import traceback
        traceback.print_exc()  # lands in the supervisor's sanitized stderr tail
        receipt = {'status': 'failed', 'error_type': type(error).__name__}
    receipt.update(epoch=expected['epoch'], phase=job['phase'], wall_seconds=time.monotonic() - started)
    pr100_filter.atomic(directory / 'worker-receipt.json', receipt)
    return 0 if receipt['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
