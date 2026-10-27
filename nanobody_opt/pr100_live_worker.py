"""Measured baseline phases for the Pi-owned PR100 epoch controller."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from nanobody_opt import rules, pr100_filter, pr100_live_selection as selection

DEFAULT_WORKERS = 48
DEFAULT_MAX_WORKERS = 64
DEFAULT_CANDIDATE_BUDGET = 300.0
DEFAULT_PROPOSALS = 600
DEFAULT_REFERENCE_SLOTS = 16
WORKER_RSS_BYTES = 1.4 * 1024 ** 3
SCREEN_BACKENDS = frozenset({'resident_cpu', 'cuda_refine'})


def screen_settings(job):
    supplied = job.get('native_screen', {})
    if not isinstance(supplied, dict) or not isinstance(supplied.get('enabled', False), bool):
        raise ValueError('native_screen must contain a boolean enabled flag')
    settings = {'enabled': False, 'backend': 'resident_cpu', 'negative_audit_fraction': .1,
                'psh_rescue_margin': 20.0, 'batch_budget_seconds': 300.0, **supplied}
    if not isinstance(settings['backend'], str) or settings['backend'] not in SCREEN_BACKENDS:
        raise ValueError('unsupported native screen backend')
    for key in ('negative_audit_fraction', 'psh_rescue_margin', 'batch_budget_seconds'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('native screen bounds must be finite numbers')
    if (not 0 < settings['negative_audit_fraction'] <= 1 or settings['psh_rescue_margin'] < 0
            or not 0 < settings['batch_budget_seconds'] <= 1800):
        raise ValueError('invalid native screen bounds')
    return settings


def screen_identity(profile, sequence, challenge):
    """Only this epoch's explicitly non-authoritative, identified screens qualify."""
    if (not isinstance(profile, dict) or profile.get('authoritative') is not False
            or not isinstance(profile.get('evidence_backend'), str)
            or profile.get('evidence_backend') not in SCREEN_BACKENDS
            or profile.get('sequence_hash') != rules.sequence_hash(sequence)
            or type(profile.get('epoch')) is not int
            or profile.get('epoch') != challenge['epoch']
            or profile.get('challenge_sha256') != rules.digest(challenge)):
        return None
    if any(not isinstance(profile.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', profile[key])
           for key in ('protocol_sha256', 'runtime_identity_sha256')):
        return None
    return {key: profile[key] for key in ('evidence_backend', 'authoritative', 'sequence_hash',
            'epoch', 'challenge_sha256', 'protocol_sha256', 'runtime_identity_sha256')}


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


def native_phase(job, directory, *, clock=time.monotonic, measure=None, novelty=None, screen=None):
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
    screening = screen_settings(job)
    confirmation_count = max(repeats, margin['min_repeats'])
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
                  fresh_evaluations=0, repeat_evaluations=0,
                  evaluation_counter_scope='authoritative_cpu_calls_only',
                  total_profile_evaluations=0, unique_profiled_candidates=0,
                  not_started=len(rows), not_completed=len(rows), in_flight=0,
                  candidates=[], rejection_counts={}, cache_hits=0, workers=workers,
                  candidate_budget_seconds=budget, repeats=repeats, allow_marginal=allow_marginal,
                  margin={**margin, 'threshold_at_k': selection.margin_threshold(repeats, margin),
                          'lower_threshold_at_k': selection.margin_lower_threshold(repeats, margin)},
                  novelty_checked=False, native_classification_complete=False,
                  observation_errors=0,
                  native_screen={'enabled': screening['enabled'], 'requested_backend': screening['backend'],
                    'attempted_batches': 0, 'completed_profiles': 0, 'actual_backends': {},
                    'cpu_fallback_candidates': 0, 'screen_negative_candidates': 0,
                    'audit_candidates': 0, 'audit_disagreements': 0, 'audit_inconclusive': 0,
                    'rescued_candidates': 0, 'rescue_not_run': 0, 'errors': 0,
                    'rejection_disabled': False, 'cpu_confirmation_minimum': confirmation_count,
                    'negative_audit_fraction': screening['negative_audit_fraction'],
                    'psh_rescue_margin': screening['psh_rescue_margin'],
                    'recall_guarantee': False, 'authoritative': False},
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
    screened = set()
    screen_evidence = {}
    audited = set()
    screen_skipped = set()
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
            counter = 'fresh_evaluations' if expected[key] == 1 else 'repeat_evaluations'
            result[counter] += 1
            result['total_profile_evaluations'] = result['completed'] + result['native_screen']['completed_profiles']
            result['unique_profiled_candidates'] = len(set(measurements) | screened)
            result['not_completed'] = len(rows) - len(measurements)
            result['in_flight'] -= 1
            try:
                # An unexpected backend must not lose its provenance and enter
                # the authoritative CPU journal as a successful measurement.
                journal_value = (value if selection.authoritative_cpu_measurement(value) else
                                 {'error': True, 'error_kind': 'profile_exception'})
                observed = journal_module.observation(journal_value, requirements, key)
                result['measurement_outcomes'][observed['outcome']] += 1
                progress = {k: result[k] for k in ('status', 'input_count', 'completed',
                    'fresh_evaluations', 'repeat_evaluations',
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

    def screen_batch_rows(batch, index):
        state = result['native_screen']
        if not screening['enabled'] or state['rejection_disabled']:
            return batch, []
        # Reserve each full CPU timeout, including every required confirmation.
        available = (math.inf if deadline is None else deadline - clock() - budget * confirmation_count)
        timeout = min(float(screening['batch_budget_seconds']), available)
        if timeout < 10:
            state['cpu_fallback_candidates'] += len(batch)
            return batch, []
        state['attempted_batches'] += 1
        batch_dir = directory / 'native-screen' / f'batch-{index:05d}'
        try:
            backend = screen
            if backend is None:
                from nanobody_opt.pr100_native_screen import screen_batch
                backend = screen_batch
            payload = backend([r['sequence'] for r in batch], job={**job, 'native_screen': screening},
                              directory=batch_dir, timeout_seconds=timeout)
            profiles = payload.get('profiles', {}) if isinstance(payload, dict) else {}
            if not isinstance(profiles, dict) or set(profiles) - {r['sequence'] for r in batch}:
                raise ValueError('invalid native screen batch')
        except Exception:
            # Availability failures retain the CPU path. Cancellation propagates.
            state['errors'] += 1
            state['cpu_fallback_candidates'] += len(batch)
            return batch, []
        ready, negatives, records = [], [], []
        for row in batch:
            key = row['sequence']; value = profiles.get(key)
            identity = screen_identity(value, key, job['challenge'])
            if identity is None:
                state['cpu_fallback_candidates'] += 1
                ready.append(row); continue
            observed = journal_module.observation(value, requirements, key)
            record_value = {**identity, 'observation': observed}
            records.append(record_value)
            complete = (not value.get('error') and all(
                isinstance(value.get(k), (int, float)) and not isinstance(value.get(k), bool)
                and math.isfinite(value[k]) for k in journal_module.METRICS))
            if not complete:
                state['cpu_fallback_candidates'] += 1
                ready.append(row); record_value['screen_decision'] = 'inconclusive_cpu_fallback'; continue
            screened.add(key); screen_evidence[key] = identity
            state['completed_profiles'] += 1
            name = identity['evidence_backend']
            state['actual_backends'][name] = state['actual_backends'].get(name, 0) + 1
            psh = value[selection.PSH_KEY]
            width = screening['psh_rescue_margin']
            far_psh = (psh < selection.PSH_LOWER_BOUND - width or
                       psh > float(requirements['psh_strict_rejection_threshold']) + width)
            if observed['gate_reason'] in ('psh_threshold', 'developability_red_flags') and far_psh:
                negatives.append(row); record_value['screen_decision'] = 'far_psh_negative'
            else:
                # Near-boundary, non-PSH and inconclusive cases all get CPU
                # measurements. This heuristic makes no recall guarantee.
                ready.append(row); record_value['screen_decision'] = 'cpu_confirmation'
        ordered = sorted(negatives, key=lambda r: rules.digest(
            [job['challenge'], r['sequence'], 'negative-audit-v1']))
        count = math.ceil(len(ordered) * screening['negative_audit_fraction'])
        audit = ordered[:count]
        audited.update(r['sequence'] for r in audit)
        audit_hashes = {rules.sequence_hash(r['sequence']) for r in audit}
        for entry in records:
            entry['selected_for_cpu_audit'] = entry['sequence_hash'] in audit_hashes
        state['audit_candidates'] += len(audit)
        ready.extend(audit)
        result['total_profile_evaluations'] = result['completed'] + state['completed_profiles']
        result['unique_profiled_candidates'] = len(set(measurements) | screened)
        # Allowlisted screen records never enter the authoritative CPU journal.
        try:
            batch_dir.mkdir(parents=True, exist_ok=True)
            pr100_filter.atomic(batch_dir / 'screen-observations.json', {
                'schema': 'pr100-nonauthoritative-screen-v1', 'authoritative': False,
                'epoch': job['challenge']['epoch'], 'challenge_sha256': rules.digest(job['challenge']),
                'records': records})
        except Exception:
            observation_error('ScreenObservationWriteError')
        flush()
        return ready, ordered[count:]

    def collect_measurements():
        # Confirm each batch's survivors before spending the remaining window on
        # another fresh batch. A global first pass exhausted the admission window
        # before any repeats, leaving repeats=3 effectively at one observation.
        # Every admission still reserves the original full candidate timeout.
        measured_rows = []
        for offset in range(0, len(rows), workers):
            original_batch = rows[offset:offset + workers]
            if out_of_time():
                break
            batch, held = screen_batch_rows(original_batch, offset // workers)
            if not batch or out_of_time():
                break
            _acquired, elapsed = measure_and_record(batch)
            measured_rows.extend(batch)
            result['last_batch_seconds'] = elapsed
            flush()
            disagreements = [row for row in batch if row['sequence'] in audited and selection.decide(
                measurements[row['sequence']], requirements, sequence=row['sequence'],
                repeats=repeats, margin=margin)['tier'] is not None]
            inconclusive = [row for row in batch if row['sequence'] in audited and (
                not selection.authoritative_cpu_measurement(measurements[row['sequence']][0])
                or journal_module.observation(measurements[row['sequence']][0], requirements,
                                              row['sequence'])['outcome'] == 'inconclusive')]
            if disagreements or inconclusive:
                # A CPU passer among audited negatives invalidates the shortcut.
                # Missing/failed CPU evidence cannot establish agreement either.
                # Rescue held negatives and stop filtering later batches.
                state = result['native_screen']
                state['audit_disagreements'] += len(disagreements)
                state['audit_inconclusive'] += len(inconclusive)
                state['rejection_disabled'] = True
                if held and not out_of_time():
                    _acquired, elapsed = measure_and_record(held)
                    state['rescued_candidates'] += len(held)
                    measured_rows.extend(held); batch.extend(held)
                    result['last_batch_seconds'] = elapsed
                    flush()
                elif held:
                    state['rescue_not_run'] += len(held)
            else:
                screen_skipped.update(r['sequence'] for r in held)
                result['native_screen']['screen_negative_candidates'] += len(held)
            pending = batch
            needed = lambda row: confirmation_count if row['sequence'] in screen_evidence else repeats
            for _ in range(max((needed(r) - 1 for r in batch), default=0)):
                # Recheck the full native gate after every observation. Once a
                # candidate fails, a later lucky draw cannot restore eligibility.
                pending = [row for row in pending if len(measurements[row['sequence']]) < needed(row) and selection.decide(
                    measurements[row['sequence']], requirements, sequence=row['sequence'],
                    repeats=repeats, margin=margin)['tier'] is not None]
                if not pending or out_of_time():
                    break
                _acquired, elapsed = measure_and_record(pending)
                result['last_batch_seconds'] = elapsed
                flush()
        # Include measured failures so the final classification counts each
        # candidate exactly once, including failures on confirmation passes.
        return measured_rows

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
                                   repeats=repeats, margin=margin,
                                   cpu_confirmation_minimum=confirmation_count if key in screen_evidence else 0)
        verdicts[key] = verdict
        if verdict['tier'] is None:
            record(verdict['reason'] or 'rejected')
            continue
        record('passed' if verdict['tier'] == 'A' else verdict['reason'])
        evidence = {'challenge': job['challenge'], 'filter_sha': rules.FILTER_SHA,
                    'annotation': row['pr100_annotation'],
                    'developability': verdict['measurements'][0]}
        if key in screen_evidence:
            # Carry the confirmation requirement across export and Pi validation;
            # a forged tier label cannot discard it at a publication boundary.
            evidence['native_screen_confirmation'] = {
                'screen': screen_evidence[key], 'minimum': confirmation_count,
                'measurements': verdict['measurements']}
        rules.validate_evidence(evidence, key, job['challenge'], config)
        passers.append({**row, 'layer2_passed': True, 'pr100_evidence': evidence,
                        'accelerated_screen': screen_evidence.get(key),
                        'cpu_confirmation_minimum': verdict['cpu_confirmation_minimum'],
                        'tier': verdict['tier'], 'margin_safe': verdict['margin_safe'],
                        'measurements': verdict['measurements'],
                        'psh_values': verdict['psh_values'], 'psh_mean': verdict['psh_mean'],
                        'psh_sd': verdict['psh_sd'], 'repeats': verdict['repeats'],
                         'margin_threshold': verdict['margin_threshold'],
                         'margin_lower_threshold': verdict['margin_lower_threshold'],
                         'psh_boundary_margin': verdict['psh_boundary_margin'],
                         'selection_reason': verdict['reason']})
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
    result['cpu_not_run'] = len(rows) - len(started)
    result['not_run'] = len(rows) - len(started | screened)
    if screen_skipped:
        result['rejection_counts']['nonauthoritative_screen_negative'] = len(screen_skipped)
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
    environment = {k: os.environ[k] for k in SCORE_ENV_KEYS if k in os.environ and k in ENV_KEYS}
    jobs = []
    for gpu in job['gpus']:
        rows = shards[gpu][:SCORE_SHARD_LIMIT]
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
        result = {'sequence': sequence_phase, 'native': native_phase,
                  'score': score_phase}[job['phase']](job, directory)
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
