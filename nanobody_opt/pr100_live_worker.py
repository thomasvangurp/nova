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


def measure_batch(sequences, workers, *, clock=time.monotonic, develop=None):
    """Profile a batch and report wall measured from slot acquisition.

    The upstream ProfileExecution timeout starts before the semaphore is
    acquired, so a queued candidate can burn its whole budget waiting. We never
    enqueue more profiles than can start at once, and the wall we attribute to a
    candidate begins here, when its slot is already held.
    """
    develop = develop or pr100_filter.develop
    acquired = clock()
    values = asyncio.run(develop(list(sequences), workers))
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


def native_phase(job, directory, *, clock=time.monotonic, measure=measure_batch, novelty=None):
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
    result = {k: incoming[k] for k in ('challenge', 'target', 'epoch', 'archive_sha256',
                                       'archive_checked_at_unix')}
    result.update(status='running', input_count=len(rows), completed=0, not_run=len(rows),
                  candidates=[], rejection_counts={}, cache_hits=0, workers=workers,
                  candidate_budget_seconds=budget, repeats=repeats, allow_marginal=allow_marginal,
                  margin={**margin, 'threshold_at_k': selection.margin_threshold(repeats, margin)},
                  novelty_checked=False)

    def record(reason):
        result['rejection_counts'][reason] = result['rejection_counts'].get(reason, 0) + 1

    def flush():
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
    # Pass 1: one measurement each. Never enqueue more profiles than can start,
    # so a candidate's budget is never consumed waiting for a cpu slot.
    for offset in range(0, len(rows), workers):
        batch = rows[offset:offset + workers]
        if out_of_time():
            break
        values, acquired, elapsed = measure([r['sequence'] for r in batch], workers, clock=clock)
        for row, value in zip(batch, values, strict=True):
            key = row['sequence']
            started.add(key)
            measurements[key] = [value]
            result['completed'] += 1
        result['not_run'] = len(rows) - len(started)
        result['last_batch_seconds'] = elapsed
        flush()

    survivors, verdicts = [], {}
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
            values, _acquired, elapsed = measure([r['sequence'] for r in batch], workers, clock=clock)
            for row, value in zip(batch, values, strict=True):
                measurements[row['sequence']].append(value)
                result['completed'] += 1
            result['last_batch_seconds'] = elapsed
            flush()

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
        result = {'sequence': sequence_phase, 'native': native_phase}[job['phase']](job, directory)
        receipt = {'status': result['status'], 'candidate_count': len(result['candidates']),
                   'not_run': result.get('not_run', 0),
                   'tier_counts': result.get('tier_counts', {})}
    except Exception as error:
        receipt = {'status': 'failed', 'error_type': type(error).__name__}
    receipt.update(epoch=expected['epoch'], phase=job['phase'], wall_seconds=time.monotonic() - started)
    pr100_filter.atomic(directory / 'worker-receipt.json', receipt)
    return 0 if receipt['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
