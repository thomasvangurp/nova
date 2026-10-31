"""Measured baseline phases for the Pi-owned PR100 epoch controller."""
from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from nanobody_opt import rules, pr100_filter, pr100_live_selection as selection
from nanobody_opt import pr100_fresh_budget as fresh_budget

DEFAULT_WORKERS = 48
DEFAULT_MAX_WORKERS = 64
DEFAULT_CANDIDATE_BUDGET = 300.0
DEFAULT_PROPOSALS = 600
DEFAULT_REFERENCE_SLOTS = 16
DEFAULT_EDIT_LIMIT = 300
WORKER_RSS_BYTES = 1.4 * 1024 ** 3
SCREEN_BACKENDS = frozenset({'resident_cpu', 'cuda_refine'})
# Only execution settings that can change the numerical backend are bound.
# Release paths, worker counts and other scheduling settings stay portable.
NATIVE_EXECUTION_ENV_KEYS = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENMM_CPU_THREADS',
    'NUMEXPR_NUM_THREADS', 'OMP_DYNAMIC', 'MKL_DYNAMIC',
    'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
    'OPENMM_PLUGIN_DIR', 'OPENMM_DEFAULT_PLATFORM', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
)


def interleave_segment_controls(eligible, *, segment_source, edit_source='edit_generation'):
    """Order eligible rows for native admission: four segment rows per one control.

    The 80% share is an opportunity target while both sources have eligible
    work; differing eligibility or source exhaustion can change the final share.
    Edit and baseline controls alternate while both arms still hold rows, and
    each arm keeps its internal order. An exhausted arm releases its slots to
    the remaining arms (work-conserving), so native capacity is never left idle
    while another arm still has eligible work. This reorders only the admission
    order of rows that already passed eligibility filters; it changes no gate,
    budget or biological rule.
    """
    from collections import deque
    segment_pool = deque(row for row in eligible if row.get('source') == segment_source)
    edit_pool = deque(row for row in eligible if row.get('source') == edit_source)
    baseline_pool = deque(row for row in eligible
                          if row.get('source') not in (segment_source, edit_source))
    edit_next = True

    def next_control():
        nonlocal edit_next
        if edit_next and edit_pool:
            edit_next = False
            return edit_pool.popleft()
        if not edit_next and baseline_pool:
            edit_next = True
            return baseline_pool.popleft()
        if edit_pool:
            edit_next = False
            return edit_pool.popleft()
        if baseline_pool:
            edit_next = True
            return baseline_pool.popleft()
        return None

    ordered = []
    while segment_pool or edit_pool or baseline_pool:
        for _ in range(4):
            if not segment_pool:
                break
            ordered.append(segment_pool.popleft())
        control = next_control() if (edit_pool or baseline_pool) else None
        if control is not None:
            ordered.append(control)
    return ordered


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


def native_slot_resources(allocated, settings, adapter, *, cores):
    """Resident queue slots never expand the actual CPU/model allocation."""
    threads = settings['registration']['policy']['preparation_threads']
    physical_limit = min(allocated, max(1, cores // threads))
    if settings.get('resident') is True:
        scientific = len(settings['gpus']) * settings['workers_per_gpu']
        if scientific > physical_limit:
            raise ValueError('resident model processes exceed native CPU allocation')
        return {'queue_slots':adapter.capacity, 'scientific_processes':scientific,
                'executor_workers':scientific}
    count = min(physical_limit, adapter.capacity)
    return {'queue_slots':count,'scientific_processes':count,'executor_workers':count}


def annotation_worker_count(job, *, segment_enabled, cores=None):
    """Small annotation pool; retain the baseline's serial execution path.

    The unchanged AbNumber ANARCI call leaves HMMER's thread count implicit.
    Reserve three CPU slots per call (two HMMER workers plus its master), or
    more if HMMER_NCPU is already configured. Never modify that environment
    or the scientific call's parameters to obtain concurrency.
    """
    if not segment_enabled:
        return 1
    bounds = [job.get('workers', DEFAULT_WORKERS), job.get('max_workers', DEFAULT_MAX_WORKERS),
              job.get('cpu_slots', 1)]
    if any(type(n) is not int or n < 1 for n in bounds):
        raise ValueError('annotation resources must be positive integers')
    try:
        hmmer_workers = int(os.environ.get('HMMER_NCPU', '2'))
    except ValueError:
        raise ValueError('invalid HMMER worker count') from None
    if hmmer_workers < 0:
        raise ValueError('invalid HMMER worker count')
    slots_per_call = max(3, hmmer_workers + 1)
    cores = available_cores() if cores is None else cores
    return max(1, min(8, bounds[0], bounds[1], bounds[2] // slots_per_call,
                      max(1, cores) // slots_per_call))


def _warm_sequence_annotation():
    """Initialize imported modules and read-only reference caches before threads."""
    import importlib
    importlib.import_module('abnumber')
    importlib.import_module('anarci.anarci')
    pr100_filter.validate_annotation_runtime()
    germline = pr100_filter.helper('germline_identity')
    germline._anarci_panel('human')
    germline._hmm_digest()
    return pr100_filter.annotation_identity()


def _sequence_annotation(sequence, requirements, expected, identity=None):
    from nanobody_opt import gates, hydrophobicity
    # The exact existing sequence, annotation and hydrophobicity gates retain
    # their order. Execution errors propagate; they are never biological rejects.
    if gates.check_sequence(sequence, requirements):
        return None, 'basic_sequence'
    annotation = {**pr100_filter._annotate_uncached(sequence),
                  'annotation_identity': (pr100_filter.annotation_identity()
                                          if identity is None else identity)}
    reason = rules.annotation_reason(annotation, requirements,
                                     sequence=sequence, epoch_challenge=expected)
    if not reason and not hydrophobicity.check(sequence)[1]:
        reason = hydrophobicity.REJECT_REASON
    return annotation, reason


def sequence_annotations(sequences, requirements, expected, *, workers=1):
    """Yield exact gate results in input order with at most workers tasks held.

    No process pool, disk annotation cache, HMM batching or extraction changes.
    A failed input raises in input order, as on the serial path; cancel work
    that has not started and join running tasks before propagating the failure.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('annotation workers must be within one to eight')
    if workers == 1:
        for sequence in sequences:
            yield _sequence_annotation(sequence, requirements, expected)
        return
    from collections import deque
    identity = _warm_sequence_annotation()
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='pr100-annotation') as pool:
        try:
            for sequence in sequences:
                pending.append(pool.submit(_sequence_annotation, sequence, requirements, expected, identity))
                if len(pending) == workers:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


def candidate_budget(job):
    budget = job.get('candidate_budget_seconds', DEFAULT_CANDIDATE_BUDGET)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
        raise ValueError('candidate_budget_seconds must be a positive number')
    return float(budget)


def edit_generation_settings(job):
    """Opt-in v2 edit-based generation lane; OFF unless the operator enables it.

    The worker runs the de-novo scaffold lane by default. ``job['edit_generation']``
    is an optional object -- ``{"enabled": true, "limit": 300}`` turns on the
    edit-based lane (``pr100_edit_generation``, ported from
    ``nanobody_opt/local_neighborhood``) fed by rival parents from the reference
    bank. ``limit`` caps the number of emitted edit candidates and defaults to
    ``DEFAULT_EDIT_LIMIT``; an enabled lane is never unlimited, so it must be >= 1.
    Optional ``policy`` selects ``balanced_v2`` (default) or the separately
    qualified ``chemistry_conservative_v3``. Policy is retained in settings
    identity and emitted lineage; it does not alter any downstream gate.
    """
    supplied = job.get('edit_generation', {})
    if not isinstance(supplied, dict) or not isinstance(supplied.get('enabled', False), bool):
        raise ValueError('edit_generation must contain a boolean enabled flag')
    if set(supplied) - {'enabled', 'limit', 'policy'}:
        raise ValueError('unknown edit_generation setting')
    from nanobody_opt.pr100_edit_generation import BALANCED_POLICY, validate_policy
    validate_policy(supplied.get('policy', BALANCED_POLICY))
    # Preserve legacy normalized settings/digest when policy is omitted. An
    # explicit policy is retained and therefore needs its own qualification.
    settings = {'enabled': False, 'limit': DEFAULT_EDIT_LIMIT, **supplied}
    limit = settings['limit']
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError('edit_generation.limit must be a nonnegative integer')
    if settings['enabled'] and limit < 1:
        raise ValueError('edit_generation.limit must be >= 1 when edit_generation is enabled')
    return settings


def measure_batch(sequences, workers, *, clock=time.monotonic, develop=None,
                  on_result=None, on_result_error=None, timeout_seconds=None, executor_workers=None):
    """Profile a batch and report wall measured from slot acquisition.

    The upstream ProfileExecution timeout starts before the semaphore is
    acquired, so a queued candidate can burn its whole budget waiting. We never
    enqueue more profiles than can start at once, and the wall we attribute to a
    candidate begins here, when its slot is already held.
    """
    develop = develop or pr100_filter.develop
    executor_workers = workers if executor_workers is None else executor_workers
    if type(executor_workers) is not int or not 1 <= executor_workers <= workers:
        raise ValueError('invalid native executor allocation')
    acquired = clock()
    callbacks = ({'on_result': on_result, 'on_result_error': on_result_error}
                 if on_result is not None else {})
    if timeout_seconds is not None:
        callbacks['task_timeout_seconds'] = timeout_seconds
    def private_loop():
        loop = asyncio.new_event_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=executor_workers,
                                                    thread_name_prefix='pr100-native'))
        return loop
    # Only this batch owns this loop. Avoid the implicit32-thread ceiling.
    with asyncio.Runner(loop_factory=private_loop) as runner:
        values = runner.run(develop(list(sequences), workers, **callbacks))
    return values, acquired, clock() - acquired


def sequence_novelty_prefilter(job, directory, rows, *, check=None, clock=time.monotonic):
    """Optional exact archive search before any native admission; no native claim."""
    enabled = job.get('sequence_full_archive_novelty', False)
    if type(enabled) is not bool:
        raise ValueError('sequence_full_archive_novelty must be a boolean')
    if not enabled:
        return rows, None
    from nanobody_opt.pr100_novelty_runtime import validate_config
    runtime = validate_config(job.get('novelty_runtime'))
    archive = directory / 'archive.csv'
    if file_digest(archive) != job['archive_sha256']:
        raise ValueError('early novelty archive snapshot changed')
    started = clock()
    kept, archive_rows = ((check or check_novelty)(rows, archive, runtime=runtime)
                          if rows else ([], None))
    kept_hashes = {rules.sequence_hash(row['sequence']) for row in kept}
    removed = [rules.sequence_hash(row['sequence']) for row in rows
               if rules.sequence_hash(row['sequence']) not in kept_hashes]
    return kept, {'runtime_sha256': runtime['identity_sha256'],
                  'archive_sha256': job['archive_sha256'], 'archive_rows': archive_rows,
                  'input_sha256': rules.digest(rows), 'input_count': len(rows),
                  'kept_count': len(kept), 'removed_count': len(removed),
                  'removed_sequence_hashes': removed, 'elapsed_seconds': clock() - started}


def sequence_phase(job, directory):
    from abnumber import Chain
    import pandas as pd
    from nanobody_opt import gates, hydrophobicity, pr100_proposal_pilot as proposals
    from nanobody_opt import check_validator_novelty as novelty
    from nanobody_opt import pr100_reference_donors as donors
    from nanobody_opt import pr100_edit_generation
    from nanobody_opt import pr100_v_identity_budget
    from nanobody_opt import pr100_segment_generation as segments
    from collections import Counter
    expected = job['challenge']
    config = rules.pinned_config(expected['epoch'])
    pr100_filter.validate_annotation_runtime()
    archive = directory / 'archive.csv'
    if file_digest(archive) != job['archive_sha256']:
        raise ValueError('archive snapshot changed')
    frame = pd.read_csv(archive)
    fallback = donors.enabled(job)
    bank_path = Path(job['references'])
    bank_bytes = bank_path.read_bytes()
    bank_digest = hashlib.sha256(bank_bytes).hexdigest()
    expected_bank = job.get('reference_bank_sha256')
    if expected_bank is not None and (not isinstance(expected_bank, str)
            or not re.fullmatch(r'[0-9a-f]{64}', expected_bank) or expected_bank != bank_digest):
        raise ValueError('reference bank identity mismatch')
    bank_value = json.loads(bank_bytes)
    frozen = job.get('parent_bank_snapshot')
    if frozen and frozen.get('status') in ('fresh', 'retained_qualified'):
        from nanobody_opt.pr100_parent_bank import require
        require(bank_value.get('source_sha') == frozen.get('source_sha')
                and bank_value.get('runtime_sha256') == frozen.get('runtime_sha256')
                and bank_value.get('target') == expected['target']
                and type(bank_value.get('before_epoch')) is int
                and bank_value['before_epoch'] <= expected['epoch'], 'frozen_parent_bank_identity')
    bank = bank_value['rows']
    slots = job.get('reference_slots', DEFAULT_REFERENCE_SLOTS)
    proposals_wanted = job.get('proposals', DEFAULT_PROPOSALS)
    proposal_offset = job.get('proposal_offset', 0)
    segment_settings = segments.settings(job)
    if type(proposal_offset) is not int or proposal_offset < 0:
        raise ValueError('proposal_offset must be a nonnegative integer')
    excluded = job.get('excluded_sequence_hashes', [])
    if (not isinstance(excluded, list) or any(not isinstance(v, str) or
            not re.fullmatch(r'[0-9a-f]{64}', v) for v in excluded)):
        raise ValueError('invalid excluded sequence hashes')
    excluded = set(excluded)
    for name, value in (('reference_slots', slots), ('proposals', proposals_wanted)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + ' must be a positive integer')
    refs, proposal_donors, reference_coverage = donors.select(bank, expected, slots,
                                                           fallback=fallback)
    reference_coverage['bank_sha256'] = bank_digest
    if frozen is not None:
        reference_coverage['parent_bank_snapshot'] = frozen
    panels, _ = pr100_filter.helper('germline_identity')._anarci_panel('human')
    archive_hashes = {rules.sequence_hash(str(value).strip().upper())
                      for value in frame['sequence'].dropna() if str(value).strip()}
    # Exact archive duplicates cannot pass validator novelty. Avoid annotation
    # and native work for them, while retaining the full later novelty check.
    parent_hashes = ({r['sequence_hash'] for r in bank}
                     if reference_coverage['fallback_used'] else set())
    blocked = excluded | archive_hashes | parent_hashes
    rows = [{**r, 'source': 'existing_native_reference'} for r in refs
            if r['sequence_hash'] not in blocked]
    seen = {r['sequence_hash'] for r in rows}
    historical_collisions = {r['sequence_hash'] for r in refs} & archive_hashes
    parent_collisions = set()
    source_counts, duplicate_counts = Counter(), Counter()
    proposal_attempts = 0
    loop_cache = {}
    fresh_count = 0

    def emit(quota, use_donors):
        nonlocal proposal_attempts, fresh_count
        emitted = 0
        # The cap bounds cheap proposal work only. Native admission remains
        # independently bounded by the shared fresh-evaluation ledger.
        for _ in range(quota * 10):
            if emitted >= quota:
                break
            ordinal = proposal_offset + proposal_attempts
            proposal_attempts += 1
            ref, provenance = (proposal_donors[ordinal % len(proposal_donors)]
                if use_donors else (None, {'reference_stratum': 'unseeded_control'}))
            cdr3 = None
            if ref:
                if ref['sequence_hash'] not in loop_cache:
                    loop_cache[ref['sequence_hash']] = Chain(ref['sequence'], scheme='imgt',
                        cdr_definition='imgt', use_anarcii=False).cdr3_seq
                cdr3 = loop_cache[ref['sequence_hash']]
            row = proposals.propose(panels, expected['allowed_v'], expected['allowed_cdr3_length'],
                                    seed=expected['epoch'], ordinal=ordinal, reference_cdr3=cdr3)
            digest = rules.sequence_hash(row['sequence'])
            if digest in seen or digest in blocked:
                duplicate_counts[provenance['reference_stratum']] += 1
                if digest in archive_hashes:
                    historical_collisions.add(digest)
                if digest in parent_hashes:
                    parent_collisions.add(digest)
                continue
            seen.add(digest)
            rows.append({**row, 'source': 'fresh_baseline_proposal',
                         'reference_hash': ref['sequence_hash'] if ref else None, **provenance})
            source_counts[provenance['reference_stratum']] += 1
            emitted += 1
            fresh_count += 1

    if proposal_donors:
        emit(proposals_wanted // 2, True)
    # An exhausted donor arm releases its unused proposal slots to cold start.
    emit(proposals_wanted - fresh_count, False)
    edit_settings = edit_generation_settings(job)
    edit_source_counts = Counter()
    edit_stats = {}
    if edit_settings['enabled']:
        # The edit lane mutates parents IN PLACE, preserving their CDR3 length
        # and V gene. Only parents already matching this epoch's challenge cell
        # can seed children that pass the wrong_epoch_* annotation gates, so the
        # lane is fed by the bank rows that carry the exact cell. The public
        # archive supplies the same-length background residue variation (and the
        # haplotype donors), exactly as local_neighborhood uses the recent field.
        edit_cell = f"{expected['allowed_v']}:{expected['allowed_cdr3_length']}"
        # Graft every exact-cell parent. Leaderboard parents already passed the
        # validator's developability gate, and developability_rank is a relative
        # rank within the accepted field, not a PSH pass/fail — so a hard
        # out_of_range drop discarded the winning parents. developability_rank
        # stays a sort tiebreak in select(); children are re-checked at the
        # native PSH gate.
        edit_parents = [r['sequence'] for r in refs
                        if edit_cell in r.get('cells', [])]
        edit_background = [str(s).strip().upper() for s in frame['sequence']
                           if isinstance(s, str) and s.strip()]
        parent_cdrs = {}
        parent_identity_budgets = {}
        for parent in edit_parents:
            try:
                if set(parent) - gates.ALLOWED_AAS:
                    raise ValueError('invalid parent alphabet')
                chain = Chain(parent, scheme='imgt', cdr_definition='imgt', use_anarcii=False)
                parent_identity_budgets[parent] = pr100_v_identity_budget.build(
                    parent, allowed_gene=expected['allowed_v'],
                    minimum_identity_pct=config['nanobody_requirements']['min_human_v_identity'])
                parent_cdrs[parent] = {key: str(getattr(chain, key + '_seq'))
                                       for key in ('cdr1', 'cdr2', 'cdr3')}
            except Exception:
                # A malformed bank row must not abort all useful parent work.
                edit_stats['parent_annotation_failed'] = edit_stats.get('parent_annotation_failed', 0) + 1
        edit_candidates = pr100_edit_generation.generate(
            edit_parents, background_sequences=edit_background,
            limit=edit_settings['limit'], excluded_hashes=excluded | archive_hashes | seen,
            stats=edit_stats, novelty_aware=True, parent_cdrs=parent_cdrs,
            parent_identity_budgets=parent_identity_budgets,
            policy=edit_settings.get('policy', pr100_edit_generation.BALANCED_POLICY))
        for edit in edit_candidates:
            rows.append({**edit, 'source': 'edit_generation',
                         'requested_v': expected['allowed_v'],
                         'requested_cdr3_length': expected['allowed_cdr3_length']})
            edit_source_counts[edit['proposal_lane']] += 1
    segment_stats = {}
    segment_donors = []
    if segment_settings['enabled']:
        # Segment donors need not already occupy the child's challenge cell.
        # Only the independently annotated child can satisfy that constraint.
        for donor in segments.choose_donors(bank, expected, segment_settings['max_donors']):
            try:
                chain = Chain(donor['sequence'], scheme='imgt', cdr_definition='imgt', use_anarcii=False)
                numbered = [{'number': pos.number, 'letter': pos.letter, 'aa': aa}
                            for pos, aa in chain]
                segment_donors.append(segments.prepare(donor, numbered))
            except Exception:
                # Bad source annotation drops only that donor; the baseline
                # and other measured generation arms remain runnable.
                segment_stats['donor_numbering_failed'] = segment_stats.get('donor_numbering_failed', 0) + 1
        rows.extend(segments.generate(segment_donors, expected,
            limit=segment_settings['limit'], beam_width=segment_settings['beam_width'],
            max_per_cdr3=segment_settings['max_per_cdr3'],
            offset=proposal_offset // max(1, proposals_wanted),
            excluded=excluded | archive_hashes | {rules.sequence_hash(r['sequence']) for r in rows},
            stats=segment_stats))
    generated = sorted({rules.sequence_hash(r['sequence']) for r in rows})
    unique = {r['sequence']: r for r in rows
              if rules.sequence_hash(r['sequence']) not in blocked}
    # Preserve the source relationship before cheap eligibility filters discard
    # candidates. A background importer owns SQLite writes; mining never waits
    # on a database or reconstructs parentage from sequence similarity.
    from nanobody_opt.pr100_history import generation_records
    history = generation_records(list(unique.values()), bank)
    try:
        pr100_filter.atomic(directory / 'generation-history.json', {'challenge': expected, **history})
        history['history_capture_status'] = 'sidecar_saved'
    except OSError:
        # The final result still carries the records. Optional sidecar failure
        # is observable, without making the database pipeline a mining gate.
        history['history_capture_status'] = 'sidecar_unavailable'
    eligible, reasons = [], {}
    annotation_workers = annotation_worker_count(job, segment_enabled=segment_settings['enabled'])
    annotations = sequence_annotations(unique, config['nanobody_requirements'], expected,
                                       workers=annotation_workers)
    for (annotation, reason), (sequence, row) in zip(annotations, unique.items()):
        reasons[reason or 'passed'] = reasons.get(reason or 'passed', 0) + 1
        if not reason:
            eligible.append({**row, 'sequence_hash': rules.sequence_hash(sequence),
                             'pr100_annotation': annotation})
    eligible, early_novelty = sequence_novelty_prefilter(job, directory, eligible)
    chemistry_priority_count = 0
    if (edit_settings['enabled'] and
            edit_settings.get('policy') == pr100_edit_generation.CHEMISTRY_POLICY):
        def chemistry_row(row):
            return (row.get('source') == 'edit_generation' and
                    row.get('edit_policy') == pr100_edit_generation.CHEMISTRY_POLICY)
        chemistry_priority_count = sum(chemistry_row(row) for row in eligible)
        # The lane is already bounded by its emission limit. Reach these
        # qualified parent edits before the baseline can exhaust the native
        # window. Stable sorting preserves both lanes' internal order; default
        # and balanced policy retain their existing order exactly.
        eligible.sort(key=lambda row: not chemistry_row(row))
    if segment_settings['enabled']:
        # Give the bounded experimental arm most native opportunity: four
        # segment rows per one combined control, alternating edit/baseline and
        # work-conserving when an arm is exhausted (see interleave_segment_controls).
        eligible = interleave_segment_controls(eligible, segment_source=segments.SOURCE)
    # The opt-in prefilter saves native work. Publication still requires the
    # final archive check; this early result never claims native qualification.
    result = {'challenge': expected, 'target': expected['target'], 'epoch': expected['epoch'], **history,
              'archive_sha256': job['archive_sha256'], 'archive_checked_at_unix': job['archive_fetched_at'],
              'archive_rows': len(frame), 'existing_references': len(refs),
              'reference_coverage': reference_coverage,
              'proposal_source_counts': dict(source_counts),
              'edit_generation_enabled': edit_settings['enabled'],
              'edit_generation_policy': edit_settings.get('policy', pr100_edit_generation.BALANCED_POLICY),
              'edit_generation_priority_count': chemistry_priority_count,
              'edit_generation_source_counts': dict(edit_source_counts),
              'edit_generation_candidates': sum(edit_source_counts.values()),
              'edit_generation_stats': dict(edit_stats),
              'segment_generation_enabled': segment_settings['enabled'],
              'segment_generation_settings': segment_settings,
              'segment_generation_stats': segment_stats,
              'eligible_segment_generation': sum(r.get('source') == segments.SOURCE for r in eligible),
              'eligible_proposal_source_counts': dict(Counter(
                  r['reference_stratum'] for r in eligible if 'reference_stratum' in r)),
              'eligible_edit_generation_source_counts': dict(Counter(
                  r['proposal_lane'] for r in eligible if r.get('source') == 'edit_generation')),
              'donor_parent_reuse_blocked': len(parent_collisions),
              'historical_exact_reuse_blocked': len(historical_collisions),
              'proposal_attempts': proposal_attempts,
              'proposal_duplicate_counts': dict(duplicate_counts),
              'fresh_unique_proposals': fresh_count,
              'proposal_exhausted': fresh_count < proposals_wanted,
              'proposals': proposals_wanted, 'annotation_reasons': reasons,
              'annotation_workers': annotation_workers,
              'proposal_offset': proposal_offset, 'next_proposal_offset': proposal_offset + proposal_attempts,
              'generated_sequence_hashes': generated, 'excluded_generated': len(generated) - len(unique) + sum(duplicate_counts.values()),
              'candidates': eligible, 'novelty_checked': False,
              'annotated_total': len(eligible), 'status': 'complete'}
    if early_novelty is not None:
        result['early_archive_novelty'] = early_novelty
    pr100_filter.atomic(directory / 'result.json', result)
    return result


def check_novelty(rows, archive, *, novelty=None, runtime=None):
    """Exact archive novelty, independent of the native structural measurement."""
    if novelty is None:
        from nanobody_opt.pr100_novelty_runtime import check
        return check(rows, archive, runtime=runtime)
    # Explicit dependency injection is used by the scheduling unit tests.
    import pandas as pd
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


def novelty_phase(job, directory, *, novelty=None):
    """Recheck a fresh archive using completed native evidence; no profiling."""
    incoming = json.loads((directory / 'input.json').read_text())
    if incoming.get('challenge') != job['challenge']:
        raise ValueError('novelty input challenge mismatch')
    archive = directory / 'archive.csv'
    if file_digest(archive) != job['archive_sha256']:
        raise ValueError('novelty archive snapshot changed')
    config = rules.pinned_config(job['challenge']['epoch'])
    for row in incoming['candidates']:
        if row.get('layer2_passed') is not True:
            raise ValueError('missing completed native gate')
        rules.validate_evidence(row.get('pr100_evidence'), row['sequence'], job['challenge'], config)
    kept, archive_rows = check_novelty(incoming['candidates'], archive, novelty=novelty,
                                       runtime=job.get('novelty_runtime'))
    result = {**incoming, 'candidates': kept, 'novelty_checked': True, 'status': 'complete',
              'archive_rows': archive_rows, 'archive_sha256': job['archive_sha256'],
              'archive_checked_at_unix': job['archive_fetched_at'],
              'novelty_runtime_sha256': (job.get('novelty_runtime') or {}).get('identity_sha256')}
    pr100_filter.atomic(directory / 'result.json', result)
    return result


def native_resume_identity(job, directory, requirements):
    """Same-runtime checkpoint binding; deliberately no model-equivalence claim."""
    from nanobody_opt import pr100_native_reference
    registered = pr100_native_reference.settings_for(job)
    if registered is not None:
        row = registered['registration']
        return {'challenge_sha256': rules.digest(job['challenge']), 'resume_eligible': True,
                'requirements_sha256': rules.digest(requirements), 'filter_sha': rules.FILTER_SHA,
                'backend': pr100_native_reference.BACKEND,
                'registration_sha256': registered['registration_sha256'],
                'runtime_identity_sha256': row['runtime_sha256'],
                'policy_sha256': rules.digest(row['policy']),
                'runtime_identity_status': 'registered_native_reference'}
    import importlib.util
    from nanobody_opt.pr100_native_journal import _context
    context = _context(directory, job, requirements)
    scientific, weights = {}, {}
    try:
        immune = Path(importlib.util.find_spec('ImmuneBuilder').origin).parent
        tnp = Path(importlib.util.find_spec('theraprofnano').origin).parent
        for name, path in {
            'builder': immune / 'NanoBodyBuilder2.py', 'refine': immune / 'refine.py',
            'cdr': tnp / 'CDR_Profiler/CDR3_Conf_Assigner.py',
            'surface': tnp / 'Hydrophobicity_and_Charge_Profiler/Hydrophobicity_and_Charge_Assigner.py',
        }.items():
            scientific[name] = file_digest(path)
        for index in range(1, 5):
            weights[str(index)] = file_digest(immune / 'trained_model' / f'nanobody_model_{index}')
    except (OSError, ImportError, AttributeError, TypeError, ValueError):
        pass
    runtime = context['runtime']
    complete = (len(scientific) == 4 and len(weights) == 4
                and bool(runtime['python_binary_sha256'])
                and all(runtime['packages'][name]['version_sha256'] for name in ('torch', 'numpy', 'openmm'))
                and all(runtime['tools'][name]['sha256'] for name in ('aggrescan3d-rs', 'freesasa', 'TNP')))
    return {'challenge_sha256': rules.digest(job['challenge']), 'resume_eligible': bool(complete),
            # Raw environment values must never enter evidence or diagnostics.
            'execution_environment_sha256': rules.digest({key: os.environ.get(key)
                                                          for key in NATIVE_EXECUTION_ENV_KEYS}),
            'requirements_sha256': rules.digest(requirements), 'filter_sha': rules.FILTER_SHA,
            'runtime': runtime, 'sources': {k: v for k, v in context['source_sha256'].items()
                if k not in ('worker', 'journal', 'rules')},
            'scientific_sources': scientific, 'model_weights': weights,
            'model_weight_identity': 'complete' if len(weights) == 4 else 'incomplete',
            'runtime_identity_status': 'checkpoint_bound' if complete else 'incomplete'}


def terminal_rejection_reason(values, requirements, sequence, reason):
    """Distinguish a completed aggregation rejection from an unavailable tool.

    The publication gate checks the backend's passed flag before numeric caps.
    Its shared failure label alone cannot revoke earlier evidence. Classify only
    completed authoritative CPU profiles with the existing finite-metric and
    provenance checks; never load audit journal rows as measurements.
    """
    if reason != 'aggrescan_unavailable_or_failed':
        return reason
    from nanobody_opt.pr100_native_journal import observation
    for value in values:
        if not selection.authoritative_cpu_measurement(value):
            continue
        observed = observation(value, requirements, sequence)
        if observed['outcome'] == 'rejected' and observed['gate_reason'] == 'aggrescan_threshold':
            return 'aggrescan_threshold'
    return reason


def native_phase(job, directory, *, clock=time.monotonic, measure=None, novelty=None, screen=None,
                 refill=None):
    from nanobody_opt import pr100_native_journal as journal_module
    reference_phase_started = clock() if job.get('native_reference', {}).get('enabled') is True else None
    incoming = json.loads((directory / 'input.json').read_text())
    if incoming['challenge'] != job['challenge']:
        raise ValueError('native input challenge mismatch')
    if novelty is None and (directory / 'archive.csv').is_file():
        from nanobody_opt.pr100_novelty_runtime import validate_config
        validate_config(job.get('novelty_runtime'))
    tools = Path(job['native_tools'])
    pr100_filter.helper('aggrescan3d').tool_paths = lambda: (tools / 'aggrescan3d-rs', tools / 'freesasa')
    workers = worker_count(job)
    from nanobody_opt import pr100_native_reference
    reference_settings = pr100_native_reference.settings_for(job)
    native_adapter = None
    scientific_workers = workers
    if reference_settings is not None:
        if measure is not None or screen_settings(job)['enabled']:
            raise ValueError('registered reference cannot mix measurement overrides or screens')
        if reference_settings.get('resident') is True:
            from nanobody_opt.pr100_registered_resident import RegisteredResidentAdapter
            native_adapter = RegisteredResidentAdapter(job, directory)
        else:
            native_adapter = pr100_native_reference.Adapter(job, directory)
        preparation_threads = reference_settings['registration']['policy']['preparation_threads']
        resources = native_slot_resources(workers,reference_settings,native_adapter,cores=available_cores())
        workers, scientific_workers = resources['queue_slots'],resources['scientific_processes']
    fresh_limit = fresh_budget.validate_limit(job.get('fresh_admission_limit', fresh_budget.DEFAULT_LIMIT), allow_zero=True)
    budget = candidate_budget(job)
    timed_admission = 'native_minimum_call_window_seconds' in job
    minimum_window = job.get('native_minimum_call_window_seconds', budget)
    cleanup_reserve = job.get('native_cleanup_reserve_seconds', 10. if timed_admission else 0.)
    if (isinstance(minimum_window, bool) or not isinstance(minimum_window, (int, float))
            or not math.isfinite(minimum_window) or minimum_window <= 0 or minimum_window > budget
            or timed_admission and minimum_window < 10
            or isinstance(cleanup_reserve, bool) or not isinstance(cleanup_reserve, (int, float))
            or not math.isfinite(cleanup_reserve) or not 0 <= cleanup_reserve <= 60):
        raise ValueError('invalid native call admission bounds')
    repeats = job.get('repeats', selection.DEFAULT_REPEATS)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    margin = selection.margin_settings(job.get('margin'))
    screening = screen_settings(job)
    confirmation_count = max(repeats, margin['min_repeats'])
    allow_marginal = job.get('allow_marginal', True)
    if not isinstance(allow_marginal, bool):
        raise ValueError('allow_marginal must be a boolean')
    interleave = job.get('native_interleave_confirmations', False)
    if not isinstance(interleave, bool):
        raise ValueError('native_interleave_confirmations must be a boolean')
    if interleave and screening['enabled']:
        raise ValueError('interleaved native confirmations require original CPU profiling')
    work_conserving = job.get('native_work_conserving', False)
    if not isinstance(work_conserving, bool):
        raise ValueError('native_work_conserving must be a boolean')
    if work_conserving and not interleave:
        raise ValueError('native slot refill requires confirmation interleaving')
    if work_conserving and measure is not None:
        raise ValueError('native slot refill cannot use a batch measurement override')
    config = rules.pinned_config(job['challenge']['epoch'])
    requirements = config['nanobody_requirements']
    fleet_policy = job.get('native_fleet_policy')
    if fleet_policy is not None:
        fleet_policy = selection.fleet_policy_settings(fleet_policy)
        if native_adapter is None:
            raise ValueError('native fleet policy requires registered reference evidence')
    elif native_adapter is not None:
        raise ValueError('registered reference requires explicit native fleet policy')

    def decide_native(values, requirements, **options):
        return selection.decide(values, requirements,
            **({'fleet_policy': fleet_policy} if fleet_policy is not None else {}), **options)

    resume_identity = native_resume_identity(job, directory, requirements)
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
                  native_scheduling=('bounded_slot_refill' if work_conserving else
                      'interleaved_confirmations' if interleave else 'confirmation_barriers'),
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
    result['fresh_admission_limit'] = fresh_limit
    if fleet_policy is not None:
        result['native_fleet_policy'] = fleet_policy
        result['native_inconclusive_observations'] = []
        result['native_first_failure_rescue'] = {'fraction_of_fresh': .25, 'new_reservations': 0,
                                                'carried_reservations': 0, 'available_quota': 0}
    if native_adapter is not None:
        result['evaluation_counter_scope'] = 'registered_native_reference_calls'
        result['native_reference'] = {'registration_sha256': reference_settings['registration_sha256'],
            'policy': reference_settings['registration']['policy'], 'gpus': reference_settings['gpus'],
            'workers_per_gpu': reference_settings['workers_per_gpu'],
            'actual_validator_worker_attested': False}
        if reference_settings.get('resident') is True:
            result['native_reference']['resources'] = resources
    result['native_admission'] = {'deadline_clamped': timed_admission,
        'minimum_call_window_seconds': minimum_window, 'cleanup_reserve_seconds': cleanup_reserve,
        'candidate_timeout_cap_seconds': budget, 'allocations': []}

    def record(reason):
        result['rejection_counts'][reason] = result['rejection_counts'].get(reason, 0) + 1

    def checkpoint_measurements(values, sequence):
        if fleet_policy is not None:
            if any(not pr100_native_reference.authoritative_measurement(value) for value in values):
                raise ValueError('non_authoritative_native_history')
            return list(values)  # Complete biological failures are never erased.
        kept = []
        for value in values:
            if not selection.authoritative_cpu_measurement(value):
                continue
            observed = journal_module.observation(value, requirements, sequence)
            if observed['outcome'] == 'rejected':
                return []  # A real negative cannot be erased by a later draw.
            if observed['outcome'] == 'passed':
                kept.append(value)
        return kept

    def wants_repeat(values, sequence, required, *, claim=False, for_checkpoint=False):
        if fleet_policy is None:
            return (len(values) < required and decide_native(values, requirements,
                sequence=sequence, repeats=repeats, margin=margin)['tier'] is not None)
        if sequence in inconclusive_sequences and not for_checkpoint:
            return False
        quota = int(len(fresh_admitted) * .25)
        available = sequence in rescue_reserved or len(rescue_new) < quota
        wanted = selection.should_repeat(values, requirements, sequence=sequence,
            repeats=required, margin=margin, fleet_policy=fleet_policy,
            failed_first_rescue_available=available)
        first_failure = len(values) == 1 and values[0].get('passed') is False
        if wanted and first_failure and claim and sequence not in rescue_reserved:
            rescue_reserved.add(sequence); rescue_new.add(sequence)
            if sequence in confirmation_progress:
                confirmation_progress[sequence]['native_resume']['failed_first_rescue_reserved'] = True
            result['native_first_failure_rescue']['new_reservations'] = len(rescue_new)
        result['native_first_failure_rescue']['available_quota'] = quota
        return wanted

    def flush():
        result['observation_journal'] = journal.progress()
        admitted = started | screened | screen_admitted
        result['attempted_sequence_hashes'] = sorted(rules.sequence_hash(s) for s in admitted)
        result['fresh_admitted_sequence_hashes'] = sorted(rules.sequence_hash(s) for s in fresh_admitted)
        result['fresh_admitted_unique'] = len(fresh_admitted)
        result['fresh_never_started_sequence_hashes'] = sorted(rules.sequence_hash(r['sequence']) for r in rows
            if r.get('native_resume') is None and r['sequence'] not in fresh_admitted)
        result['deferred_candidates'] = [r for r in rows if r['sequence'] not in admitted]
        result['deferred_confirmations'] = []
        for row in rows:
            key = row['sequence']; values = checkpoint_measurements(measurements.get(key, []), key)
            required = max(confirmation_count if key in screen_evidence else repeats,
                           resumed_required.get(key, 0))
            if values and wants_repeat(values, key, required, claim=fleet_policy is not None,
                                       for_checkpoint=True):
                result['deferred_confirmations'].append({**row, 'native_resume': {
                    'schema': 'pr100-native-confirmation-checkpoint-v1',
                    'sequence_hash': rules.sequence_hash(key), 'identity': resume_identity,
                    'measurements': values, 'required_cpu_observations': required,
                    'accelerated_screen': screen_evidence.get(key),
                    **({'failed_first_rescue_reserved': key in rescue_reserved,
                        'fleet_policy': fleet_policy} if fleet_policy is not None else {}),
                    'execution_host': job.get('native_host', 'master')}})
        if fleet_policy is not None:
            result['native_complete_histories'] = [
                {'sequence_hash': rules.sequence_hash(key), 'measurements': values,
                 'fleet_policy': fleet_policy, 'identity': resume_identity,
                 'execution_host': job.get('native_host', 'master'),
                 'request_sha256s': [value['native_reference']['receipt']['request_sha256'] for value in values]}
                for key, values in measurements.items()]
            # Every complete draw stays in this attempt's current-history file,
            # including terminal support failures used to revoke older evidence.
            pr100_filter.atomic(directory / 'native-confirmations.json', {
                'challenge': job['challenge'], 'records': list(confirmation_progress.values())})
        pr100_filter.atomic(directory / 'result.json', result)

    phase_budget = job.get('phase_budget_seconds')
    if phase_budget is not None and (isinstance(phase_budget, bool)
                                     or not isinstance(phase_budget, (int, float))
                                     or phase_budget <= 0):
        raise ValueError('phase_budget_seconds must be a positive number')
    # Without an explicit phase budget the controller owns the wall clock; we do
    # not invent a deadline that would mark candidates not_run for no reason.
    deadline = (None if phase_budget is None else
                (reference_phase_started if reference_phase_started is not None else clock()) + float(phase_budget))

    def out_of_time():
        return deadline is not None and clock() + minimum_window + cleanup_reserve > deadline

    measurements = {}
    confirmation_progress = {}
    rescue_reserved, rescue_new, inconclusive_sequences = set(), set(), set()
    started = set()
    fresh_admitted = set()
    screened = set()
    screen_evidence = {}
    screen_admitted = set()
    resumed_required = {}
    audited = set()
    screen_skipped = set()
    blocked = []
    for row in rows:
        saved = row.get('native_resume')
        if saved is None:
            continue
        if not isinstance(saved, dict):
            raise ValueError('invalid native confirmation checkpoint')
        if saved.get('identity') != resume_identity or not resume_identity['resume_eligible']:
            blocked.append({**row, 'continuation_blocked': 'native_runtime_identity_mismatch'})
            continue
        values = saved.get('measurements') if isinstance(saved, dict) else None
        required = saved.get('required_cpu_observations') if isinstance(saved, dict) else None
        if fleet_policy is not None:
            if saved.get('fleet_policy') != fleet_policy:
                raise ValueError('native fleet checkpoint policy mismatch')
            if saved.get('failed_first_rescue_reserved') is True:
                rescue_reserved.add(row['sequence'])
                result['native_first_failure_rescue']['carried_reservations'] += 1
        if (not resume_identity['resume_eligible']
                or saved.get('schema') != 'pr100-native-confirmation-checkpoint-v1'
                or saved.get('identity') != resume_identity
                or saved.get('sequence_hash') != rules.sequence_hash(row['sequence'])
                or not isinstance(values, list) or not 1 <= len(values) < 64
                or type(required) is not int or not len(values) <= required <= 64
                or not (wants_repeat(values, row['sequence'], required)
                        or decide_native(values, requirements, sequence=row['sequence'],
                                         repeats=repeats, margin=margin)['tier'] is not None)):
            raise ValueError('invalid native confirmation checkpoint')
        key = row['sequence']
        measurements[key] = list(values); started.add(key); resumed_required[key] = required
        if fleet_policy is not None:
            confirmation_progress[key] = {**row, 'native_resume': {**saved, 'measurements': list(values)}}
        if saved.get('accelerated_screen') is not None:
            identity = screen_identity(saved['accelerated_screen'], key, job['challenge'])
            if identity is None:
                raise ValueError('invalid accelerated confirmation checkpoint')
            screen_evidence[key] = identity
    blocked_keys = {r['sequence'] for r in blocked}
    rows = [r for r in rows if r['sequence'] not in blocked_keys]
    result['blocked_confirmations'] = blocked
    journal = journal_module.AsyncJournal(directory, job, requirements)
    result['carried_cpu_observations'] = sum(map(len, measurements.values()))
    result['observation_journal'] = journal.progress()
    flush()

    def observation_error(_error_type):
        # Diagnostics cannot convert a valid scientific result to failure.
        result['observation_errors'] += 1

    def admit_fresh(batch):
        added = {r['sequence'] for r in batch if r['sequence'] not in resumed_required} - fresh_admitted
        if len(fresh_admitted) + len(added) > fresh_limit:
            raise ValueError('fresh native admission budget exceeded')
        fresh_admitted.update(added)

    def prepare_measurements(batch):
        allocated = min(budget, deadline - clock() - cleanup_reserve) if deadline is not None else budget
        if timed_admission and allocated < minimum_window:
            raise TimeoutError('native admission window closed')
        result['native_admission']['allocations'].append({'candidate_count': len(batch),
            'timeout_seconds': allocated if timed_admission else budget,
            'deadline_limited': timed_admission and allocated < budget})
        by_sequence = {row['sequence']: row for row in batch}
        admit_fresh(batch)
        expected = {key: len(measurements.get(key, [])) + 1 for key in by_sequence}
        started.update(by_sequence)
        result['not_started'] = result['input_count'] - len(started)
        result['in_flight'] += len(batch)
        # Persist admission before starting scientific calls. An interrupted phase
        # must never cause these candidates to be blindly evaluated again.
        flush()

        completed_attempts = set()
        def completed(key, value):
            if key in completed_attempts or len(measurements.get(key, [])) >= expected[key]:
                return
            if (fleet_policy is not None
                    and not isinstance(value, pr100_native_reference.Inconclusive)
                    and not pr100_native_reference.authoritative_measurement(value)):
                # Validate before marking completion. Adapter callbacks may
                # report diagnostic errors; the final collection must still
                # raise on an unauthenticated scientific result.
                raise ValueError('unattested_native_result')
            completed_attempts.add(key)
            if fleet_policy is not None and isinstance(value, pr100_native_reference.Inconclusive):
                inconclusive_sequences.add(key)
                if value.get('native_started') is False and not measurements.get(key):
                    started.discard(key); fresh_admitted.discard(key)
                    result['not_started'] = result['input_count'] - len(started)
                result['in_flight'] -= 1
                result['measurement_outcomes']['inconclusive'] += 1
                result['native_inconclusive_observations'].append({
                    'sequence_hash': rules.sequence_hash(key), 'after_complete_draws': len(measurements.get(key, [])),
                    'error': dict(value)})
                # This process-local type can only be produced by the owned
                # runner. An arbitrary serialized error cannot erase a failure.
                flush()
                return
            measurements.setdefault(key, []).append(value)
            result['completed'] += 1
            counter = 'fresh_evaluations' if expected[key] == 1 else 'repeat_evaluations'
            result[counter] += 1
            result['total_profile_evaluations'] = result['completed'] + result['native_screen']['completed_profiles']
            result['unique_profiled_candidates'] = len(set(measurements) | screened)
            result['not_completed'] = result['input_count'] - len(measurements)
            result['in_flight'] -= 1
            # Dedicated continuation checkpoints are not audit journal records.
            # Save each completion before a slower sibling can exhaust the phase.
            required = max(confirmation_count if key in screen_evidence else repeats,
                           resumed_required.get(key, 0))
            verdict = decide_native(measurements[key], requirements, sequence=key,
                                       repeats=repeats, margin=margin)
            confirmed = checkpoint_measurements(measurements[key], key)
            if confirmed:
                confirmation_progress[key] = {**by_sequence[key], 'native_resume': {
                    'schema': 'pr100-native-confirmation-checkpoint-v1',
                    'sequence_hash': rules.sequence_hash(key), 'identity': resume_identity,
                    'measurements': confirmed, 'required_cpu_observations': required,
                    'accelerated_screen': screen_evidence.get(key),
                    **({'failed_first_rescue_reserved': key in rescue_reserved,
                        'fleet_policy': fleet_policy} if fleet_policy is not None else {}),
                    'execution_host': job.get('native_host', 'master')}}
            else:
                confirmation_progress[key] = {'sequence_hash': rules.sequence_hash(key),
                    'terminal_reason': terminal_rejection_reason(
                        measurements[key], requirements, key, verdict['reason'])}
            try:
                pr100_filter.atomic(directory / 'native-confirmations.json', {
                    'challenge': job['challenge'], 'records': list(confirmation_progress.values())})
            except OSError:
                observation_error('ConfirmationCheckpointWriteError')
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

        return completed, allocated

    def measure_and_record(batch):
        completed, allocated = prepare_measurements(batch)
        if measure is None:
            deadline_options = {'timeout_seconds': allocated} if timed_admission else {}
            values, acquired, elapsed = measure_batch(
                [r['sequence'] for r in batch], workers, clock=clock,
                on_result=completed, on_result_error=observation_error,
                **({'develop': native_adapter.develop} if native_adapter is not None else {}),
                **({'executor_workers':scientific_workers} if reference_settings is not None
                   and reference_settings.get('resident') is True else {}),
                **deadline_options)
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
        admit_fresh(batch)
        screen_admitted.update(r['sequence'] for r in batch)
        flush()
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
        # Late admissions use a bounded native timeout, while the supervisor
        # retains the absolute phase and epoch deadlines.
        measured_rows = []
        for offset in range(0, len(rows), workers):
            original_batch = rows[offset:offset + workers]
            complete = [r for r in original_batch if r['sequence'] in resumed_required
                        and len(measurements[r['sequence']]) >= max(repeats, resumed_required[r['sequence']])]
            measured_rows.extend(complete)
            original_batch = [r for r in original_batch if r not in complete]
            if not original_batch:
                continue
            if out_of_time():
                break
            carry = [r for r in original_batch if r['sequence'] in resumed_required]
            fresh = [r for r in original_batch if r['sequence'] not in resumed_required]
            fresh = fresh[:max(0, fresh_limit - len(fresh_admitted))]
            if not fresh and not carry:
                continue  # Confirmation rows in later batches still get a turn.
            batch, held = screen_batch_rows(fresh, offset // workers) if fresh else ([], [])
            batch = carry + batch
            if not batch or out_of_time():
                break
            _acquired, elapsed = measure_and_record(batch)
            measured_rows.extend(batch)
            result['last_batch_seconds'] = elapsed
            flush()
            disagreements = [row for row in batch if row['sequence'] in audited and decide_native(
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
            needed = lambda row: max(confirmation_count if row['sequence'] in screen_evidence else repeats,
                                     resumed_required.get(row['sequence'], 0))
            for _ in range(max((needed(r) - 1 for r in batch), default=0)):
                # Repeat eligibility is independent of fleet admission. The
                # registered route has a bounded first-failure rescue quota.
                pending = [row for row in pending if wants_repeat(
                    measurements.get(row['sequence'], []), row['sequence'], needed(row), claim=True)]
                if not pending or out_of_time():
                    break
                _acquired, elapsed = measure_and_record(pending)
                result['last_batch_seconds'] = elapsed
                flush()
        # Include measured failures so the final classification counts each
        # candidate exactly once, including failures on confirmation passes.
        return measured_rows

    def collect_interleaved_measurements():
        # Give confirmations the first slots in every wave, filling remaining
        # slots with fresh work. A handful of survivors must not leave the
        # other native workers idle for several entire confirmation rounds.
        # Each sequence still has at most one call per wave, and every next
        # draw requires a passing preceding observation.
        from collections import deque
        def needed(row):
            key = row['sequence']
            return max(confirmation_count if key in screen_evidence else repeats,
                       resumed_required.get(key, 0))

        measured = {r['sequence'] for r in rows if r['sequence'] in resumed_required
                    and len(measurements[r['sequence']]) >= needed(r)}
        pending = deque(r for r in rows if r['sequence'] in resumed_required
                        and r['sequence'] not in measured)
        fresh = deque(r for r in rows if r['sequence'] not in resumed_required)
        while pending or fresh:
            if out_of_time():
                break
            # Finish near-complete carried work before spreading the remaining
            # window over many still-incomplete candidates. Stable sorting
            # retains queue order when the remaining observation count ties.
            pending = deque(sorted(pending, key=lambda row:
                needed(row) - len(measurements.get(row['sequence'], []))))
            batch = []
            while pending and len(batch) < workers:
                batch.append(pending.popleft())
            capacity = min(workers - len(batch), fresh_limit - len(fresh_admitted))
            if batch and deadline is not None:
                # A slow fresh sibling holds the entire wave open. Mix only
                # when the remaining confirmations can still receive their
                # full timeout caps; otherwise use a confirmation-only wave.
                rounds = max(needed(row) - len(measurements[row['sequence']]) for row in batch)
                if deadline - clock() < rounds * budget + cleanup_reserve:
                    capacity = 0
            for _ in range(min(capacity, len(fresh))):
                batch.append(fresh.popleft())
            if not batch:
                break
            _acquired, elapsed = measure_and_record(batch)
            measured.update(r['sequence'] for r in batch)
            result['last_batch_seconds'] = elapsed
            flush()
            for row in batch:
                key = row['sequence']
                if wants_repeat(measurements.get(key, []), key, needed(row), claim=True):
                    pending.append(row)
        return [r for r in rows if r['sequence'] in measured]

    def collect_refill_measurements():
        from collections import deque
        from nanobody_opt import pr100_native_refill
        needed = lambda row: max(confirmation_count if row['sequence'] in screen_evidence else repeats,
                                 resumed_required.get(row['sequence'], 0))
        measured = {r['sequence'] for r in rows if r['sequence'] in resumed_required
                    and len(measurements[r['sequence']]) >= needed(r)}
        pending = deque(r for r in rows if r['sequence'] in resumed_required
                        and r['sequence'] not in measured)
        fresh = deque(r for r in rows if r['sequence'] not in resumed_required)

        def next_calls(free):
            if not free or out_of_time():
                return []
            ordered = sorted(pending, key=lambda row:
                needed(row) - len(measurements.get(row['sequence'], [])))
            pending.clear(); pending.extend(ordered)
            batch = [pending.popleft() for _ in range(min(free, len(pending)))]
            capacity = min(free - len(batch), fresh_limit - len(fresh_admitted), len(fresh))
            batch.extend(fresh.popleft() for _ in range(capacity))
            if not batch:
                return []
            record, allocated = prepare_measurements(batch)
            return [{'sequence': row['sequence'], 'row': row, 'record': record,
                     'record_error': observation_error, 'timeout_seconds': allocated,
                     'deadline': deadline - cleanup_reserve if deadline is not None else None,
                     'minimum_window': minimum_window}
                    for row in batch]

        def completed(call, value):
            row = call['row']; key = row['sequence']
            if isinstance(value, pr100_native_refill.NotStarted):
                # This result is produced only before invoking the adapter.
                # Persist an exact never-started proof, not an inconclusive
                # scientific observation or a terminal native rejection.
                result['in_flight'] -= 1
                if key not in measurements:
                    started.discard(key); fresh_admitted.discard(key)
                result['not_started'] = result['input_count'] - len(started)
                flush()
                return
            call['record'](key, value)
            measured.add(key)
            # The preceding complete call has released its scientific worker.
            # This candidate's next draw gets priority on the newly free slot.
            if wants_repeat(measurements.get(key, []), key, needed(row), claim=True):
                pending.append(row)
            flush()

        result['native_refill'] = (refill or pr100_native_refill.run)(
            next_calls, completed, workers, clock=clock,
            **({'develop': native_adapter.develop} if native_adapter is not None else {}),
            **({'executor_workers':scientific_workers} if reference_settings is not None
               and reference_settings.get('resident') is True else {}))
        return [r for r in rows if r['sequence'] in measured]

    try:
        survivors = (collect_refill_measurements() if work_conserving else
                     collect_interleaved_measurements() if interleave else collect_measurements())
    finally:
        try:
            # A resident pool owns its existing cooperative leases until the
            # native measurement boundary. Verify owned cleanup before any
            # completed phase can hand candidates to novelty/publication.
            shutdown = getattr(native_adapter, 'shutdown', None)
            if shutdown is not None:
                shutdown()
        finally:
            journal.close(timeout=1.0)
            result['observation_journal'] = journal.progress()
    verdicts = {}

    passers = []
    for row in survivors:
        key = row['sequence']
        verdict = decide_native(measurements.get(key, []), requirements, sequence=key,
                                   repeats=repeats, margin=margin,
                                   cpu_confirmation_minimum=max(confirmation_count if key in screen_evidence else 0,
                                                                resumed_required.get(key, 0) if fleet_policy is None else 0))
        verdicts[key] = verdict
        if verdict['tier'] is None:
            record(verdict['reason'] or 'rejected')
            continue
        record('passed' if verdict['tier'] == 'A' else verdict['reason'])
        evidence = {'challenge': job['challenge'], 'filter_sha': rules.FILTER_SHA,
                    'annotation': row['pr100_annotation'],
                    'developability': verdict['measurements'][verdict.get('primary_passing_index', 0)]}
        if fleet_policy is not None:
            evidence['native_fleet_selection'] = verdict['native_fleet_selection']
        if fleet_policy is None and (key in screen_evidence or key in resumed_required):
            # Carry the confirmation requirement across export and Pi validation;
            # a forged tier label cannot discard it at a publication boundary.
            evidence['native_screen_confirmation'] = {
                'screen': screen_evidence.get(key), 'minimum': verdict['cpu_confirmation_minimum'],
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
    result['terminal_native_outcomes'] = {}
    for key, verdict in verdicts.items():
        reason = terminal_rejection_reason(measurements.get(key, []), requirements, key, verdict['reason'])
        fleet_terminal = (fleet_policy is not None and measurements.get(key)
                          and reason in ('observed_support_too_low', 'no_measured_pass')
                          and not wants_repeat(measurements[key], key,
                              max(repeats, resumed_required.get(key, 0)), for_checkpoint=True))
        if verdict['tier'] is None and (reason in journal_module.REJECTION_REASONS or fleet_terminal):
            result['terminal_native_outcomes'][rules.sequence_hash(key)] = reason
    result['native_classification_complete'] = True
    flush()

    # Funnel order: annotation -> native developability -> archive novelty.
    archive = directory / 'archive.csv'
    if passers and archive.is_file():
        if file_digest(archive) != job['archive_sha256']:
            raise ValueError('archive snapshot changed')
        passers, archive_rows = check_novelty(passers, archive, novelty=novelty,
                                             runtime=job.get('novelty_runtime'))
        result.update(novelty_checked=True, archive_rows=archive_rows,
                      novelty_runtime_sha256=(job.get('novelty_runtime') or {}).get('identity_sha256'))
    result['candidates'] = selection.order_candidates(passers, allow_marginal=allow_marginal)
    result['tier_counts'] = {tier: sum(1 for r in result['candidates'] if r['tier'] == tier)
                             for tier in ('A', 'B')}
    result['cpu_not_run'] = result['input_count'] - len(started)
    result['not_run'] = result['input_count'] - len(started | screened)
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
                     'candidate_count': len(rows),
                     'candidate_hashes': [rules.sequence_hash(r['sequence']) for r in rows]})
    return jobs


def collect_scores(out, jobs):
    """Merge whatever per-GPU probe groups finished into score/report.json (partial allowed)."""
    groups, shards, scorer_hashes, settings, admitted = [], [], set(), None, set()
    for gpu_job in jobs:
        gpu = gpu_job['gpu']
        entry = {'gpu': gpu, 'candidate_count': gpu_job['candidate_count']}
        try:
            lease = json.loads(Path(gpu_job['receipt']).read_text())
            entry['lease'] = lease.get('status')
            if lease.get('reason') in ('insufficient_gpu_headroom', 'gpu_not_idle', 'coordination_lock_busy',
                                      'gpu_missing', 'gpu_telemetry_failed'):
                entry['lease_reason'] = lease['reason']
        except (OSError, ValueError):
            lease = None
            entry['lease'] = None
        try:
            report = json.loads((out / f'gpu-{gpu}' / 'report.json').read_text())
        except (OSError, ValueError):
            report = None
        entry['probe_status'] = report.get('status') if report else None
        if lease is None or lease.get('child_pid') is not None or report is not None:
            admitted.update(gpu_job.get('candidate_hashes', []))
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
              'attempted_sequence_hashes': sorted(admitted),
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
    planned = sorted({key for item in jobs for key in item['candidate_hashes']})
    pr100_filter.atomic(directory / 'result.json', {
        'challenge': job['challenge'], 'status': 'running', 'candidates': [],
        'attempted_sequence_hashes': planned,
        'deferred_sequence_hashes': sorted({rules.sequence_hash(s) for s in sequences} - set(planned))})
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
            pr100_filter.atomic(path, {k: v for k, v in gpu_job.items()
                                      if k not in ('candidate_count', 'candidate_hashes')})
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
              'attempted_sequence_hashes': report['attempted_sequence_hashes'],
              'deferred_sequence_hashes': sorted({rules.sequence_hash(s) for s in sequences}
                                                 - set(report['attempted_sequence_hashes'])),
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
        result = {'sequence': sequence_phase, 'native': native_phase, 'novelty': novelty_phase,
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
