"""Durable same-epoch attempts. Journals remain audit-only, never reusable evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from nanobody_opt import rules, pr100_fresh_budget as fresh_budget

SCHEMA = 'pr100-epoch-continuation-v1'


def failure_details(error):
    """Identify refusals without persisting exception text, locals or raw tracebacks."""
    codes = {
        'saved phase input or source identity mismatch': 'phase_binding_mismatch',
        'saved phase epoch mismatch': 'phase_epoch_mismatch',
        'native dispatch routing or settings changed': 'native_dispatch_binding_mismatch',
        'saved fresh admission plan input changed': 'fresh_plan_input_changed',
        'saved native input exceeds or conflicts with fresh reservation': 'fresh_saved_input_conflict',
        'fresh admission outside persisted epoch reservation': 'fresh_admission_outside_reservation',
        'fresh admission and never-started evidence conflict': 'fresh_admission_conflict',
        'incomplete carried native validity': 'native_validity_incomplete',
        'score phase returned no scored group': 'score_no_group',
        'scorer target changed': 'score_target_mismatch',
        'duplicate scorer group': 'score_duplicate_group',
        'continuation scoring identity changed': 'score_identity_mismatch',
        'continuation target changed': 'continuation_target_mismatch',
    }
    message = error.args[0] if len(error.args) == 1 and type(error.args[0]) is str else None
    result = {'error_code': codes.get(message, 'unclassified')}
    root = Path(__file__).resolve().parent
    allowed = {'pr100_epoch_continuation.py', 'pr100_live_controller.py',
               'pr100_native_dispatch.py', 'pr100_fresh_budget.py', 'rules.py',
               'pr100_novelty_runtime.py'}
    trace = error.__traceback__
    while trace is not None:
        source = Path(trace.tb_frame.f_code.co_filename).resolve()
        if source.parent == root and source.name in allowed:
            result['error_location'] = f'{source.name}:{trace.tb_lineno}'
        trace = trace.tb_next
    return result


def empty_score_attempt(files, challenge, rows):
    """Recognize an explicit zero-score result without inventing scorer identity.

    A partial run (including unavailable GPU leases) can finish with no groups
    and therefore no scorer settings. Missing/malformed artifacts still follow
    the strict merge path. Admission evidence decides which rows may be retried.
    """
    value = json.loads(files.get('result.json', b'{}'))
    report = json.loads(files.get('score/report.json', b'{}'))
    if not isinstance(value, dict) or not isinstance(report, dict):
        return False
    if (value.get('challenge') != challenge or value.get('epoch') != challenge['epoch']
            or value.get('status') not in ('complete', 'partial')
            or value.get('status') != report.get('status')
            or value.get('input_count') != len(rows) or value.get('candidates') != []
            or type(value.get('scored_candidates')) is not int or value['scored_candidates'] != 0
            or report.get('schema') != 'pr100-live-score-v1' or report.get('groups') != []
            or type(report.get('scored_candidates')) is not int or report['scored_candidates'] != 0
            or report.get('registered_scorer_sha256') is not None
            or 'score/scorer_settings.json' in files):
        return False
    attempted, deferred = value.get('attempted_sequence_hashes'), value.get('deferred_sequence_hashes')
    if (not isinstance(attempted, list) or not isinstance(deferred, list)
            or any(type(h) is not str for h in attempted + deferred)
            or report.get('attempted_sequence_hashes') != attempted):
        return False
    expected = {rules.sequence_hash(r['sequence']) for r in rows}
    return (len(set(attempted + deferred)) == len(attempted) + len(deferred)
            and set(attempted + deferred) == expected)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default

def union_rows(old, new):
    rows = {row['sequence']: row for row in old}
    for row in new:
        previous = rows.get(row['sequence'])
        if previous is None or len(row.get('measurements', [])) > len(previous.get('measurements', [])):
            rows[row['sequence']] = row
    return list(rows.values())


def reconcile_native_histories(state, value, assigned_hashes):
    """Keep complete registered draw histories monotone within this epoch.

    Store compact request+measurement digests, not another copy of every trace.
    A shorter replay is ignored; a conflicting or mixed history is refused. Any
    extended history supersedes the old candidate evidence, even if the new
    result is rejected or has not completed novelty. Raw receipts remain in the
    immutable collected attempt artifacts and the worker's checkpoints.
    """
    from nanobody_opt import pr100_fleet_probability as probability
    from nanobody_opt import pr100_live_selection as selection
    records = value.get('native_complete_histories')
    previous = state.get('native_reference_histories', {})
    registered_rows = [r for r in value.get('candidates', [])
                       if r.get('pr100_evidence', {}).get('developability', {}).get('evidence_backend')
                       == 'native_reference']
    if records is None:
        if registered_rows or set(value.get('terminal_native_outcomes', {})) & set(previous):
            raise ValueError('native complete history missing')
        return
    if (not isinstance(records, list) or value.get('challenge') != state['challenge']
            or value.get('epoch') != state['challenge']['epoch']):
        raise ValueError('native complete history binding mismatch')
    updates, seen, stale, superseded = {}, set(), set(), set()
    complete, preserved_checkpoints = {}, {}
    terminal = dict(value.get('terminal_native_outcomes', {}))
    pending = {rules.sequence_hash(r['sequence']): r.get('native_resume', {})
               for r in value.get('deferred_confirmations', []) if isinstance(r, dict) and 'sequence' in r}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('native complete history invalid')
        key, measurements = record.get('sequence_hash'), record.get('measurements')
        if key not in assigned_hashes or key in seen or not isinstance(measurements, list):
            raise ValueError('native complete history outside assignment or duplicated')
        seen.add(key)
        if not measurements:
            if record.get('request_sha256s') != []:
                raise ValueError('empty native history has draw identifiers')
            continue
        policy = selection.fleet_policy_settings(record.get('fleet_policy'))
        summary = probability.from_measurements(measurements,
            minimum_smoothed_fraction=policy['minimum_smoothed_fraction'])
        if summary['sequence_hash'] != key:
            raise ValueError('native complete history sequence mismatch')
        requests = [r['request_sha256'] for r in summary['observations']]
        if record.get('request_sha256s') != requests:
            raise ValueError('native complete history request mismatch')
        digests = [rules.digest(row) for row in measurements]
        incoming = {'registration': summary['registration'], 'policy': policy,
                    'request_sha256s': requests, 'measurement_sha256s': digests,
                    'admitted': summary['admit_to_binding_comparison'],
                    'complete_draws': summary['complete_draws'],
                    'passing_draws': summary['passing_draws'], 'reason': summary['reason']}
        carried = pending.get(key, {})
        rescue_pending = (summary['complete_draws'] == 1 and summary['passing_draws'] == 0
            and carried.get('failed_first_rescue_reserved') is True
            and carried.get('measurements') == measurements and carried.get('fleet_policy') == policy
            and type(carried.get('required_cpu_observations')) is int
            and 2 <= carried['required_cpu_observations'] <= 3)
        incoming['bounded_rescue_pending'] = rescue_pending
        old = previous.get(key)
        if old is not None:
            if old.get('registration') != incoming['registration'] or old.get('policy') != policy:
                raise ValueError('native complete history identity changed')
            old_ids, old_digests = old.get('request_sha256s', []), old.get('measurement_sha256s', [])
            shared = min(len(old_ids), len(requests))
            if old_ids[:shared] != requests[:shared] or old_digests[:shared] != digests[:shared]:
                raise ValueError('native complete history conflicts with prior evidence')
            if len(requests) < len(old_ids):
                stale.add(key); terminal.pop(key, None)
                if old.get('admitted') is False and not old.get('bounded_rescue_pending'):
                    terminal[key] = old['reason']
                else:
                    checkpoint = next((r for r in state.get('pending_confirmation', [])
                        if rules.sequence_hash(r['sequence']) == key
                        and r.get('native_resume', {}).get('fleet_policy') == old['policy']
                        and [rules.digest(m) for m in r.get('native_resume', {}).get('measurements', [])]
                            == old_digests), None)
                    if checkpoint is not None:
                        preserved_checkpoints[key] = checkpoint
                continue
            if (len(requests) == len(old_ids) and old.get('bounded_rescue_pending')
                    and not rescue_pending):
                checkpoint = next((r for r in state.get('pending_confirmation', [])
                    if rules.sequence_hash(r['sequence']) == key
                    and r.get('native_resume', {}).get('measurements') == measurements
                    and r.get('native_resume', {}).get('fleet_policy') == policy
                    and r.get('native_resume', {}).get('failed_first_rescue_reserved') is True), None)
                if checkpoint is None:
                    raise ValueError('reserved native rescue checkpoint missing')
                preserved_checkpoints[key] = checkpoint
                rescue_pending = incoming['bounded_rescue_pending'] = True
            if len(requests) > len(old_ids):
                superseded.add(key)
        updates[key] = incoming
        complete[key] = (measurements, policy)
        if incoming['admitted'] or rescue_pending:
            # One failed draw cannot overturn a supported aggregate decision.
            terminal.pop(key, None)
        else:
            terminal[key] = incoming['reason']
    for row in registered_rows:
        key = rules.sequence_hash(row['sequence'])
        if key in stale:
            continue
        history = row['pr100_evidence'].get('native_fleet_selection', {})
        if (key not in complete or history.get('measurements') != complete[key][0]
                or history.get('policy') != complete[key][1]
                or updates[key]['admitted'] is not True):
            raise ValueError('candidate omits complete native history')
    if (set(terminal) & set(previous)) - set(complete) - stale:
        raise ValueError('registered rejection lacks complete native history')
    # Apply only after the entire incoming result is checked.
    state['native_reference_histories'] = {**previous, **updates}
    state['native_rows'] = [r for r in state.get('native_rows', [])
                            if rules.sequence_hash(r['sequence']) not in superseded]
    value['candidates'] = [r for r in value.get('candidates', [])
                           if rules.sequence_hash(r['sequence']) not in stale]
    value['deferred_confirmations'] = [r for r in value.get('deferred_confirmations', [])
        if rules.sequence_hash(r['sequence']) not in stale | set(preserved_checkpoints)]
    value['deferred_confirmations'].extend(preserved_checkpoints.values())
    value['terminal_native_outcomes'] = terminal


def recover_native_checkpoints(state, value, assigned, required_observations):
    """Keep authenticated partial positives schedulable without blessing novelty.

    Reconciliation must run first. Only its current accepted history may become
    a checkpoint, bound to this assignment and the observed runtime. A complete
    checkpoint is consumed by the ordinary native/novelty path without another
    draw; an incomplete one retains the original bounded repeat requirement.
    """
    from nanobody_opt import pr100_live_selection as selection
    by_hash = {rules.sequence_hash(row['sequence']): row for row in assigned}
    pending = list(value.get('deferred_confirmations', []))
    queued = {rules.sequence_hash(row['sequence']) for row in pending}
    expected_requirements = rules.digest(rules.pinned_config(
        state['challenge']['epoch'])['nanobody_requirements'])
    for record in value.get('native_complete_histories', []):
        key = record['sequence_hash']; ledger = state.get('native_reference_histories', {}).get(key, {})
        values = record['measurements']
        if key in queued or not values or ledger.get('admitted') is not True:
            continue
        # Shorter stale artifacts were intentionally ignored by reconciliation.
        if ([rules.digest(row) for row in values] != ledger.get('measurement_sha256s')
                or record.get('request_sha256s') != ledger.get('request_sha256s')):
            continue
        original = by_hash.get(key)
        identity, registration = record.get('identity'), ledger.get('registration', {})
        policy = selection.fleet_policy_settings(record.get('fleet_policy'))
        if (original is None or not isinstance(identity, dict)
                or identity.get('resume_eligible') is not True
                or identity.get('challenge_sha256') != rules.digest(state['challenge'])
                or identity.get('requirements_sha256') != expected_requirements
                or identity.get('filter_sha') != rules.FILTER_SHA
                or any(identity.get(field) != registration.get(field) for field in
                       ('backend', 'registration_sha256', 'runtime_identity_sha256', 'policy_sha256'))
                or policy != ledger.get('policy')):
            raise ValueError('partial native checkpoint binding mismatch')
        prior = original.get('native_resume') or {}
        if not isinstance(prior, dict):
            raise ValueError('partial native checkpoint invalid')
        if prior and (prior.get('identity') != identity or prior.get('fleet_policy') != policy):
            raise ValueError('partial native checkpoint identity changed')
        host = prior.get('execution_host', record.get('execution_host'))
        if (host not in ('master', 'slave') or record.get('execution_host') != host):
            raise ValueError('partial native checkpoint host mismatch')
        required = prior.get('required_cpu_observations', required_observations)
        if (type(required) is not int or not 1 <= required <= 64
                or not 1 <= len(values) < 64):
            raise ValueError('partial native checkpoint repeat count invalid')
        pending.append({**original, 'native_resume': {
            'schema': 'pr100-native-confirmation-checkpoint-v1', 'sequence_hash': key,
            'identity': identity, 'measurements': values, 'fleet_policy': policy,
            'required_cpu_observations': max(required, len(values)),
            'execution_host': host,
            'failed_first_rescue_reserved': prior.get('failed_first_rescue_reserved') is True}})
        queued.add(key)
    value['deferred_confirmations'] = pending

def valid_rows(value, challenge):
    if not isinstance(value, dict) or value.get('challenge') != challenge:
        return []
    if value.get('status') != 'complete' or value.get('novelty_checked') is not True:
        return []
    config = rules.pinned_config(challenge['epoch'])
    rows = []
    for row in value.get('candidates', []):
        rules.validate_evidence(row.get('pr100_evidence'), row['sequence'], challenge, config)
        if row.get('layer2_passed') is not True or row.get('novelty', {}).get('validator_novel') is not True:
            raise ValueError('incomplete carried native validity')
        rows.append({**row, '_novelty_archive_sha256': value['archive_sha256']})
    return rows

def merge_scoring(state, scores, settings, scorer_hash, settings_hash):
    identity = {'scorer_sha256': scorer_hash, 'settings_sha256': rules.digest(settings),
                'settings_file_sha256': settings_hash, 'target': state['challenge']['target']}
    if state.get('scoring_identity') not in (None, identity):
        raise ValueError('continuation scoring identity changed')
    if scores.get('target') != state['challenge']['target']:
        raise ValueError('continuation target changed')
    state['scoring_identity'] = identity
    state['settings'] = settings
    state.setdefault('score_rows', {}).update(scores['per_nanobody_components']['0'])

def cleanup_verified(receipt):
    cleanup = receipt.get('cleanup', {})
    return cleanup.get('remaining_descendants') == 0 and cleanup.get('fixed_point_reached') is True

def novelty_runtime_ids(config):
    """Qualified annotation identities, independent of native CPU identity."""
    result = {}
    for host, runtime in config.get('hosts', {}).items():
        value = runtime.get('novelty_runtime', {}).get('identity_sha256')
        if value is not None:
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid novelty runtime identity')
            result[host] = value
    return result

def qualified_novelty_result(value, config, *, final=False):
    expected = novelty_runtime_ids(config)
    if not expected:
        return True  # Legacy/inert callers; real novelty execution requires qualification.
    if final:
        identity = expected.get('master')
        return (identity is not None and value.get('novelty_runtime_sha256') == identity
                and all(row.get('novelty', {}).get('runtime_sha256') == identity
                        for row in value.get('candidates', [])))
    hosts = value.get('native_host_status')
    if hosts:
        return all(host in expected and detail.get('novelty_runtime_sha256') == expected[host]
                   and detail.get('ownership_unresolved') is not True for host, detail in hosts.items())
    return value.get('novelty_runtime_sha256') in expected.values()

def novelty_proof_reusable(row, archive_sha256, config):
    expected = novelty_runtime_ids(config).get('master')
    return (row.get('_novelty_archive_sha256') == archive_sha256
            and (expected is None or row.get('novelty', {}).get('runtime_sha256') == expected))

def recover_owned_attempt_for_novelty(state, local, config, chain, stop_at):
    """Poll the exact old launches; never restart old-code scientific work."""
    from nanobody_opt import pr100_live_controller as c
    import base64
    active = state.get('active')
    if not active:
        return True
    def charge_uncertain(intent, request, phase, receipt):
        if (phase != 'native' or receipt.get('reason') == 'not_launched'
                or (intent.parent / 'native/result.json').is_file()):
            return
        incoming = json.loads(base64.b64decode(request['files']['input.json'], validate=True))
        uncertain = {rules.sequence_hash(row['sequence']) for row in incoming.get('candidates', [])
                     if row.get('native_resume') is None}
        state['native_attempted_hashes'] = sorted(set(state.get('native_attempted_hashes', [])) | uncertain)
        state['migration_uncertain_native_admissions'] = sorted(
            set(state.get('migration_uncertain_native_admissions', [])) | uncertain)
    directory = Path(local) / 'attempts' / str(active['number']).zfill(5)
    requests = [directory / (phase + '-request.json') for phase in ('sequence', 'native', 'score')]
    requests += [directory / 'native-hosts' / host / 'native-request.json' for host in ('master', 'slave')]
    for group, phase in (('novelty-rechecks', 'novelty'), ('score-batches', 'score')):
        requests.extend((directory / group).glob('*/' + phase + '-request.json'))
    for intent in requests:
        if not intent.is_file():
            continue
        saved = read(intent); request = saved['request']; phase = intent.name.removesuffix('-request.json')
        if request['spec']['epoch'] != state['challenge']['epoch']:
            raise ValueError('old novelty migration owner epoch mismatch')
        terminal_path = intent.parent / (phase + '-supervisor.json')
        terminal = read(terminal_path, {})
        if cleanup_verified(terminal) and (intent.parent / (phase + '-collected.json')).is_file():
            charge_uncertain(intent, request, phase, terminal)
            continue
        clock = c.clock_state(chain)
        answer = c.remote({**config, '_refresh_clock': lambda: c.clock_state(chain)}, saved['host'],
            {'operation': 'poll', 'directory': request['directory'], 'clock': clock,
             'transport_deadline_unix': stop_at})
        if answer.get('running') or not cleanup_verified(answer.get('receipt', {})):
            return False
        c.atomic(terminal_path, answer['receipt'])
        names = []
        for name, encoded in answer.get('files', {}).items():
            if name not in ('result.json', 'worker-receipt.json', 'native-confirmations.json',
                            'score/report.json', 'score/scorer_settings.json') and not c.re.fullmatch(
                    r'score/gpu-[0-9]+/length[0-9]+/exact_all[.]json', name):
                raise ValueError('unexpected old owner artifact')
            output = intent.parent / phase / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(base64.b64decode(encoded, validate=True)); names.append(name)
        c.atomic(intent.parent / (phase + '-collected.json'), {'files': sorted(names)})
        # No result does not prove no admission, including cached collections.
        charge_uncertain(intent, request, phase, answer['receipt'])
    # This local transition occurs only after every recorded owner is clean.
    state['active'] = None
    return True

def restore_for_qualified_novelty(state, local, config):
    """Restore native evidence hidden by old novelty, only after owner cleanup.

    Journals never supply reusable evidence. Full checkpoints return to their
    original native host for classification with zero new draws; partial ones
    retain their original missing-confirmation requirement.
    """
    expected = novelty_runtime_ids(config)
    if not expected:
        return True
    migration_key = rules.digest(expected)
    if state.get('qualified_novelty_migration', {}).get('identity_sha256') == migration_key:
        return True
    if state.get('active') or state.get('legacy_uncertain_phase'):
        return False
    from nanobody_opt import pr100_live_selection as selection
    from nanobody_opt import pr100_live_controller as c
    from nanobody_opt.pr100_native_journal import REJECTION_REASONS
    challenge = state['challenge']
    pinned = rules.pinned_config(challenge['epoch'])
    requirements = pinned['nanobody_requirements']
    sources = [Path(local)]
    attempt_numbers = {}
    for attempt in sorted((Path(local) / 'attempts').glob('[0-9][0-9][0-9][0-9][0-9]')):
        sources.append(attempt)
        attempt_numbers[attempt] = int(attempt.name)
        for host in ('master', 'slave'):
            directory = attempt / 'native-hosts' / host
            sources.append(directory); attempt_numbers[directory] = int(attempt.name)
    for directory in sources:
        if ((directory / 'native-request.json').exists()
                and not cleanup_verified(read(directory / 'native-supervisor.json', {}))):
            return False
    # Reconcile completed phases that may have finished between the final
    # old-controller write and cutover. These are collected results, not reruns.
    pending = list(state.get('pending_native', []))
    for directory in sources:
        value = read(directory / 'sequence/result.json', {})
        if (cleanup_verified(read(directory / 'sequence-supervisor.json', {}))
                and value.get('challenge') == challenge and value.get('status') == 'complete'):
            pending = union_rows(pending, value.get('candidates', []))
            state['generated_hashes'] = sorted(set(state.get('generated_hashes', [])) |
                set(value.get('generated_sequence_hashes', [])))
        score_dirs = [directory] + list((directory / 'score-batches').glob('*'))
        for score_dir in score_dirs:
            report = read(score_dir / 'score/score/report.json')
            if report is None or not cleanup_verified(read(score_dir / 'score-supervisor.json', {})):
                continue
            names = ['score/report.json', 'score/scorer_settings.json']
            names += ['score/' + group['file'] for group in report.get('groups', [])
                      if group.get('valid') is True and c.re.fullmatch(
                          r'gpu-[0-9]+/length[0-9]+/exact_all[.]json', str(group.get('file', '')))]
            files = {name: (score_dir / 'score' / name).read_bytes() for name in names
                     if (score_dir / 'score' / name).is_file()}
            scores, settings, scorer_hash = c.merge_scores(files, challenge['target'])
            merge_scoring(state, scores, settings, scorer_hash,
                          hashlib.sha256(files['score/scorer_settings.json']).hexdigest())
    rows = list(state.get('native_rows', []))
    novelty_rejected = set()
    for directory in sources:
        for operation in [directory] + list((directory / 'novelty-rechecks').glob('*')):
            try:
                if not cleanup_verified(read(operation / 'novelty-supervisor.json', {})):
                    continue
                value = read(operation / 'novelty/result.json', {})
                if (value.get('challenge') != challenge or value.get('status') != 'complete'
                        or value.get('novelty_checked') is not True
                        or qualified_novelty_result(value, config, final=True)):
                    continue
                intent = read(operation / 'novelty-request.json', {})
                import base64
                incoming = json.loads(base64.b64decode(intent['request']['files']['input.json'], validate=True))
                if any(incoming.get(k) != value.get(k) for k in ('challenge', 'epoch', 'target', 'archive_sha256')):
                    continue
                original = valid_rows(incoming, challenge)
                original_hashes = {rules.sequence_hash(row['sequence']) for row in original}
                kept = {rules.sequence_hash(row['sequence']) for row in value['candidates']}
                if not kept <= original_hashes:
                    continue
                novelty_rejected.update(original_hashes - kept)
                rows.extend(original)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
    records = list(state.get('pending_confirmation', []))
    rejected, attempted, checked, unusable = set(), set(), 0, 0
    for directory in sources:
        try:
            if not cleanup_verified(read(directory / 'native-supervisor.json', {})):
                if (directory / 'native-request.json').exists():
                    return False
                continue
            value = read(directory / 'native/result.json', {})
            if any(value.get(k) != challenge.get(k) for k in ('epoch', 'target')) or value.get('challenge') != challenge:
                continue
            intent = read(directory / 'native-request.json')
            if intent is not None:
                import base64
                incoming = json.loads(base64.b64decode(intent['request']['files']['input.json'], validate=True))
            else:
                incoming = read(directory / 'native-input.json')
            if incoming is not None:
                if any(incoming.get(k) != value.get(k) for k in ('challenge', 'epoch', 'target', 'archive_sha256')):
                    raise ValueError('migration native input binding mismatch')
                assigned = {rules.sequence_hash(row['sequence']) for row in incoming['candidates']}
            else:
                assigned = set(state.get('native_attempted_hashes', []))
            if (any(rules.sequence_hash(row['sequence']) not in assigned for row in value.get('candidates', []))
                    or set(value.get('attempted_sequence_hashes', [])) - assigned):
                raise ValueError('migration native result outside assigned cohort')
            if (any(set(value.get(key, [])) - assigned for key in (
                    'fresh_admitted_sequence_hashes', 'fresh_never_started_sequence_hashes',
                    'verified_fresh_never_started_sequence_hashes', 'fresh_admission_uncertain_sequence_hashes'))
                    or any(rules.sequence_hash(row['sequence']) not in assigned
                           for row in value.get('deferred_candidates', []))):
                raise ValueError('migration admission proof outside assigned cohort')
            if state.get('fresh_native_budget') is not None and directory in attempt_numbers:
                fresh_budget.reconcile(state['fresh_native_budget'], attempt_numbers[directory], value,
                                       cleanup_verified=True)
            checked += 1
            attempted.update(value.get('attempted_sequence_hashes', []))
            rejected.update(key for key, reason in value.get('terminal_native_outcomes', {}).items()
                            if reason in REJECTION_REASONS and key in assigned and isinstance(key, str)
                            and len(key) == 64 and all(c in '0123456789abcdef' for c in key))
            # Old novelty is only a source of validated native gate evidence;
            # its scientific verdict is never promoted to the new identity.
            for row in value.get('candidates', []):
                try:
                    rows.extend(valid_rows({**value, 'candidates': [row]}, challenge))
                except (ValueError, TypeError, KeyError, AttributeError):
                    unusable += 1
            progress = read(directory / 'native/native-confirmations.json', {})
            if progress.get('challenge') == challenge:
                qualified_classification = (value.get('status') == 'complete'
                    and value.get('native_classification_complete') is True
                    and value.get('novelty_checked') is True
                    and qualified_novelty_result(value, config))
                kept = {row['sequence'] for row in value.get('candidates', [])}
                for record in progress.get('records', []):
                    key = record.get('sequence_hash')
                    record_hash = rules.sequence_hash(record['sequence']) if 'sequence' in record else key
                    if record_hash not in assigned:
                        unusable += 1
                        continue
                    if (record.get('terminal_reason') in REJECTION_REASONS and isinstance(key, str)
                            and len(key) == 64 and all(c in '0123456789abcdef' for c in key)):
                        rejected.add(key)
                    elif 'native_resume' in record:
                        saved = record['native_resume']
                        if (qualified_classification and record.get('sequence') not in kept
                                and len(saved.get('measurements', [])) >= saved.get('required_cpu_observations', 65)):
                            continue
                        records.append(record)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            unusable += 1
    # An old positive cannot erase an unexplained invalidation. Only a bound
    # old novelty operation proves that this specific rejection needs repair.
    rejected.update(set(state.get('invalidated_native_hashes', [])) - novelty_rejected)
    recovered = []
    for row in rows:
        try:
            key = rules.sequence_hash(row['sequence'])
            if key in rejected or row.get('layer2_passed') is not True:
                continue
            rules.validate_evidence(row.get('pr100_evidence'), row['sequence'], challenge, pinned)
            recovered.append(row)
        except (ValueError, TypeError, KeyError, AttributeError):
            unusable += 1
    checkpoints = {}
    for row in records:
        try:
            sequence = row['sequence']; key = rules.sequence_hash(sequence)
            saved = row['native_resume']; identity = saved['identity']; values = saved['measurements']
            required = saved['required_cpu_observations']
            if (key in rejected or saved.get('schema') != 'pr100-native-confirmation-checkpoint-v1'
                    or saved.get('sequence_hash') != key or identity.get('resume_eligible') is not True
                    or identity.get('challenge_sha256') != rules.digest(challenge)
                    or identity.get('filter_sha') != rules.FILTER_SHA
                    or identity.get('requirements_sha256') != rules.digest(requirements)
                    or not isinstance(values, list) or type(required) is not int
                    or not 1 <= len(values) <= required <= 64):
                continue
            for measurement in values:
                if not selection.authoritative_cpu_measurement(measurement):
                    raise ValueError('non-authoritative checkpoint')
                rules.validate_evidence({'challenge': challenge, 'filter_sha': rules.FILTER_SHA,
                    'annotation': row['pr100_annotation'], 'developability': measurement},
                    sequence, challenge, pinned)
            previous = checkpoints.get(sequence)
            if previous is None or len(values) >= len(previous['native_resume']['measurements']):
                checkpoints[sequence] = row
        except (ValueError, TypeError, KeyError, AttributeError):
            unusable += 1
    state['native_rows'] = union_rows([], recovered)
    state['pending_confirmation'] = list(checkpoints.values())
    restored = {rules.sequence_hash(row['sequence']) for row in recovered} | {
        rules.sequence_hash(sequence) for sequence in checkpoints}
    state['invalidated_native_hashes'] = sorted(
        (set(state.get('invalidated_native_hashes', [])) - novelty_rejected) | rejected)
    state['native_attempted_hashes'] = sorted(set(state.get('native_attempted_hashes', [])) | restored | attempted)
    state['pending_native'] = [row for row in pending if rules.sequence_hash(row['sequence'])
                               not in set(state['native_attempted_hashes'])]
    if any(row['sequence'] in state['score_rows'] for row in state['native_rows']):
        state['pending_publication'] = True
    state['qualified_novelty_migration'] = {'identity_sha256': migration_key,
        'runtime_identities': expected, 'checked_artifacts': checked, 'unusable_artifacts_or_rows': unusable,
        'restored_native_rows': len(state['native_rows']), 'retained_checkpoints': len(checkpoints),
        'terminal_or_unexplained_invalidations': len(rejected),
        'old_novelty_rejections_reopened': len(novelty_rejected - rejected), 'completed_at_unix': time.time()}
    return True

def migrate(local, challenge, config, *, ownership_verified=False):
    """Import completed legacy artifacts, never raw journal measurements."""
    from nanobody_opt import pr100_live_controller as c
    state = {'schema': SCHEMA, 'challenge': challenge, 'next_attempt': 1, 'proposal_offset': 0,
             'generated_hashes': [], 'native_attempted_hashes': [], 'score_attempted_hashes': [],
             'pending_native': [], 'pending_confirmation': [], 'native_rows': [], 'score_rows': {}, 'active': None}
    receipt = read(local / 'receipt.json')
    if receipt is None and not ownership_verified:
        for phase in ('score', 'native', 'sequence'):
            if (local / phase).exists() or (local / (phase + '-supervisor.json')).exists():
                state['legacy_uncertain_phase'] = phase
                break
        return state
    receipt = receipt or {}
    state['legacy_receipt_status'] = receipt.get('status')
    sequence = read(local / 'sequence/result.json', {})
    native = read(local / 'native/result.json', {})
    if sequence.get('challenge') == challenge:
        candidates = sequence.get('candidates', [])
        state['proposal_offset'] = sequence.get('next_proposal_offset',
            sequence.get('proposals', config['worker'].get('proposals', 600)))
        state['generated_hashes'] = sequence.get('generated_sequence_hashes',
            [rules.sequence_hash(row['sequence']) for row in candidates])
        # Legacy files cannot identify every admitted interrupted call. Never
        # blindly repeat old native inputs when admission records are absent.
        state['native_attempted_hashes'] = native.get('attempted_sequence_hashes',
            [rules.sequence_hash(row['sequence']) for row in candidates])
        state['pending_native'] = native.get('deferred_candidates', [])
        state['pending_confirmation'] = [r for r in native.get('deferred_confirmations', [])
            if r.get('native_resume', {}).get('identity', {}).get('resume_eligible') is True]
    if native.get('challenge') == challenge:
        state['native_rows'] = valid_rows(native, challenge)
    score_dir = local / 'score'
    report = read(score_dir / 'score/report.json')
    if report is not None:
        names = ['score/report.json', 'score/scorer_settings.json']
        names += ['score/' + group['file'] for group in report.get('groups', [])
                  if group.get('valid') is True and c.re.fullmatch(
                      r'gpu-[0-9]+/length[0-9]+/exact_all[.]json', str(group.get('file', '')))]
        files = {name: (score_dir / name).read_bytes() for name in names if (score_dir / name).is_file()}
        if 'score/scorer_settings.json' in files:
            scores, settings, scorer_hash = c.merge_scores(files, challenge['target'])
            merge_scoring(state, scores, settings, scorer_hash,
                          hashlib.sha256(files['score/scorer_settings.json']).hexdigest())
        state['score_attempted_hashes'] = [rules.sequence_hash(row['sequence']) for row in state['native_rows']]
    if not ownership_verified and receipt.get('status') not in ('published', 'dry_run'):
        phase = receipt.get('stage')
        supervisor = read(local / (str(phase) + '-supervisor.json'), {})
        if phase in ('sequence', 'native', 'score') and not cleanup_verified(supervisor):
            state['legacy_uncertain_phase'] = phase
    return state

def mode_for(state, *, now=None):
    now = time.time() if now is None else now
    if state.get('pending_publication'):
        return 'publish'
    scored, attempted = set(state['score_rows']), set(state['score_attempted_hashes'])
    unscored = any(r['sequence'] not in scored and rules.sequence_hash(r['sequence']) not in attempted
                   for r in state['native_rows'])
    score_ready = now >= state.get('score_retry_after_unix', 0)
    if unscored and score_ready:
        return 'score'
    if state.get('pending_confirmation'):
        return 'native'
    if state.get('fresh_native_budget') and fresh_budget.summary(state['fresh_native_budget'])['remaining'] == 0:
        return 'score_backoff' if unscored else 'fresh_budget_exhausted'
    seen = set(state['native_attempted_hashes'])
    if any(rules.sequence_hash(r['sequence']) not in seen for r in state['pending_native']):
        return 'native'
    return 'sequence'

def required_seconds(mode, config):
    from nanobody_opt import pr100_live_controller as c
    if mode == 'publish':
        return c.PUBLISH_RESERVE_SECONDS + float(config.get('novelty_reserve_seconds', c.NOVELTY_RESERVE_SECONDS))
    required = c.PUBLISH_RESERVE_SECONDS + c.PHASE_FLOORS['score']
    if mode in ('native', 'sequence'):
        required += float(config['worker'].get('native_minimum_call_window_seconds',
                                              config['worker'].get('candidate_budget_seconds', 300)))
        required += float(config['worker'].get('native_cleanup_reserve_seconds',
                            10 if 'native_minimum_call_window_seconds' in config['worker'] else 0))
        required += float(config.get('novelty_reserve_seconds', c.NOVELTY_RESERVE_SECONDS))
        required += c.NATIVE_SETUP_RESERVE_SECONDS
    if mode == 'sequence':
        required += c.PHASE_FLOORS['sequence']
    return required

def run_epoch(config, chain, clock, *, dry_run=False):
    from nanobody_opt import pr100_live_controller as c
    clock = c.clock_state(chain)
    epoch, length = clock['epoch'], clock['epoch_length']
    local = Path(config['state_root']) / str(epoch)
    local.mkdir(parents=True, exist_ok=True)
    challenge = rules.challenge(rules.pinned_config(epoch), epoch,
                                chain.determine_block_hash(epoch * length), length)
    stop_at = c.publication_cutoff(clock, epoch, reserve=config.get('reserve_blocks', c.RESERVE_BLOCKS))
    path = local / 'continuation.json'
    state = read(path)
    if state is None:
        state = migrate(local, challenge, config)
        c.atomic(path, state)
    if state.get('schema') != SCHEMA or state.get('challenge') != challenge:
        raise ValueError('continuation challenge mismatch')
    if state.get('legacy_uncertain_phase'):
        import base64
        for phase in ('sequence', 'native', 'score'):
            host = 'slave' if phase == 'sequence' else 'master'
            answer = c.remote({**config, '_refresh_clock': lambda: c.clock_state(chain)}, host,
                {'operation': 'poll', 'directory': f"{config['remote_root']}/{epoch}/{phase}",
                 'clock': c.clock_state(chain), 'transport_deadline_unix': stop_at})
            if answer.get('running') or not cleanup_verified(answer.get('receipt', {})):
                return {'status': 'recovering_legacy_ownership', 'epoch': epoch}
            c.atomic(local / (phase + '-supervisor.json'), answer['receipt'])
            for name, encoded in answer.get('files', {}).items():
                if name not in ('result.json', 'worker-receipt.json', 'native-confirmations.json',
                                'score/report.json', 'score/scorer_settings.json') and not c.re.fullmatch(
                        r'score/gpu-[0-9]+/length[0-9]+/exact_all[.]json', name):
                    raise ValueError('unexpected recovered file')
                output = local / phase / name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(base64.b64decode(encoded, validate=True))
        state = migrate(local, challenge, config, ownership_verified=True)
        c.atomic(path, state)
    expected = novelty_runtime_ids(config)
    if (expected and state.get('qualified_novelty_migration', {}).get('identity_sha256') != rules.digest(expected)
            and not recover_owned_attempt_for_novelty(state, local, config, chain, stop_at)):
        return {'status': 'qualified_novelty_migration_waiting_for_ownership', 'epoch': epoch}
    if not restore_for_qualified_novelty(state, local, config):
        return {'status': 'qualified_novelty_migration_waiting_for_ownership', 'epoch': epoch}
    c.atomic(path, state)
    budget = fresh_budget.ensure(state, config)
    c.atomic(path, state)
    mode = state['active']['phase'] if state.get('active') else mode_for(state)
    if not c.admissible(clock, epoch, reserve=config.get('reserve_blocks', c.RESERVE_BLOCKS)):
        return {'status': 'epoch_cutoff', 'epoch': epoch}
    if mode in ('fresh_budget_exhausted', 'score_backoff'):
        return {'status': mode, 'epoch': epoch, 'fresh_native_budget': fresh_budget.summary(budget),
                'score_retry_after_unix': state.get('score_retry_after_unix')}
    if not state.get('active') and stop_at - time.time() < required_seconds(mode, config):
        # Do not let useful-work scheduling hide the last viable score window.
        # The persisted resource backoff is capped at that window's start.
        score_due = (mode_for(state, now=time.time()) == 'score')
        if score_due and stop_at - time.time() >= required_seconds('score', config):
            mode = 'score'
        else:
            return {'status': 'insufficient_remaining_window', 'epoch': epoch, 'next_phase': mode}

    if not state.get('active'):
        number = state['next_attempt']; state['next_attempt'] += 1
        state['active'] = {'number': number, 'phase': mode, 'proposal_offset': state['proposal_offset']}
        if mode == 'sequence':
            state['proposal_offset'] += config['worker'].get('proposals', 600)
        c.atomic(path, state)
    active = state['active']
    attempt_dir = local / 'attempts' / f"{active['number']:05d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    worker = active.get('worker') or {**config['worker'], 'proposal_offset': active['proposal_offset'],
              'excluded_sequence_hashes': sorted(set(state['generated_hashes']) | set(state['native_attempted_hashes']))}
    active['worker'] = worker
    c.atomic(path, state)
    attempt_config = {**config, 'worker': worker, '_attempt': active['number'], '_continuation': True}
    receipt = {'epoch': epoch, 'attempt': active['number'], 'started_at_unix': time.time(),
               'source_sha': config['source_sha'], 'publications': [], 'mac_runtime_dependency': False}
    diagnostics = {}
    def save(): c.atomic(path, state)
    def phase(name, payload):
        active['phase'] = name; save()
        native_config = {**attempt_config, 'worker': {**worker,
            'fresh_admission_limit': fresh_budget.planned_fresh_count(budget, active['number'])}} if name == 'native' else attempt_config
        if name == 'native' and 'native_hosts' in config:
            from nanobody_opt.pr100_native_dispatch import run as dispatch
            return dispatch(native_config, challenge, payload, stop_at, chain, attempt_dir,
                            diagnostics=diagnostics)
        phase_config, phase_local = native_config, attempt_dir
        if name == 'novelty':
            operation = hashlib.sha256(payload['archive.csv']).hexdigest()
            identity = novelty_runtime_ids(config).get('master')
            if identity is not None:
                bound_input = json.loads(payload['input.json'])
                bound_input.pop('archive_checked_at_unix', None)
                operation = rules.digest({'archive_sha256': operation, 'novelty_runtime_sha256': identity,
                                          'input_sha256': rules.digest(bound_input)})
            active['novelty_operation'] = operation; save()
            phase_config = {**attempt_config, '_phase_operation': operation}
            phase_local = attempt_dir / 'novelty-rechecks' / operation
            phase_local.mkdir(parents=True, exist_ok=True)
        if name == 'score' and 'native_hosts' in config:
            operation = rules.digest(json.loads(payload['input.json']))
            active['score_operation'] = operation; save()
            phase_config = {**attempt_config, '_phase_operation': operation}
            phase_local = attempt_dir / 'score-batches' / operation
            phase_local.mkdir(parents=True, exist_ok=True)
        return c.run_phase(phase_config, challenge, name, payload, stop_at, chain, phase_local,
                           diagnostics=diagnostics)
    try:
        archive_path = attempt_dir / 'archive.csv'
        if not archive_path.is_file():
            archive_path.write_bytes(c.fetch_archive(challenge['target']))
        archive = archive_path.read_bytes()
        binding = {'challenge': challenge, 'epoch': epoch, 'target': challenge['target'],
                   'archive_sha256': hashlib.sha256(archive).hexdigest(),
                   'archive_checked_at_unix': time.time()}
        if active['phase'] == 'sequence':
            value = json.loads(phase('sequence', {'archive.csv': archive})['result.json'])
            if value.get('challenge') != challenge or value.get('status') != 'complete':
                raise ValueError('incomplete sequence phase')
            state['generated_hashes'] = sorted(set(state['generated_hashes']) |
                set(value.get('generated_sequence_hashes', [rules.sequence_hash(r['sequence']) for r in value['candidates']])))
            next_offset = value.get('next_proposal_offset', state['proposal_offset'])
            if type(next_offset) is not int or next_offset < active['proposal_offset']:
                raise ValueError('invalid sequence continuation offset')
            state['proposal_offset'] = max(state['proposal_offset'], next_offset)
            state['pending_native'] = union_rows(state['pending_native'], value['candidates'])
            active['phase'] = 'native'; save()
        if active['phase'] == 'native':
            seen = set(state['native_attempted_hashes'])
            rows = union_rows(state.get('pending_confirmation', []),
                [r for r in state['pending_native'] if rules.sequence_hash(r['sequence']) not in seen])
            if rows or active.get('native_unresolved'):
                native_input = attempt_dir / 'native-input.json'
                if native_input.is_file():
                    native_payload = native_input.read_bytes()
                    fresh_budget.reserve_saved_input(budget, active['number'], json.loads(native_payload)['candidates'])
                else:
                    rows = fresh_budget.reserve(budget, active['number'], rows)
                    save()  # Reserve shared capacity before any remote launch.
                    native_payload = json.dumps({**binding, 'candidates': rows}).encode()
                    c.atomic(native_input, json.loads(native_payload))
                save()
                files = phase('native', {'archive.csv': archive,
                    'input.json': native_payload})
                value = json.loads(files['result.json'])
                fresh_budget.reconcile(budget, active['number'], value,
                    cleanup_verified=cleanup_verified(read(attempt_dir / 'native-supervisor.json', {})))
                save()
                active['native_unresolved'] = bool(value.get('unresolved_native_hosts'))
                state['blocked_confirmations'] = union_rows(state.get('blocked_confirmations', []),
                                                            value.get('blocked_confirmations', []))
                held = {rules.sequence_hash(r['sequence']) for r in state['blocked_confirmations']}
                progress = json.loads(files.get('native-confirmations.json', b'{}'))
                if progress.get('challenge') == challenge:
                    records = progress.get('records', [])
                    finished = {r['sequence'] for r in value.get('candidates', [])}
                    # A completed, archive-bound novelty rejection cannot improve
                    # by replaying the same fully measured checkpoint. Missing or
                    # partial classification still needs the recovery path.
                    classified = (value.get('status') == 'complete'
                        and value.get('native_classification_complete') is True
                        and value.get('novelty_checked') is True
                        and all(value.get(k) == binding[k] for k in
                                ('challenge', 'epoch', 'target', 'archive_sha256'))
                        and not value.get('unresolved_native_hosts'))
                    classified = classified and qualified_novelty_result(value, config)
                    value['deferred_confirmations'] = [r for r in records if 'native_resume' in r and (
                        (r['sequence'] not in finished and not classified) or len(r['native_resume']['measurements'])
                        < r['native_resume']['required_cpu_observations'])]
                    value.setdefault('terminal_native_outcomes', {}).update({r['sequence_hash']: r['terminal_reason']
                        for r in records if r.get('terminal_reason') in ('developability_red_flags',
                            'psh_threshold', 'aggrescan_threshold', 'developability_failed')})
                    value['attempted_sequence_hashes'] = sorted(set(value.get('attempted_sequence_hashes', [])) |
                        {r.get('sequence_hash', rules.sequence_hash(r['sequence'])) if 'sequence' in r
                         else r['sequence_hash'] for r in records})
                reconcile_native_histories(state, value,
                    {rules.sequence_hash(r['sequence']) for r in json.loads(native_payload)['candidates']})
                attempted = value.get('attempted_sequence_hashes', [rules.sequence_hash(r['sequence']) for r in rows])
                state['native_attempted_hashes'] = sorted(seen | set(attempted))
                state['pending_native'] = [r for r in state['pending_native']
                    if rules.sequence_hash(r['sequence']) not in set(attempted) | held]
                state['pending_confirmation'] = [r for r in value.get('deferred_confirmations', [])
                    if r.get('native_resume', {}).get('identity', {}).get('resume_eligible') is True]
                rejected = set(value.get('terminal_native_outcomes', {}))
                state['invalidated_native_hashes'] = sorted(set(state.get('invalidated_native_hashes', [])) | rejected)
                state['native_rows'] = [r for r in state['native_rows']
                    if rules.sequence_hash(r['sequence']) not in rejected]
                state['native_rows'] = union_rows(state['native_rows'], valid_rows(value, challenge))
            active['phase'] = 'score'; save()
        if active['phase'] == 'score':
            seen = set(state['score_attempted_hashes'])
            rows = [r for r in state['native_rows'] if r['sequence'] not in state['score_rows']
                    and rules.sequence_hash(r['sequence']) not in seen]
            if rows and time.time() >= state.get('score_retry_after_unix', 0):
                files = phase('score', {'input.json': json.dumps({'sequences': [r['sequence'] for r in rows]}).encode()})
                if empty_score_attempt(files, challenge, rows):
                    receipt['score_outcome'] = 'no_scored_candidate'
                    if not json.loads(files['result.json'])['attempted_sequence_hashes']:
                        failures = state.get('score_resource_failures', 0) + 1
                        delay = min(300, 30 * (2 ** min(failures - 1, 4)))
                        report = json.loads(files['score/report.json'])
                        state.update(score_resource_failures=failures,
                                     score_retry_after_unix=min(time.time() + delay,
                                         stop_at - required_seconds('score', config)),
                                     score_resource_reasons=sorted({v['lease_reason'] for v in report.get('shards', [])
                                         if v.get('lease_reason') in ('insufficient_gpu_headroom', 'gpu_not_idle',
                                             'coordination_lock_busy', 'gpu_missing', 'gpu_telemetry_failed')}))
                        receipt['score_retry_after_unix'] = state['score_retry_after_unix']
                else:
                    scores, settings, scorer_hash = c.merge_scores(files, challenge['target'])
                    state.pop('score_retry_after_unix', None)
                    state.pop('score_resource_failures', None)
                    state.pop('score_resource_reasons', None)
                    merge_scoring(state, scores, settings, scorer_hash,
                        hashlib.sha256(files['score/scorer_settings.json']).hexdigest())
                admitted = json.loads(files.get('result.json', b'{}')).get('attempted_sequence_hashes',
                    [rules.sequence_hash(r['sequence']) for r in rows])
                state['score_attempted_hashes'] = sorted(seen | set(admitted))
            active['phase'] = 'publish'; save()
        if state['score_rows']:
            novelty_snapshot = attempt_dir / 'novelty-archive.csv'
            if active['phase'] == 'novelty' and novelty_snapshot.is_file():
                archive = novelty_snapshot.read_bytes()
            else:
                archive = c.fetch_archive(challenge['target'])
            digest = hashlib.sha256(archive).hexdigest()
            rows = [r for r in state['native_rows'] if r['sequence'] in state['score_rows']]
            validity = {**binding, 'archive_sha256': digest, 'status': 'complete', 'novelty_checked': True,
                        'candidates': rows}
            if any(not novelty_proof_reusable(r, digest, config) for r in rows):
                novelty_snapshot.write_bytes(archive)
                value = json.loads(phase('novelty', {'archive.csv': archive,
                    'input.json': json.dumps(validity).encode()})['result.json'])
                if (value.get('challenge') != challenge or value.get('epoch') != epoch
                        or value.get('target') != challenge['target'] or value.get('archive_sha256') != digest
                        or value.get('status') != 'complete' or value.get('novelty_checked') is not True):
                    raise ValueError('novelty recheck binding mismatch')
                if not qualified_novelty_result(value, config, final=True):
                    raise ValueError('unqualified novelty recheck result')
                validity = value
                accepted = {r['sequence']: r for r in valid_rows(value, challenge)}
                rejected = {rules.sequence_hash(r['sequence']) for r in rows if r['sequence'] not in accepted}
                state['invalidated_native_hashes'] = sorted(set(state.get('invalidated_native_hashes', [])) | rejected)
                state['native_rows'] = [accepted.get(r['sequence'], r) for r in state['native_rows']
                                       if r['sequence'] not in state['score_rows'] or r['sequence'] in accepted]
            scores = {'target': challenge['target'], 'per_nanobody_components': {'0': state['score_rows']},
                      'structures': {}, 'n': len(state['score_rows'])}
            active['phase'] = 'publish'; save()
            identity = state['scoring_identity']
            ordered = c.order_candidates(validity['candidates'], allow_marginal=worker.get('allow_marginal', True))
            if not ordered:
                previous = read(config['publication_path'], {})
                invalidated = set(state.get('invalidated_native_hashes', []))
                pool = previous.get('sequence_pool', [])
                if previous.get('epoch') == epoch and pool and all(rules.sequence_hash(s) in invalidated for s in pool):
                    if not c.admissible(c.clock_state(chain), epoch, reserve=config.get('reserve_blocks', c.RESERVE_BLOCKS)):
                        raise TimeoutError('publication cutoff before withdrawal')
                    c.atomic(local / 'invalidated-publication.json', previous)
                    if not dry_run:
                        c.atomic(Path(config['publication_path']), {**previous, 'status': 'invalidated',
                            'sequence_pool': [], 'sequence_pool_size': 0, 'sequence_pool_checkpoints': {},
                            'submissions': [], 'submission_count': 0, 'updated_at_unix': time.time(),
                            'invalidation_reason': 'fresh_validity_recheck'})
                    receipt['publications'].append({'status': 'withdrawn' if not dry_run else 'dry_run_withdrawal'})
            for subset in c.publication_sequence(ordered, allow_marginal=worker.get('allow_marginal', True)):
                if not subset: continue
                try:
                    outcome = c.publish(config, chain, challenge, {**validity, 'candidates': subset}, scores,
                        state['settings'], identity['scorer_sha256'], identity['settings_file_sha256'], local,
                        tiers=c.candidate_tiers(subset), dry_run=dry_run,
                        invalidated_sequence_hashes=state.get('invalidated_native_hashes', []))
                except ValueError as error:
                    if str(error) not in ('cannot degrade an existing valid partial fleet',
                                          'cannot discard an existing valid partial fleet'):
                        raise
                    outcome = {'status': 'upgrade_refused'}
                receipt['publications'].append(outcome)
            receipt['status'] = ('withdrawn' if any(p.get('status') == 'withdrawn' for p in receipt['publications'])
                                 else 'published' if any(p.get('status') == 'published' for p in receipt['publications'])
                                 else 'dry_run' if any(p.get('status') == 'dry_run' for p in receipt['publications'])
                                 else 'unchanged')
        else:
            receipt['status'] = receipt.get('score_outcome', 'no_passing_candidate')
        state.pop('pending_publication', None)
        if active.get('native_unresolved'):
            active['phase'] = 'native'
        else:
            state['active'] = None
        save()
    except Exception as error:
        receipt.update(status='recoverable_failure', stage=active['phase'], error_type=type(error).__name__)
        receipt.update(failure_details(error))
        terminal_dir = (attempt_dir / 'novelty-rechecks' / active['novelty_operation']
                        if active['phase'] == 'novelty' and active.get('novelty_operation') else attempt_dir)
        if active['phase'] == 'score' and active.get('score_operation'):
            terminal_dir = attempt_dir / 'score-batches' / active['score_operation']
        terminal = read(terminal_dir / (active['phase'] + '-supervisor.json'), {})
        if cleanup_verified(terminal):
            if active['phase'] in ('novelty', 'publish'):
                state['pending_publication'] = True
            if active['phase'] == 'native':
                partial = read(attempt_dir / 'native/result.json', {})
                fresh_budget.reconcile(budget, active['number'], partial, cleanup_verified=True)
                assigned = read(attempt_dir / 'native-input.json', {}).get('candidates', [])
                try:
                    reconcile_native_histories(state, partial,
                        {rules.sequence_hash(r['sequence']) for r in assigned})
                    recover_native_checkpoints(state, partial, assigned,
                        worker.get('repeats', 3))
                except (ValueError, TypeError, KeyError):
                    receipt['native_history_recovery_rejected'] = True
                else:
                    rejected = set(partial.get('terminal_native_outcomes', {}))
                    state['invalidated_native_hashes'] = sorted(
                        set(state.get('invalidated_native_hashes', [])) | rejected)
                    state['native_rows'] = [r for r in state['native_rows']
                        if rules.sequence_hash(r['sequence']) not in rejected]
                    if rejected & {rules.sequence_hash(sequence) for sequence in state['score_rows']}:
                        # Reconcile any published evidence before another long
                        # science batch can leave a disproven member in the pool.
                        state['pending_publication'] = True
                    attempted = partial.get('attempted_sequence_hashes')
                    if attempted is None:
                        attempted = [rules.sequence_hash(r['sequence']) for r in state['pending_native']]
                    state['native_attempted_hashes'] = sorted(set(state['native_attempted_hashes']) | set(attempted))
                    state['pending_confirmation'] = [r for r in partial.get('deferred_confirmations', [])
                        if r.get('native_resume', {}).get('identity', {}).get('resume_eligible') is True]
            elif active['phase'] == 'score':
                partial = read(terminal_dir / 'score/result.json', {})
                admitted = partial.get('attempted_sequence_hashes',
                    [rules.sequence_hash(r['sequence']) for r in state['native_rows']])
                state['score_attempted_hashes'] = sorted(set(state['score_attempted_hashes']) | set(admitted))
            if active.get('native_unresolved'):
                # Cleaning up a downstream job cannot release a different
                # host's still-owned native batch.
                active['phase'] = 'native'
            else:
                state['active'] = None
        save()
    receipt.update(phase_diagnostics=diagnostics, publication_cutoff_unix=stop_at,
                   wall_seconds=time.time() - receipt['started_at_unix'],
                   continuation_pending=True, cumulative_scored_candidates=len(state['score_rows']))
    receipt['fresh_native_budget'] = fresh_budget.summary(budget)
    c.atomic(attempt_dir / 'receipt.json', receipt)
    c.atomic(local / 'receipt.json', receipt)
    if 'native' in diagnostics:
        c.start_native_journal_collection(attempt_config, epoch, attempt_dir, stop_at)
    return receipt
