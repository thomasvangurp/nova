"""Durable same-epoch attempts. Journals remain audit-only, never reusable evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from nanobody_opt import rules

SCHEMA = 'pr100-epoch-continuation-v1'

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

def mode_for(state):
    if state.get('pending_publication'):
        return 'publish'
    scored, attempted = set(state['score_rows']), set(state['score_attempted_hashes'])
    if any(r['sequence'] not in scored and rules.sequence_hash(r['sequence']) not in attempted
           for r in state['native_rows']):
        return 'score'
    if state.get('pending_confirmation'):
        return 'native'
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
    mode = state['active']['phase'] if state.get('active') else mode_for(state)
    if not c.admissible(clock, epoch, reserve=config.get('reserve_blocks', c.RESERVE_BLOCKS)):
        return {'status': 'epoch_cutoff', 'epoch': epoch}
    if not state.get('active') and stop_at - time.time() < required_seconds(mode, config):
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
        if name == 'native' and 'native_hosts' in config:
            from nanobody_opt.pr100_native_dispatch import run as dispatch
            return dispatch(attempt_config, challenge, payload, stop_at, chain, attempt_dir,
                            diagnostics=diagnostics)
        phase_config, phase_local = attempt_config, attempt_dir
        if name == 'novelty':
            operation = hashlib.sha256(payload['archive.csv']).hexdigest()
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
                else:
                    native_payload = json.dumps({**binding, 'candidates': rows}).encode()
                    c.atomic(native_input, json.loads(native_payload))
                files = phase('native', {'archive.csv': archive,
                    'input.json': native_payload})
                value = json.loads(files['result.json'])
                active['native_unresolved'] = bool(value.get('unresolved_native_hosts'))
                state['blocked_confirmations'] = union_rows(state.get('blocked_confirmations', []),
                                                            value.get('blocked_confirmations', []))
                held = {rules.sequence_hash(r['sequence']) for r in state['blocked_confirmations']}
                progress = json.loads(files.get('native-confirmations.json', b'{}'))
                if progress.get('challenge') == challenge:
                    records = progress.get('records', [])
                    finished = {r['sequence'] for r in value.get('candidates', [])}
                    value['deferred_confirmations'] = [r for r in records if 'native_resume' in r and (
                        r['sequence'] not in finished or len(r['native_resume']['measurements'])
                        < r['native_resume']['required_cpu_observations'])]
                    value.setdefault('terminal_native_outcomes', {}).update({r['sequence_hash']: r['terminal_reason']
                        for r in records if r.get('terminal_reason') in ('developability_red_flags',
                            'psh_threshold', 'aggrescan_threshold', 'developability_failed')})
                    value['attempted_sequence_hashes'] = sorted(set(value.get('attempted_sequence_hashes', [])) |
                        {r.get('sequence_hash', rules.sequence_hash(r['sequence'])) if 'sequence' in r
                         else r['sequence_hash'] for r in records})
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
            if rows:
                files = phase('score', {'input.json': json.dumps({'sequences': [r['sequence'] for r in rows]}).encode()})
                scores, settings, scorer_hash = c.merge_scores(files, challenge['target'])
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
            if any(r.get('_novelty_archive_sha256') != digest for r in rows):
                novelty_snapshot.write_bytes(archive)
                value = json.loads(phase('novelty', {'archive.csv': archive,
                    'input.json': json.dumps(validity).encode()})['result.json'])
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
            receipt['status'] = 'no_passing_candidate'
        state.pop('pending_publication', None)
        if active.get('native_unresolved'):
            active['phase'] = 'native'
        else:
            state['active'] = None
        save()
    except Exception as error:
        receipt.update(status='recoverable_failure', stage=active['phase'], error_type=type(error).__name__)
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
            state['active'] = None
        save()
    receipt.update(phase_diagnostics=diagnostics, publication_cutoff_unix=stop_at,
                   wall_seconds=time.time() - receipt['started_at_unix'],
                   continuation_pending=True, cumulative_scored_candidates=len(state['score_rows']))
    c.atomic(attempt_dir / 'receipt.json', receipt)
    c.atomic(local / 'receipt.json', receipt)
    if 'native' in diagnostics:
        c.start_native_journal_collection(attempt_config, epoch, attempt_dir, stop_at)
    return receipt
