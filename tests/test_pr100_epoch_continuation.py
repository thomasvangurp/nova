"""Recorded native evidence exercises real continuation/export/Pi boundaries."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from nanobody_opt import pr100_live_controller as c, pr100_epoch_continuation as continuation
from nanobody_opt import pr100_live_worker as worker, pr100_filter, rules
from tests.test_pr100_live_handoff_contract import replay
from tests.test_pr100_native_screen_integration import profile, accelerated, CHALLENGE

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

def score_files(replay, rows):
    group = {'target': replay.challenge['target'], 'structures': {},
             'per_nanobody_components': {'0': {r['sequence']: r['metrics'] for r in rows}}}
    return {'score/report.json': json.dumps({'registered_scorer_sha256': replay.data['settings']['scorer_sha256'],
        'groups': [{'valid': True, 'file': 'gpu-0/length125/exact_all.json'}]}).encode(),
        'score/scorer_settings.json': json.dumps(replay.data['settings']).encode(),
        'score/gpu-0/length125/exact_all.json': json.dumps(group).encode()}

@pytest.fixture
def legacy(replay, monkeypatch):
    ch = replay.challenge
    class Chain:
        block = ch['epoch_start_block'] + 10
        def get_current_block(self): return self.block
        def tempo(self, _): return ch['epoch_length'] - 1
        def determine_block_hash(self, _): return ch['epoch_start_block_hash']
    config = {'state_root': str(replay.directory / 'state'), 'remote_root': '/remote',
              'source_sha': 'a' * 40, 'publication_path': str(replay.directory / 'published.json'),
              'worker': {'native_tools': '/tools', 'proposals': 600, 'repeats': 3, 'allow_marginal': True}}
    local = Path(config['state_root']) / str(ch['epoch'])
    rows = replay.validity['candidates']; native = {**replay.validity, 'candidates': rows[:1],
        'attempted_sequence_hashes': [rules.sequence_hash(rows[0]['sequence'])], 'deferred_candidates': []}
    write(local / 'sequence/result.json', {**native, 'proposals': 600})
    write(local / 'native/result.json', native)
    write(local / 'receipt.json', {'status': 'published', 'epoch': ch['epoch']})
    files = score_files(replay, [r for r in replay.data['rows'] if r['sequence'] == rows[0]['sequence']])
    for name, value in files.items():
        path = local / 'score' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
    monkeypatch.setattr(c, 'fetch_archive', lambda _: replay.archive)
    monkeypatch.setattr(c, 'start_native_journal_collection', lambda *a: None)
    scores, settings, sha = c.merge_scores(files, ch['target'])
    c.publish(config, Chain(), ch, native, scores, settings, sha,
              hashlib.sha256(files['score/scorer_settings.json']).hexdigest(), local)
    return SimpleNamespace(replay=replay, config=config, local=local, chain=Chain(), rows=rows)

def test_migration_validates_recorded_native_and_keeps_raw_scoring(legacy):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    assert state['proposal_offset'] == 600
    assert len(state['native_rows']) == len(state['score_rows']) == 1
    assert state['score_rows'][x.rows[0]['sequence']] == next(r['metrics'] for r in x.replay.data['rows']
                                                          if r['sequence'] == x.rows[0]['sequence'])

def test_legacy_fleet_continues_fresh_and_reranks_union_without_old_rescore(legacy, monkeypatch):
    x = legacy; calls = []
    def phase(cfg, challenge, name, files, stop, chain, local, **kw):
        calls.append((name, cfg['_attempt']))
        if name == 'sequence':
            assert cfg['worker']['proposal_offset'] == 600
            assert rules.sequence_hash(x.rows[0]['sequence']) in cfg['worker']['excluded_sequence_hashes']
            return {'result.json': json.dumps({**x.replay.validity, 'candidates': [x.rows[1]]}).encode()}
        if name == 'native':
            assert [r['sequence'] for r in json.loads(files['input.json'])['candidates']] == [x.rows[1]['sequence']]
            return {'result.json': json.dumps({**x.replay.validity, 'candidates': [x.rows[1]],
                'fresh_admitted_sequence_hashes': [rules.sequence_hash(x.rows[1]['sequence'])],
                'attempted_sequence_hashes': [rules.sequence_hash(x.rows[1]['sequence'])]}).encode()}
        assert name == 'score' and json.loads(files['input.json'])['sequences'] == [x.rows[1]['sequence']]
        return score_files(x.replay, [r for r in x.replay.data['rows'] if r['sequence'] == x.rows[1]['sequence']])
    monkeypatch.setattr(c, 'run_phase', phase)
    outcome = c.run_epoch(x.config, x.chain, c.clock_state(x.chain))
    assert outcome['status'] == 'published'
    fleet = json.loads(Path(x.config['publication_path']).read_text())
    assert set(fleet['sequence_pool']) == {r['sequence'] for r in x.rows}
    assert any(v['selection']['score'] > 10 for v in fleet['sequence_pool_checkpoints'].values())
    assert calls == [('sequence', 1), ('native', 1), ('score', 1)]

def test_late_window_scores_pending_valid_rows_without_start_guard(legacy, monkeypatch):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state['native_rows'].append({**x.rows[1], '_novelty_archive_sha256': x.replay.validity['archive_sha256']})
    write(x.local / 'continuation.json', state)
    x.chain.block = x.replay.challenge['epoch_start_block'] + 361 - 80
    def phase(cfg, challenge, name, files, *_a, **kw):
        assert name == 'score'
        return score_files(x.replay, [r for r in x.replay.data['rows'] if r['sequence'] == x.rows[1]['sequence']])
    monkeypatch.setattr(c, 'run_phase', phase)
    outcome = c.run_epoch(x.config, x.chain, c.clock_state(x.chain))
    assert outcome['status'] == 'published' and outcome['cumulative_scored_candidates'] == 2

def test_failed_continuation_preserves_publication_and_ownership(legacy, monkeypatch):
    x = legacy; before = Path(x.config['publication_path']).read_bytes()
    monkeypatch.setattr(c, 'run_phase', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('transport')))
    one = c.run_epoch(x.config, x.chain, c.clock_state(x.chain))
    two = c.run_epoch(x.config, x.chain, c.clock_state(x.chain))
    assert one['status'] == two['status'] == 'recoverable_failure'
    assert one['attempt'] == two['attempt'] == 1
    assert Path(x.config['publication_path']).read_bytes() == before

@pytest.mark.parametrize('prior_count', [1, 2, 3])
def test_cpu_checkpoint_finishes_without_restarting_work(tmp_path, monkeypatch, prior_count):
    sequence = 'AAAA'
    identity = {'resume_eligible': True, 'challenge_sha256': rules.digest(CHALLENGE), 'model_weights': {'1': 'a'*64}}
    monkeypatch.setattr(worker, 'native_resume_identity', lambda *a: identity)
    monkeypatch.setattr(pr100_filter, 'helper', lambda _: SimpleNamespace(tool_paths=None))
    monkeypatch.setattr(rules, 'validate_evidence', lambda *a: None)
    row = {'sequence': sequence, 'pr100_annotation': {}, 'native_resume': {
        'schema': 'pr100-native-confirmation-checkpoint-v1', 'sequence_hash': rules.sequence_hash(sequence),
        'identity': identity, 'measurements': [profile(sequence) for _ in range(prior_count)],
        'required_cpu_observations': 3, 'accelerated_screen': accelerated(sequence)}}
    write(tmp_path / 'input.json', {'challenge': CHALLENGE, 'epoch': CHALLENGE['epoch'], 'target': CHALLENGE['target'],
        'archive_sha256': 'a'*64, 'archive_checked_at_unix': 1, 'candidates': [row]})
    calls = []
    def measure(sequences, *_a, **kw):
        calls.extend(sequences)
        return [profile(s) for s in sequences], 0., 1.
    result = worker.native_phase({'challenge': CHALLENGE, 'native_tools': '/tools', 'workers': 1,
        'repeats': 3, 'native_screen': {'enabled': True}}, tmp_path, measure=measure,
        screen=lambda *a, **kw: pytest.fail('resumed evidence was screened again'))
    assert len(calls) == 3 - prior_count
    assert result['fresh_evaluations'] == 0 and result['repeat_evaluations'] == 3 - prior_count
    assert result['carried_cpu_observations'] == prior_count
    assert len(result['candidates'][0]['measurements']) == 3

def test_incomplete_native_identity_cannot_qualify_resume(tmp_path):
    value = worker.native_resume_identity({'challenge': CHALLENGE, 'native_tools': '/inert-missing'},
                                          tmp_path, {'psh_strict_rejection_threshold': 89})
    assert value['resume_eligible'] is False

def test_scoring_protocol_drift_never_mixes_raw_rows():
    state = {'challenge': CHALLENGE, 'score_rows': {}}
    score = {'target': CHALLENGE['target'], 'per_nanobody_components': {'0': {'A': {}}}}
    continuation.merge_scoring(state, score, {'steps': 35}, 'a'*64, 'b'*64)
    with pytest.raises(ValueError):
        continuation.merge_scoring(state, score, {'steps': 200}, 'a'*64, 'c'*64)

def test_confirmation_upsert_improves_prior_row():
    old = {'sequence': 'A', 'tier': 'B', 'measurements': [{}]}
    new = {'sequence': 'A', 'tier': 'A', 'measurements': [{}, {}, {}]}
    assert continuation.union_rows([old], [new]) == [new]

@pytest.mark.parametrize('still_novel', [True, False])
def test_archive_change_rechecks_novelty_without_native_or_score_replay(legacy, monkeypatch, still_novel):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state['active'] = {'number': 1, 'phase': 'publish', 'proposal_offset': 600}
    write(x.local / 'continuation.json', state)
    fresh = x.replay.archive + b'\n'; calls = []
    monkeypatch.setattr(c, 'fetch_archive', lambda _: fresh)
    def phase(cfg, challenge, name, files, *_a, **kw):
        calls.append(name); assert name == 'novelty'
        assert cfg['_phase_operation'] == hashlib.sha256(fresh).hexdigest()
        value = json.loads(files['input.json'])
        assert value['archive_sha256'] == hashlib.sha256(fresh).hexdigest()
        if not still_novel:
            value['candidates'] = []
        return {'result.json': json.dumps(value).encode()}
    monkeypatch.setattr(c, 'run_phase', phase)
    result = c.run_epoch(x.config, x.chain, c.clock_state(x.chain))
    assert result['status'] == ('published' if still_novel else 'withdrawn') and calls == ['novelty']
    if not still_novel:
        assert json.loads(Path(x.config['publication_path']).read_text())['sequence_pool'] == []

def test_score_shard_limit_defers_unstarted_tail_and_unavailable_lease(tmp_path):
    sequences = [str(i) for i in range(257)]
    job = {'output_directory': str(tmp_path / 'out'), 'release': '/release', 'wall_seconds': 400,
           'gpus': [0, 1], 'scorer_settings': '/settings', 'source_sha': 'a'*40,
           'gpu_lock_directory': '/locks', 'scorer': '/scorer'}
    jobs = worker.score_jobs(job, tmp_path, sequences)
    assert sum(j['candidate_count'] for j in jobs) == 256
    planned = {h for j in jobs for h in j['candidate_hashes']}
    assert len({rules.sequence_hash(s) for s in sequences} - planned) == 1
    for item in jobs:
        write(Path(item['receipt']), {'status': 'unavailable'})
    report = worker.collect_scores(Path(job['output_directory']), jobs)
    assert report['attempted_sequence_hashes'] == []

def test_terminal_novelty_failure_retries_publication_without_new_scientific_calls(legacy, monkeypatch):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state['active'] = {'number': 1, 'phase': 'publish', 'proposal_offset': 600}
    state['next_attempt'] = 2
    write(x.local / 'continuation.json', state)
    monkeypatch.setattr(c, 'fetch_archive', lambda _: x.replay.archive + b'\n')
    calls = []
    def phase(cfg, challenge, name, files, stop, chain, local, **kw):
        calls.append((name, cfg['_attempt'])); assert name == 'novelty'
        if len(calls) == 1:
            write(local / 'novelty-supervisor.json', {'reason': 'nonzero_exit',
                'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}})
            raise RuntimeError('bounded novelty failure')
        return {'result.json': files['input.json']}
    monkeypatch.setattr(c, 'run_phase', phase)
    assert c.run_epoch(x.config, x.chain, c.clock_state(x.chain))['status'] == 'recoverable_failure'
    pending = continuation.read(x.local / 'continuation.json')
    assert pending['active'] is None and continuation.mode_for(pending) == 'publish'
    assert c.run_epoch(x.config, x.chain, c.clock_state(x.chain))['status'] == 'published'
    assert calls == [('novelty', 1), ('novelty', 2)]


def test_running_attempt_keeps_frozen_parent_bank_when_pointer_changes(legacy, monkeypatch):
    from nanobody_opt import pr100_parent_bank
    x = legacy; snapshots = []; dispatched = []
    first = {'status': 'fresh', 'references': '/bank/first.json',
             'reference_bank_sha256': '1' * 64}
    def snapshot(*args):
        snapshots.append(1)
        return first if len(snapshots) == 1 else {'references': '/bank/new.json',
                                                  'reference_bank_sha256': '2' * 64}
    monkeypatch.setattr(pr100_parent_bank, 'snapshot', snapshot)
    def phase(cfg, *args, **kwargs):
        dispatched.append(dict(cfg['worker']['parent_bank_snapshot']))
        raise RuntimeError('fixture interrupted before remote dispatch')
    monkeypatch.setattr(c, 'run_phase', phase)
    for _ in range(2):
        assert c.run_epoch(x.config, x.chain, c.clock_state(x.chain))['status'] == 'recoverable_failure'
    assert snapshots == [1]
    assert dispatched == [first, first]


def test_frozen_fitness_field_binds_the_epoch_entry_field(monkeypatch):
    challenge = {'target': 'Q9NZQ7', 'epoch': 26030}
    identity = {'scorer_sha256': 'a' * 64, 'settings_sha256': 'b' * 64,
                'settings_file_sha256': 'c' * 64}
    metrics = {'design_iiptm': 'max'}
    field = {'target': 'Q9NZQ7', 'epochs': [26029], 'rows': [{'design_iiptm': 0.5}]}
    state = {'fitness_field': field}
    frozen = continuation.frozen_fitness_field(state, challenge, identity, metrics=metrics)
    assert frozen is not None and frozen.reference_epochs == (26029,)
    assert frozen.scorer_sha256 == 'a' * 64 and frozen.settings_file_sha256 == 'c' * 64


def test_frozen_fitness_field_falls_back_when_field_unavailable_or_unfreezable(monkeypatch):
    identity = {'scorer_sha256': 'a' * 64, 'settings_sha256': 'b' * 64,
                'settings_file_sha256': 'c' * 64}
    assert continuation.frozen_fitness_field({'fitness_field': None},
        {'target': 'Q9NZQ7', 'epoch': 26030}, identity) is None
    assert continuation.frozen_fitness_field({},
        {'target': 'Q9NZQ7', 'epoch': 26030}, identity) is None
    stale = {'target': 'Q9NZQ7', 'epochs': [26000], 'rows': [{'design_iiptm': 0.5}]}
    assert continuation.frozen_fitness_field({'fitness_field': stale},
        {'target': 'Q9NZQ7', 'epoch': 26030}, identity, metrics={'design_iiptm': 'max'}) is None


def test_epoch_entry_freezes_fitness_field_once_across_restarts(legacy, monkeypatch):
    from nanobody_opt import pr100_parent_bank
    x = legacy; collections = []
    field = {'target': x.replay.challenge['target'], 'epochs': [x.replay.challenge['epoch'] - 1],
             'rows': [{'design_iiptm': 0.5}]}
    monkeypatch.setattr(pr100_parent_bank, 'fitness_field',
                        lambda config, challenge, metrics: collections.append(1) or field)
    def phase(cfg, *args, **kwargs):
        raise RuntimeError('fixture interrupted before remote dispatch')
    monkeypatch.setattr(c, 'run_phase', phase)
    for _ in range(2):
        assert c.run_epoch(x.config, x.chain, c.clock_state(x.chain))['status'] == 'recoverable_failure'
    assert collections == [1]  # collected once at epoch entry, never re-collected on restart
    saved = json.loads((x.local / 'continuation.json').read_text())
    assert saved['fitness_field'] == field


def test_run_epoch_passes_frozen_field_into_publication(legacy, monkeypatch):
    from nanobody_opt import pr100_parent_bank
    from nanobody_opt.pr100_live_trial_export import METRICS
    x = legacy
    state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state['active'] = {'number': 1, 'phase': 'publish', 'proposal_offset': 600}
    state['next_attempt'] = 2
    write(x.local / 'continuation.json', state)
    field = {'target': x.replay.challenge['target'],
             'epochs': [x.replay.challenge['epoch'] - 1],
             'rows': [{name: 0.5 for name in METRICS}]}
    monkeypatch.setattr(pr100_parent_bank, 'fitness_field', lambda *a: field)
    seen = {}
    real_publish = c.publish
    def publish(config, chain, challenge, validity, scores, settings, scorer_hash,
                settings_hash, local, **kwargs):
        seen['reference_field'] = kwargs.get('reference_field')
        return real_publish(config, chain, challenge, validity, scores, settings,
                            scorer_hash, settings_hash, local, **kwargs)
    monkeypatch.setattr(c, 'publish', publish)
    outcome = c.run_epoch(x.config, x.chain, c.clock_state(x.chain))
    assert outcome['status'] == 'published'
    assert seen['reference_field'] is not None and seen['reference_field'].row_count == 1
    assert seen['reference_field'].selection({name: 0.5 for name in METRICS})['method'] \
        == 'one_pass_frozen_public_field_dense_rank_sum'
