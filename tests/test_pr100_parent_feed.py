"""Real public/annotation fixtures; no synthetic hardware qualification claims."""
import base64
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from nanobody_opt import pr100_parent_bank as bank, pr100_parent_feed as feed
from nanobody_opt import pr100_live_controller as controller, rules
from nanobody_opt import pr100_fitness_reference as reference
from nanobody_opt.pr100_live_trial_export import METRICS as FITNESS_METRICS
from test_pr100_live_controller import config as controller_config

FIXTURE = Path(__file__).parent / 'fixtures/pr100/parent_feed'
PUBLIC = json.loads((FIXTURE / 'public_board_26028.json').read_text())
ANNOTATED = json.loads((FIXTURE / 'qualified_annotations.json').read_text())
TARGET = 'Q9NZQ7'


@pytest.fixture
def case(tmp_path):
    config = controller_config(tmp_path)
    original = copy.deepcopy(ANNOTATED['original_bank'])
    original_path = tmp_path / 'original.json'; bank.immutable(original_path, bank.raw(original))
    config['hosts']['slave'].update(references=str(original_path),
        reference_bank_sha256=bank.sha(bank.raw(original)))
    config['parent_bank_feed'] = {'enabled': True, 'cpu_affinity': [64, 65, 66, 67],
        'remote_directory': str(tmp_path / 'remote-feed')}
    cfg = bank.settings(config); root = Path(cfg['state_directory'])
    directory = root / TARGET; directory.mkdir(parents=True)
    now = time.time()
    bank.atomic(Path(config['state_root']) / 'heartbeat.json', {'epoch': 26030, 'updated_at_unix': now})
    return SimpleNamespace(config=config, cfg=cfg, root=root, directory=directory,
        now=now, challenge={'epoch': 26030, 'epoch_length': 361, 'target': TARGET}, original=original)


def fetch(epoch, target, before_epoch):
    if epoch == 26028:
        return {'epoch': epoch, 'status': 'ok',
                'board': bank.public_epoch(copy.deepcopy(PUBLIC), target, epoch, before_epoch)}
    return {'epoch': epoch, 'status': 'unavailable', 'http_status': 404}


def annotate_recorded(rows):
    for row in rows:
        annotation = ANNOTATED['annotations'].get(row['sequence_hash'])
        yield {'sequence_hash': row['sequence_hash'], 'annotation': copy.deepcopy(annotation),
               'reason': 'passed' if annotation else 'fixture_unqualified'}


def completed(case, request):
    job = request['job']
    original = json.loads(bank.read(job['original_bank_path']))
    public = json.loads(base64.b64decode(request['public']))
    result = feed.qualify_rows(job, original, public, annotate_recorded)
    result['status'] = 'complete'
    result['cleanup'] = {'remaining_descendants': 0, 'fixed_point_reached': True}
    result['qualification'].update(status='complete', cleanup=copy.deepcopy(result['cleanup']),
        completed_at_unix=case.now, elapsed_seconds=1., cpu_affinity=[64, 65, 66, 67])
    bank.immutable(Path(job['remote_directory']) / 'banks' /
                   ('bank-' + result['qualification']['bank_sha256'] + '.json'), bank.raw(result['bank']))
    return result


def dispatch(case):
    calls = []
    def call(host, release, request):
        calls.append(copy.deepcopy(request))
        return {'running': True, 'request_sha256': request['request_sha256']}
    out = feed.tick(case.config, now=case.now, fetch=fetch, call=call)
    assert out['status'] == 'qualification_dispatched'
    pending = json.loads(bank.read(case.root / 'pending.json'))
    return pending, completed(case, pending['request'])


def promote(case):
    pending, result = dispatch(case)
    out = feed.tick(case.config, now=case.now + 1, fetch=fetch,
        call=lambda *a: {'running': False, 'result': result})
    assert out['status'] == 'promoted'
    pointer = json.loads(bank.read(case.directory / 'current.json'))
    return pending, result, pointer


def test_real_board_normalization_covers_all_coldkeys_and_truthful_single_pass_percentiles():
    value = bank.public_epoch(PUBLIC, TARGET, 26028, 26030)
    assert len(value['rows']) == 20
    assert len({r['coldkey'] for r in value['rows']}) > 1
    winner = min(value['rows'], key=lambda r: r['metrics']['final_score'])
    assert winner['metrics']['final_score'] == 30
    assert winner['quality_evidence'][0]['value'] == 0
    assert all(r['quality_evidence'][0]['passes'] == 1 and
        r['quality_evidence'][0]['field_size'] == 20 for r in value['rows'])
    assert all(r['public_observations'][0]['coldkey'] == r['coldkey'] for r in value['rows'])
    provenance = json.loads((FIXTURE / 'public_board_26028_provenance.json').read_text())
    assert bank.sha((FIXTURE / 'public_board_26028.json').read_bytes()) == provenance['allowlisted_fixture_sha256']


@pytest.mark.parametrize('fault', ['future', 'failed', 'malformed', 'truncated', 'paginated'])
def test_unusable_public_board_cannot_be_qualified(fault):
    payload = copy.deepcopy(PUBLIC); epoch = 26028
    if fault == 'future': epoch = 26030
    elif fault == 'failed': payload['success'] = False
    elif fault == 'malformed': payload['data'][0]['nanobodies'] = 'bad'
    elif fault == 'truncated': payload['meta'] = {'total': 10001}
    else: payload['meta'] = {'has_more': True}
    with pytest.raises(ValueError): bank.public_epoch(payload, TARGET, epoch, 26030)


def test_incremental_ledger_bounds_calls_and_retries_missing_epochs_without_dropping_cache(case):
    ledger = {str(e): {'retry_after_unix': case.now + 1000} for e in range(rules.ACTIVATION_EPOCH, 26030)}
    ledger['26028'] = {}; ledger['26029'] = {}
    public, count = feed.public_refresh(case.directory, ledger, 26030, TARGET, now=case.now, fetch=fetch)
    assert count == 2 and len(public['rows']) == 20
    again, count = feed.public_refresh(case.directory, ledger, 26030, TARGET, now=case.now + 1, fetch=fetch)
    assert count == 0 and again == public
    old = ledger['26028']['sha256']
    failed = lambda e, *a: {'epoch': e, 'status': 'unavailable', 'http_status': 503}
    retained, count = feed.public_refresh(case.directory, ledger, 26030, TARGET,
                                         now=case.now + 301, fetch=failed)
    assert count == 2 and ledger['26028']['sha256'] == old and retained['rows'] == public['rows']
    assert ledger['26029']['failures'] == 2
    assert len(bank.due_epochs({}, 26030, case.now, 8)) == 8


def test_real_annotations_preserve_old_cells_provenance_and_owner_lineage(case):
    pending, result = dispatch(case)
    rows = {r['sequence_hash']: r for r in result['bank']['rows']}
    assert len(rows) >= 20
    for old in case.original['rows']:
        current = rows[old['sequence_hash']]
        assert current['cells'] == old['cells']
        for key in ('source', 'developability', 'pr100_evidence', 'novelty'):
            if key in old: assert current[key] == old[key]
    new = next(row for row in rows.values() if row['sequence_hash'] not in
               {r['sequence_hash'] for r in case.original['rows']})
    assert new['public_observations'][0]['coldkey']
    assert new['native_qualified'] is False
    assert not {'developability', 'pr100_evidence', 'novelty'} & set(new)
    assert result['bank']['native_qualified'] is result['bank']['novelty_qualified'] is False
    assert result['qualification']['publication_authorized'] is False


def test_updated_same_epoch_merit_preserves_previous_field_and_owner_history(case):
    a = bank.public_epoch(PUBLIC, TARGET, 26028, 26030)['rows'][0]
    b = copy.deepcopy(a); b['quality_evidence'][0].update(value=.25, field_sha256='f' * 64)
    b['public_observations'][0].update(field_sha256='f' * 64)
    rows = bank.merge_rows(case.original['rows'], [a], target=TARGET, before_epoch=26030)
    # Add real annotation cells, as a qualified previous bank would contain.
    row = next(r for r in rows if r['sequence_hash'] == a['sequence_hash'])
    annotation = ANNOTATED['annotations'][a['sequence_hash']]
    from tools.de_novo_bank import allowed_v_genes
    row['cells'] = [f'{g}:{annotation["cdr3_length"]}' for g in allowed_v_genes(annotation)]
    newer = bank.merge_rows(rows, [b], target=TARGET, before_epoch=26030)
    saved = next(r for r in newer if r['sequence_hash'] == a['sequence_hash'])
    assert saved['quality_evidence'][-1]['value'] == .25
    assert a['quality_evidence'][0] in saved['quality_evidence_history']
    assert len(saved['public_observations']) == 2


@pytest.mark.parametrize('fault', ['missing', 'transient', 'old_gate_failure', 'old_cell_change'])
def test_partial_annotation_never_promotes_or_loses_old_coverage(case, fault):
    pending, _ = dispatch(case); job = pending['request']['job']
    public = json.loads(base64.b64decode(pending['request']['public']))
    def annotate(rows):
        values = list(annotate_recorded(rows))
        if fault == 'missing': return values[:-1]
        if fault == 'transient': values[-1]['reason'] = 'annotation_error'
        elif fault == 'old_gate_failure': values[0]['reason'] = 'human_v_identity'
        else: values[0]['annotation']['cdr3_length'] += 1
        return values
    with pytest.raises(ValueError): feed.qualify_rows(job, case.original, public, annotate)
    assert not (case.directory / 'current.json').exists()


@pytest.mark.parametrize('fault', ['source', 'runtime', 'cleanup', 'bank_bytes', 'future', 'time'])
def test_changed_pointer_is_observable_and_retains_previous_qualified_bank(case, fault):
    _, _, pointer = promote(case)
    first = bank.snapshot(case.config, case.challenge)
    bad = copy.deepcopy(pointer)
    if fault == 'source': bad['source_sha'] = 'f' * 40
    elif fault == 'runtime': bad['runtime_sha256'] = 'f' * 64
    elif fault == 'future': bad['before_epoch'] = 26031
    elif fault == 'time': bad['qualified_at_unix'] = float('inf')
    elif fault == 'cleanup':
        rp = case.directory / ('qualification-' + bad['qualification_sha256'] + '.json')
        receipt = json.loads(bank.read(rp)); receipt['cleanup']['remaining_descendants'] = 1
        bad['qualification_sha256'] = bank.sha(bank.raw(receipt))
        bank.immutable(case.directory / ('qualification-' + bad['qualification_sha256'] + '.json'), bank.raw(receipt))
    else:
        # Corruption affects both pointer versions; configured fallback remains available.
        (case.directory / ('bank-' + bad['bank_sha256'] + '.json')).write_text('{}')
    if fault == 'time':
        (case.directory / 'current.json').write_text(json.dumps(bad))
    else: bank.atomic(case.directory / 'current.json', bad)
    retained = bank.snapshot(case.config, case.challenge)
    assert retained['refresh_issue']
    assert retained['status'] == ('configured_fallback' if fault == 'bank_bytes' else 'retained_qualified')
    if fault != 'bank_bytes': assert retained['reference_bank_sha256'] == first['reference_bank_sha256']


def test_source_change_reannotates_entire_accumulated_bank_but_cannot_execute_old_snapshot(case):
    _, result, pointer = promote(case); bank.snapshot(case.config, case.challenge)
    newer = copy.deepcopy(case.config); newer['source_sha'] = 'f' * 40
    assert bank.snapshot(newer, case.challenge)['status'] == 'configured_fallback'
    path, digest = bank.reannotation_base(newer, case.challenge)
    assert digest == pointer['bank_sha256'] and path == pointer['remote_bank_path']
    original = json.loads(bank.read(path))
    job = {**bank.expected_identity(newer, TARGET), 'before_epoch': 26030,
           'original_bank_sha256': digest, 'public_records_sha256': 'e' * 64}
    # A later empty/corrected public board does not erase prior bank coverage.
    requalified = feed.qualify_rows(job, original, {'rows': []}, annotate_recorded)
    assert {r['sequence_hash'] for r in requalified['bank']['rows']} == {
        r['sequence_hash'] for r in result['bank']['rows']}
    assert requalified['bank']['source_sha'] == 'f' * 40


def test_last_compatible_fallback_has_explicit_age_bound(case):
    _, _, pointer = promote(case); bank.snapshot(case.config, case.challenge)
    assert bank.snapshot(case.config, {'epoch': 26034, 'target': TARGET})['status'] == 'retained_qualified'
    assert bank.snapshot(case.config, {'epoch': 26055, 'target': TARGET})['status'] == 'configured_fallback'
    # Reannotation input retention is independent of execution approval age.
    assert bank.reannotation_base(case.config, {'epoch': 26055, 'target': TARGET})[1] == pointer['bank_sha256']


def test_clean_promotion_dedupes_unchanged_scientific_input(case):
    promote(case)
    # Mark the already known historical coverage caught up; no request is due.
    ledger = json.loads(bank.read(case.directory / 'public-ledger.json'))
    for e in range(rules.ACTIVATION_EPOCH, 26030):
        ledger.setdefault(str(e), {})['retry_after_unix'] = case.now + 1000
    bank.atomic(case.directory / 'public-ledger.json', ledger)
    out = feed.tick(case.config, now=case.now + 2, fetch=fetch,
                    call=lambda *a: pytest.fail('unchanged bank must not reannotate'))
    assert out['status'] == 'unchanged'
    health = feed.health_summary(case.config)
    assert health['qualified_rows'] >= 20 and health['cell_coverage']
    assert health['last_success_unix'] == case.now


def test_uncertain_owner_is_polled_and_becomes_unhealthy_without_relaunch(case):
    pending, _ = dispatch(case); seen = []
    def call(host, release, request):
        seen.append(request['operation'])
        return {'running': True, 'launch_uncertain': True}
    out = feed.tick(case.config, now=case.now + 400, call=call)
    assert out['status'] == 'qualification_unhealthy' and out['launch_uncertain']
    assert seen == ['poll'] and (case.root / 'pending.json').exists()
    assert out['owner_age_seconds'] == 400


def test_not_started_retries_same_durable_request_not_a_new_owner(case):
    pending, _ = dispatch(case); seen = []
    def call(host, release, request):
        seen.append(request)
        return {'running': False, 'not_started': True} if request['operation'] == 'poll' else {'running': True}
    out = feed.tick(case.config, now=case.now + 1, call=call)
    assert out['status'] == 'qualification_start_retried'
    assert [r['operation'] for r in seen] == ['poll', 'start']
    assert {r['request_sha256'] for r in seen} == {pending['request']['request_sha256']}


def test_dirty_cleanup_keeps_owner_and_never_promotes(case):
    pending, result = dispatch(case); result['cleanup']['remaining_descendants'] = 1
    with pytest.raises(ValueError, match='owner_uncertain'):
        feed.tick(case.config, now=case.now + 1, call=lambda *a: {'running': False, 'result': result})
    assert (case.root / 'pending.json').exists() and not (case.directory / 'current.json').exists()


def test_live_config_source_change_during_poll_does_not_relabel_qualification(case):
    _, result = dispatch(case)
    changed = copy.deepcopy(case.config); changed['source_sha'] = 'f' * 40
    out = feed.tick(case.config, now=case.now + 1, call=lambda *a: {'running': False, 'result': result},
                    config_provider=lambda: changed)
    assert out['status'] == 'retained_previous' and out['reason'] == 'parent_configuration_changed'
    assert not (case.directory / 'current.json').exists() and not (case.root / 'pending.json').exists()


@pytest.mark.parametrize('heartbeat_age,expected', [(0, 'promoted'), (181, 'retained_previous')])
def test_promotion_checks_heartbeat_after_slow_poll(case, monkeypatch, heartbeat_age, expected):
    _, result = dispatch(case)
    wall = [case.now + 1]
    monkeypatch.setattr(feed.time, 'time', lambda: wall[0])
    def poll(*args):
        wall[0] += 39
        bank.atomic(Path(case.config['state_root']) / 'heartbeat.json',
                    {'epoch': 26030, 'updated_at_unix': wall[0] - heartbeat_age})
        return {'running': False, 'result': result}
    out = feed.tick(case.config, call=poll)
    assert out['status'] == expected
    if heartbeat_age:
        assert out['reason'] == 'parent_clock_stale'


def test_pointer_compare_and_swap_refuses_concurrent_replacement(case):
    pending, result = dispatch(case)
    bank.atomic(case.directory / 'current.json', {'independent_new_pointer': True})
    with pytest.raises(ValueError, match='pointer_race'):
        feed.promote(case.directory, pending, result, case.config, case.challenge)
    assert json.loads(bank.read(case.directory / 'current.json')) == {'independent_new_pointer': True}


def test_real_public_board_to_pointer_then_next_job_uses_frozen_hash(case):
    _, _, pointer = promote(case)
    frozen = bank.snapshot(case.config, case.challenge)
    config = {**case.config, 'worker': {**case.config['worker'], 'parent_bank_snapshot': frozen}}
    clock = {'epoch': 26030, 'epoch_length': 361, 'block': 26030 * 361, 'updated_at_unix': time.time()}
    _, request = controller.phase_request(config, case.challenge, 'sequence', {},
                                          time.time() + 1000, clock, cap=60)
    job = json.loads(base64.b64decode(request['files']['job.json']))
    assert job['references'] == pointer['remote_bank_path']
    assert job['reference_bank_sha256'] == pointer['bank_sha256']
    # Advancing/corrupting the live pointer cannot alter the saved worker's inputs.
    bank.atomic(case.directory / 'current.json', {'invalid_new_pointer': True})
    _, repeated = controller.phase_request(config, case.challenge, 'sequence', {},
                                           time.time() + 1000, clock, cap=60)
    assert repeated['files']['job.json'] == request['files']['job.json']


def test_partial_remote_directory_is_never_relaunched(case, monkeypatch):
    pending, _ = dispatch(case); request = pending['request']
    monkeypatch.setattr(feed, 'ROOT', Path(request['job']['release']))
    directory = Path(request['directory']); directory.mkdir(parents=True)
    bank.immutable(directory / 'job.json', bank.raw(request['job']))
    monkeypatch.setattr(feed.subprocess, 'Popen', lambda *a, **k: pytest.fail('uncertain owner restarted'))
    reply = feed.remote_request(request)
    assert reply['running'] is True and reply['launch_uncertain'] is True


def test_pre_spawn_source_failure_is_terminal_clean_and_retryable(case, monkeypatch):
    pending, _ = dispatch(case); request = pending['request']
    monkeypatch.setattr(feed, 'ROOT', Path(request['job']['release']))
    monkeypatch.setattr(feed, 'verify_source', lambda *_: (_ for _ in ()).throw(ValueError('source_drift')))
    monkeypatch.setattr(feed.subprocess, 'Popen', lambda *a, **k: pytest.fail('drifted source started'))
    reply = feed.remote_request(request)
    assert reply['running'] is False and reply['result']['pre_spawn'] is True
    assert reply['result']['cleanup'] == {'remaining_descendants': 0, 'fixed_point_reached': True}


def test_content_addressed_write_never_overwrites_changed_or_partial_bytes(tmp_path):
    path = tmp_path / 'bank.json'; bank.immutable(path, b'complete')
    bank.immutable(path, b'complete')
    with pytest.raises(ValueError, match='immutable_bank_changed'): bank.immutable(path, b'other')
    assert path.read_bytes() == b'complete'
    assert list(tmp_path.iterdir()) == [path]


def test_release_upgrade_empty_public_still_dispatches_accumulated_bank(case, monkeypatch):
    _, _, pointer = promote(case)
    newer = copy.deepcopy(case.config); newer['source_sha'] = 'f' * 40
    monkeypatch.setattr(feed, 'public_refresh', lambda *a, **k: (
        {'schema': 'pr100-public-parent-bundle-v1', 'target': TARGET,
         'before_epoch': 26030, 'sources': {}, 'rows': []}, 0))
    calls = []
    out = feed.tick(newer, now=case.now + 2,
                    call=lambda h, r, q: calls.append(q) or {'running': True})
    assert out['status'] == 'qualification_dispatched'
    assert calls[0]['job']['original_bank_sha256'] == pointer['bank_sha256']
    assert calls[0]['job']['original_bank_path'] == pointer['remote_bank_path']
    assert calls[0]['job']['source_sha'] == 'f' * 40


def test_fitness_field_freezes_past_only_target_rows_from_ledger(case):
    _, _, pointer = promote(case)
    field = bank.fitness_field(case.config, case.challenge, FITNESS_METRICS)
    assert field is not None
    assert field['target'] == TARGET
    assert field['epochs'] == [26028]
    assert len(field['rows']) == 20
    for row in field['rows']:
        assert set(row) == set(FITNESS_METRICS)
        assert all(type(v) in (float, int) and not isinstance(v, bool) for v in row.values())
    frozen = reference.freeze(field, target=case.challenge['target'],
        epoch=case.challenge['epoch'], metrics=FITNESS_METRICS,
        scorer_sha256='a' * 64, settings_sha256='b' * 64, settings_file_sha256='c' * 64)
    assert frozen.row_count == 20 and frozen.reference_epochs == (26028,)
    assert frozen.source_sha256 == bank.sha(reference.serialize(field))
    assert frozen.score(field['rows'][0]) >= 10  # dense rank sum lower bound across 10 metrics


def test_fitness_field_returns_none_when_feed_disabled_or_no_past_board(case):
    disabled = copy.deepcopy(case.config); disabled.pop('parent_bank_feed')
    assert bank.fitness_field(disabled, case.challenge, FITNESS_METRICS) is None
    _, _, pointer = promote(case)
    assert bank.fitness_field(case.config, {'epoch': 26026, 'target': TARGET}, FITNESS_METRICS) is None
    # A field for a different target cannot leak rows from this target's ledger.
    other = copy.deepcopy(case.config)
    assert bank.fitness_field(other, {'epoch': 26030, 'target': 'OTHER'}, FITNESS_METRICS) is None


def test_fitness_field_uses_latest_ledger_board_not_an_older_revision(case):
    _, _, _ = promote(case)
    before = bank.fitness_field(case.config, case.challenge, FITNESS_METRICS)
    assert before is not None and len(before['rows']) == 20
    # A corrected board publishes a new content-addressed digest; the ledger now
    # points at exactly that digest, so the older revision is not combined.
    revised_payload = copy.deepcopy(PUBLIC); revised_payload['data'] = revised_payload['data'][:5]
    revised = bank.public_epoch(revised_payload, TARGET, 26028, 26030)
    content = bank.raw(revised); digest = bank.sha(content)
    bank.immutable(case.directory / 'public' / (digest + '.json'), content)
    ledger = json.loads(bank.read(case.directory / 'public-ledger.json'))
    assert ledger['26028']['sha256'] != digest
    ledger['26028']['sha256'] = digest
    bank.atomic(case.directory / 'public-ledger.json', ledger)
    after = bank.fitness_field(case.config, case.challenge, FITNESS_METRICS)
    assert after is not None and after['epochs'] == [26028]
    assert len(after['rows']) == len(revised['rows']) < 20  # replaced, never appended
