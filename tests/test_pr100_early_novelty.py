"""Funnel ordering and qualified-runtime transport, not a new novelty algorithm."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from nanobody_opt import pr100_live_controller as controller, pr100_live_worker as worker
from nanobody_opt import pr100_novelty_runtime as runtime, rules
from tests.test_pr100_live_controller import config, challenge, clock
from tests.test_pr100_segment_generation import bank, run_worker
from tests.test_pr100_live_worker import native_env, native_job, profile, StubNovelty, write_input

QUALIFIED = {'python': '/qualified/python', 'path': '/qualified/bin',
             'python_path': '/qualified/site', 'identity_sha256': 'a' * 64}


@pytest.mark.parametrize('flag', [None, False])
def test_default_off_preserves_rows_without_archive_or_runtime(tmp_path, flag):
    rows = [{'sequence': 'A'}]
    job = {} if flag is None else {'sequence_full_archive_novelty': flag}
    kept, proof = worker.sequence_novelty_prefilter(job, tmp_path, rows,
        check=lambda *_a, **_kw: pytest.fail('disabled search'))
    assert kept is rows and proof is None


@pytest.mark.parametrize('flag', [None, 1, 'true', {}])
def test_bad_explicit_flag_fails_closed(tmp_path, flag):
    with pytest.raises(ValueError, match='must be a boolean'):
        worker.sequence_novelty_prefilter({'sequence_full_archive_novelty': flag}, tmp_path, [])


def job_for(tmp_path):
    archive = tmp_path / 'archive.csv'; archive.write_text('sequence,sequence_hash\nAAAA,archive-id\n')
    return {'sequence_full_archive_novelty': True, 'novelty_runtime': QUALIFIED,
            'archive_sha256': worker.file_digest(archive)}


def test_prefilter_uses_qualified_child_with_full_rows_and_bound_input(tmp_path):
    job = job_for(tmp_path)
    rows = [{'sequence': s, 'source': 'segment_recombination', 'pr100_annotation': {'original': True}}
            for s in ('CCCC', 'DDDD', 'EEEE')]
    def run(command, **kwargs):
        assert command == [QUALIFIED['python'], '-B', '-m', 'nanobody_opt.pr100_novelty_runtime']
        assert kwargs['capture_output'] is True and 'start_new_session' not in kwargs
        request = json.loads(kwargs['input'])
        assert request['rows'] == rows and request['archive_sha256'] == job['archive_sha256']
        reply = {'schema': runtime.SCHEMA, 'status': 'complete', 'archive_rows': 1,
                 **{k: request[k] for k in ('runtime_sha256', 'archive_sha256', 'input_sha256')},
                 'verdicts': [{'sequence_hash': rules.sequence_hash(row['sequence']),
                     'validator_novel': i != 1, 'exact_historical': False,
                     'coarse_candidates': 1, 'maximum_identity': .96 if i == 1 else .7}
                     for i, row in enumerate(rows)]}
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply), stderr='')
    def check(rows, archive, *, runtime):
        return globals()['runtime'].check(rows, archive, runtime=runtime, run=run)
    kept, proof = worker.sequence_novelty_prefilter(job, tmp_path, rows, check=check)
    assert [r['sequence'] for r in kept] == ['CCCC', 'EEEE']
    assert kept[0]['pr100_annotation'] == rows[0]['pr100_annotation']
    assert proof['removed_sequence_hashes'] == [rules.sequence_hash('DDDD')]
    assert proof['input_sha256'] == rules.digest(rows)
    assert proof['input_count'] == 3 and proof['kept_count'] == 2 and proof['removed_count'] == 1


def test_empty_cohort_skips_child_but_checks_archive_and_configuration(tmp_path):
    job = job_for(tmp_path)
    kept, proof = worker.sequence_novelty_prefilter(job, tmp_path, [],
        check=lambda *_a, **_kw: pytest.fail('empty search'))
    assert kept == [] and proof['input_count'] == proof['removed_count'] == 0
    job['archive_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='archive snapshot changed'):
        worker.sequence_novelty_prefilter(job, tmp_path, [])


def test_search_failure_is_not_a_novelty_pass(tmp_path):
    job = job_for(tmp_path)
    def broken(*_args, **_kwargs): raise RuntimeError('inert qualified child failure')
    with pytest.raises(RuntimeError):
        worker.sequence_novelty_prefilter(job, tmp_path, [{'sequence': 'CCCC'}], check=broken)


def test_real_sequence_phase_filters_after_annotation_before_weighted_order(run_worker, monkeypatch):
    original_order = worker.interleave_segment_controls
    observed = []
    def check(rows, archive, *, runtime):
        assert runtime == QUALIFIED and all('pr100_annotation' in row for row in rows)
        observed.extend(rows)
        return [{**r, 'novelty': {'validator_novel': True}} for r in rows[1::2]], 1
    def order(rows, **kwargs):
        assert rows and all(r['novelty']['validator_novel'] for r in rows)
        return original_order(rows, **kwargs)
    monkeypatch.setattr(worker, 'check_novelty', check)
    monkeypatch.setattr(worker, 'interleave_segment_controls', order)
    result = run_worker(segment_generation={'enabled': True, 'limit': 16},
        sequence_full_archive_novelty=True, novelty_runtime=QUALIFIED)
    proof = result['early_archive_novelty']
    assert proof['input_count'] == len(observed) and proof['removed_count'] == len(observed[::2])
    assert {r['sequence'] for r in result['candidates']} == {r['sequence'] for r in observed[1::2]}
    assert len(result['generated_sequence_hashes']) == len(observed)
    assert result['novelty_checked'] is False  # Final publication check remains required.
    assert sum(result['annotation_reasons'].values()) == len(observed)


def test_early_pass_never_bypasses_final_archive_check(native_env):
    write_input(native_env, ['A'])
    incoming = json.loads((native_env / 'input.json').read_text())
    incoming['early_archive_novelty'] = {'runtime_sha256': 'a' * 64, 'kept_count': 1}
    incoming['candidates'][0]['novelty'] = {'validator_novel': True}
    (native_env / 'input.json').write_text(json.dumps(incoming))
    archive = native_env / 'archive.csv'; archive.write_text('sequence\nZZZ\n')
    novelty = StubNovelty(novel=False)
    result = worker.native_phase(native_job(repeats=3, archive_sha256=worker.file_digest(archive)),
        native_env, novelty=novelty,
        measure=lambda sequences, workers, **_kwargs: ([profile(70) for _ in sequences], 0., 1.))
    assert novelty.searched == ['A'] and result['native_passed'] == 1
    assert result['novelty_checked'] is True and result['candidates'] == []


@pytest.mark.parametrize('enabled', [False, True])
def test_controller_supplies_qualified_runtime_to_sequence_only_when_enabled(tmp_path, enabled):
    value = config(tmp_path)
    value['worker']['sequence_full_archive_novelty'] = enabled
    _, request = controller.phase_request(value, challenge(), 'sequence', {},
                                          time.time() + 2000, clock())
    job = json.loads(base64.b64decode(request['files']['job.json']))
    assert ('novelty_runtime' in job) is enabled
    if enabled: assert job['novelty_runtime'] == value['hosts']['slave']['novelty_runtime']
    assert request['spec']['environment']['CUDA_VISIBLE_DEVICES'] == ''


def test_controller_refuses_enabled_sequence_without_qualified_runtime(tmp_path):
    value = config(tmp_path); value['worker']['sequence_full_archive_novelty'] = True
    del value['hosts']['slave']['novelty_runtime']
    with pytest.raises(ValueError, match='dedicated novelty runtime'):
        controller.phase_request(value, challenge(), 'sequence', {}, time.time() + 2000, clock())


def test_recorded_59_loss_cohort_is_sent_before_native_without_parent_restriction(tmp_path):
    """Recorded final outcomes exercise transport; this is not a live search replay."""
    fixture = Path(__file__).parent / 'fixtures/pr100/early-novelty-26038-losses.json.gz'
    value = json.loads(gzip.decompress(fixture.read_bytes()))
    rows = value['rows']
    assert len(rows) == value['native_passed'] == 59
    assert {tier: sum(r['observed_tier'] == tier for r in rows) for tier in ('A', 'B')} == {'A': 11, 'B': 48}
    assert all(r['sequence_hash'] == rules.sequence_hash(r['sequence']) for r in rows)
    assert value['final_novelty_kept_count'] == 0
    job = job_for(tmp_path)
    def run(_command, **kwargs):
        request = json.loads(kwargs['input'])
        assert request['rows'] == rows  # No selected-parent or top-50-only subset.
        reply = {'schema': runtime.SCHEMA, 'status': 'complete', 'archive_rows': 9677,
            **{k: request[k] for k in ('runtime_sha256', 'archive_sha256', 'input_sha256')},
            'verdicts': [{'sequence_hash': row['sequence_hash'], 'validator_novel': False,
                'exact_historical': False, 'coarse_candidates': 1, 'maximum_identity': .96}
                for row in rows]}
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply), stderr='')
    def check(rows, archive, *, runtime):
        return globals()['runtime'].check(rows, archive, runtime=runtime, run=run)
    kept, proof = worker.sequence_novelty_prefilter(job, tmp_path, rows, check=check)
    assert kept == [] and proof['removed_count'] == 59
    assert set(proof['removed_sequence_hashes']) == {r['sequence_hash'] for r in rows}


def test_replay_harness_binds_real_cohort_archive_and_expected_rejections(tmp_path, monkeypatch):
    from scripts.pr100_early_novelty_replay import replay
    fixture = Path(__file__).parent / 'fixtures/pr100/early-novelty-26038-losses.json.gz'
    value = json.loads(gzip.decompress(fixture.read_bytes()))
    job = job_for(tmp_path)
    value.update(archive_sha256=job['archive_sha256'], novelty_runtime=QUALIFIED)
    cohort = tmp_path / 'cohort.json'; cohort.write_text(json.dumps(value))
    calls = []
    def check(rows, archive, *, runtime):
        calls.append((rows, archive, runtime)); return [], 1
    monkeypatch.setattr(worker, 'check_novelty', check)
    output = tmp_path / 'replay.json'
    result = replay(cohort, tmp_path / 'archive.csv', output, expected_rejected=cohort)
    assert result['expected_rejected_count'] == result['removed_count'] == 59
    recorded = json.loads(output.read_text())
    assert recorded['native_profiles_run'] == 0 and recorded['runtime_sha256'] == QUALIFIED['identity_sha256']
    assert len(calls) == 1 and calls[0][0] == value['rows']
    assert recorded['cohort_sha256'] == hashlib.sha256(cohort.read_bytes()).hexdigest()
    (tmp_path / 'archive.csv').write_text('changed')
    with pytest.raises(ValueError, match='archive identity mismatch'):
        replay(cohort, tmp_path / 'archive.csv', output)
    assert len(calls) == 1  # Stale inputs are rejected before the child starts.
