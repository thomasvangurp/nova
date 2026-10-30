"""Bounded execution around unchanged scientific annotation and gate semantics."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

from nanobody_opt import gates, hydrophobicity, pr100_filter, pr100_live_worker as w, rules

CHALLENGE = {'epoch': 26034, 'target': 'Q9NZQ7', 'allowed_v': 'IGHV3-7', 'allowed_cdr3_length': 13}
IDENTITY = 'e153985994c071b7ed383372199c7e4af3d2f8fada6d001cfe6227bdad99af4f'


def test_existing_scientific_annotation_identity_is_unchanged():
    assert pr100_filter.annotation_identity() == IDENTITY


@pytest.mark.parametrize('job,cores,expected', [
    ({'workers': 48, 'cpu_slots': 56}, 56, 8),
    ({'workers': 48, 'cpu_slots': 12}, 56, 4),
    ({'workers': 48, 'cpu_slots': 56}, 6, 2),
    ({'workers': 2, 'cpu_slots': 56}, 56, 2),
    ({'workers': 48, 'max_workers': 3, 'cpu_slots': 56}, 56, 3),
    ({'workers': 48, 'cpu_slots': 2}, 56, 1),
    ({}, 56, 1),
])
def test_annotation_resources_bound_workers_slots_affinity_and_small_hard_cap(monkeypatch, job, cores, expected):
    monkeypatch.delenv('HMMER_NCPU', raising=False)
    assert w.annotation_worker_count(job, segment_enabled=True, cores=cores) == expected


def test_existing_hmmer_thread_setting_is_accounted_for_without_modification(monkeypatch):
    monkeypatch.setenv('HMMER_NCPU', '7')
    assert w.annotation_worker_count({'workers': 48, 'cpu_slots': 24}, segment_enabled=True, cores=56) == 3
    assert w.os.environ['HMMER_NCPU'] == '7'


@pytest.mark.parametrize('job', [{'workers': True}, {'cpu_slots': 0}, {'max_workers': -1}])
def test_bad_annotation_resource_bounds_refuse_before_threads(job):
    with pytest.raises(ValueError, match='positive integers'):
        w.annotation_worker_count(job, segment_enabled=True, cores=56)
    # These fields did not control baseline sequence annotation before.
    assert w.annotation_worker_count(job, segment_enabled=False, cores=56) == 1


def test_cache_warmup_validates_imports_references_and_identity_on_main_thread(monkeypatch):
    import importlib
    calls = []
    main = threading.get_ident()
    def record(name, result=None):
        assert threading.get_ident() == main
        calls.append(name)
        return result
    monkeypatch.setattr(importlib, 'import_module', lambda name: record(name))
    monkeypatch.setattr(pr100_filter, 'validate_annotation_runtime', lambda: record('runtime'))
    monkeypatch.setattr(pr100_filter, 'helper', lambda name: SimpleNamespace(
        _anarci_panel=lambda species: record('panel:' + species),
        _hmm_digest=lambda: record('hmm')))
    monkeypatch.setattr(pr100_filter, 'annotation_identity', lambda: record('identity', IDENTITY))
    assert w._warm_sequence_annotation() == IDENTITY
    assert calls == ['abnumber', 'anarci.anarci', 'runtime', 'panel:human', 'hmm', 'identity']


def test_parallel_and_serial_exact_gate_outputs_and_order_match(monkeypatch):
    annotations = []
    monkeypatch.setattr(gates, 'check_sequence', lambda seq, req: seq == 'X')
    def annotate(seq):
        annotations.append(seq)
        return {'sequence_hash': rules.sequence_hash(seq), 'status': 'ok'}
    monkeypatch.setattr(pr100_filter, '_annotate_uncached', annotate)
    monkeypatch.setattr(pr100_filter, 'annotation_identity', lambda: IDENTITY)
    monkeypatch.setattr(w, '_warm_sequence_annotation', lambda: IDENTITY)
    monkeypatch.setattr(rules, 'annotation_reason', lambda annotation, req, *, sequence, epoch_challenge:
                        'wrong_epoch_v_gene' if sequence == 'WRONG' else '')
    monkeypatch.setattr(hydrophobicity, 'check', lambda seq: (None, seq != 'HYDRO'))
    sequences = ['OK', 'X', 'WRONG', 'HYDRO', 'OK2']
    serial = list(w.sequence_annotations(sequences, {}, CHALLENGE, workers=1))
    parallel = list(w.sequence_annotations(sequences, {}, CHALLENGE, workers=3))
    assert serial == parallel
    assert [reason for _, reason in parallel] == ['', 'basic_sequence', 'wrong_epoch_v_gene',
                                                 hydrophobicity.REJECT_REASON, '']
    assert 'X' not in annotations
    assert all(value['annotation_identity'] == IDENTITY for value, _ in parallel if value)


def test_bounded_window_and_out_of_order_completion_preserve_input_order(monkeypatch):
    second_done, release_first = threading.Event(), threading.Event()
    consumed, completed, result = [], [], []
    monkeypatch.setattr(w, '_warm_sequence_annotation', lambda: IDENTITY)
    def annotate(seq, requirements, expected, identity=None):
        assert identity == IDENTITY
        if seq == 0:
            assert release_first.wait(3)
        if seq == 1:
            second_done.set()
        completed.append(seq)
        return seq, ''
    def inputs():
        for i in range(6):
            consumed.append(i)
            yield i
    monkeypatch.setattr(w, '_sequence_annotation', annotate)
    with ThreadPoolExecutor(max_workers=1) as driver:
        future = driver.submit(lambda: list(w.sequence_annotations(inputs(), {}, CHALLENGE, workers=2)))
        try:
            assert second_done.wait(3)
            assert consumed == [0, 1]  # no eager submission of the whole batch
            assert completed == [1]
        finally:
            release_first.set()
        result = future.result(timeout=3)
    assert result == [(i, '') for i in range(6)]


def test_failures_propagate_in_input_order_and_threads_are_joined(monkeypatch):
    finished = threading.Event()
    monkeypatch.setattr(w, '_warm_sequence_annotation', lambda: IDENTITY)
    def annotate(seq, *args):
        if seq == 'first':
            assert finished.wait(3)
            raise ValueError('first execution failure')
        finished.set()
        raise RuntimeError('later execution failure')
    monkeypatch.setattr(w, '_sequence_annotation', annotate)
    with pytest.raises(ValueError, match='first execution failure'):
        list(w.sequence_annotations(['first', 'later'], {}, CHALLENGE, workers=2))
    assert not any(t.name.startswith('pr100-annotation') for t in threading.enumerate())


def test_serial_path_does_not_warm_or_create_executor(monkeypatch):
    monkeypatch.setattr(w, '_warm_sequence_annotation', lambda: pytest.fail('serial path warmed'))
    monkeypatch.setattr(w, 'ThreadPoolExecutor', lambda **kw: pytest.fail('serial path pooled'))
    monkeypatch.setattr(w, '_sequence_annotation', lambda seq, *args: (seq, ''))
    assert list(w.sequence_annotations(['a', 'b'], {}, CHALLENGE)) == [('a', ''), ('b', '')]


@pytest.mark.parametrize('workers', [0, 9, True, 1.5])
def test_invalid_parallel_bounds_refuse(workers):
    with pytest.raises(ValueError, match='within one to eight'):
        list(w.sequence_annotations([], {}, CHALLENGE, workers=workers))


# Exercise the real sequence-phase integration with the existing numbered-donor
# fixture; only external scientific runtime calls are stubbed by that fixture.
from tests.test_pr100_segment_generation import bank, run_worker


def test_sequence_phase_parallel_branch_matches_serial_candidates_and_gate_counts(run_worker, monkeypatch):
    monkeypatch.setattr(w, 'available_cores', lambda: 12)
    monkeypatch.setattr(w, '_warm_sequence_annotation', lambda: 'a' * 64)
    serial = run_worker(segment_generation={'enabled': True, 'limit': 16}, workers=1, cpu_slots=12)
    parallel = run_worker(segment_generation={'enabled': True, 'limit': 16}, workers=4, cpu_slots=12)
    assert serial['annotation_workers'] == 1 and parallel['annotation_workers'] == 4
    assert parallel['candidates'] == serial['candidates']
    assert parallel['annotation_reasons'] == serial['annotation_reasons']


def test_invalid_existing_hmmer_count_refuses_without_setting_it(monkeypatch):
    monkeypatch.setenv('HMMER_NCPU', 'invalid')
    with pytest.raises(ValueError, match='invalid HMMER worker count'):
        w.annotation_worker_count({'workers': 48, 'cpu_slots': 24}, segment_enabled=True, cores=56)
