"""Scheduling boundaries; biological profiling remains a real-runtime check."""
import ast
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import types
import pytest
from nanobody_opt import pr100_filter, pr100_live_worker as worker

def test_real_upstream_batch_loop_never_starts_more_timeouts_than_slots(monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'NOVA-nanobody-filter/metanano/services/developability_service.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DevelopabilityService')
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'analyze_batch_async')
    method.returns = None
    for arg in method.args.args: arg.annotation = None
    scope = {'asyncio': asyncio, 'hashlib': hashlib}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), scope)
    observed = []
    class Manager:
        def __init__(self, config):
            self.batch_size, self.capacity = config.batch_size, config.max_concurrent_tnp
        async def shutdown(self): observed.append('shutdown')
    class Service:
        analyze_batch_async = scope['analyze_batch_async']
        def __init__(self, config, manager): self.manager, self.active = manager, 0
        async def analyze_async(self, sequence):
            self.active += 1
            try:
                if self.active > self.manager.capacity: raise TimeoutError('queued timeout')
                await asyncio.sleep(0)
                return {'sequence': sequence, 'passed': True}
            finally: self.active -= 1
    for name, value in [('async_manager', Manager), ('developability_service', Service)]:
        module = types.ModuleType('metanano.services.' + name)
        setattr(module, 'AsyncServiceManager' if name == 'async_manager' else 'DevelopabilityService', value)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(pr100_filter, 'activate_filter', lambda: None)
    config = types.SimpleNamespace(async_config=types.SimpleNamespace(), developability=object())
    monkeypatch.setattr(pr100_filter, 'helper', lambda _: types.SimpleNamespace(developability_config=lambda: config))
    rows = asyncio.run(pr100_filter.develop(list(map(str, range(63))), 4))
    assert len(rows) == 63 and all(r.get('passed') for r in rows)
    assert observed == ['shutdown']
    config.async_config.batch_size = 63
    old = asyncio.run(Service(None, Manager(config.async_config)).analyze_batch_async(list(map(str, range(63)))))
    assert sum(bool(r.get('error')) for r in old) == 59

@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_bad_concurrency_rejected_without_backend(value):
    with pytest.raises(ValueError): asyncio.run(pr100_filter.develop(['A'], value))


@pytest.fixture
def observed_backend(monkeypatch):
    """Use the exact pinned batch loop with controllable, inert profile tasks."""
    source = pr100_filter.ROOT / 'NOVA-nanobody-filter/metanano/services/developability_service.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DevelopabilityService')
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'analyze_batch_async')
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    scope = {'asyncio': asyncio, 'hashlib': hashlib}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), scope)
    state = types.SimpleNamespace(shutdowns=0)

    class Manager:
        def __init__(self, config):
            self.batch_size = config.batch_size
        async def shutdown(self):
            state.shutdowns += 1

    class Service:
        analyze_batch_async = scope['analyze_batch_async']
        def __init__(self, config, manager):
            self.manager = manager
        async def analyze_async(self, sequence):
            return await state.analyze(sequence)

    for name, cls in [('async_manager', Manager), ('developability_service', Service)]:
        module = types.ModuleType('metanano.services.' + name)
        setattr(module, 'AsyncServiceManager' if name == 'async_manager' else 'DevelopabilityService', cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(pr100_filter, 'activate_filter', lambda: None)
    config = types.SimpleNamespace(async_config=types.SimpleNamespace(), developability=object())
    monkeypatch.setattr(pr100_filter, 'helper', lambda _: types.SimpleNamespace(developability_config=lambda: config))
    return state


def test_observations_arrive_before_slowest_profile_and_results_keep_input_order(observed_backend):
    async def check():
        release = asyncio.Event()
        seen = []
        async def analyze(sequence):
            if sequence == 'A':
                await release.wait()
            if sequence == 'C':
                raise TimeoutError('inert failure')
            return {'sequence': sequence, 'passed': True}
        def sink(sequence, value):
            seen.append((sequence, value))
            if sequence == 'C':
                assert not release.is_set()
                assert [s for s, _ in seen] == ['B', 'C']
                release.set()
        observed_backend.analyze = analyze
        result = await asyncio.wait_for(pr100_filter.develop(['A', 'B', 'C'], 3, on_result=sink), 2)
        assert [v.get('sequence') for v in result] == ['A', 'B', None]
        assert result[2]['error_kind'] == 'timeout'
        assert [s for s, _ in seen] == ['B', 'C', 'A']
    asyncio.run(check())
    assert observed_backend.shutdowns == 1


def test_observer_io_error_never_relabels_success_as_scientific_failure(observed_backend):
    async def analyze(sequence):
        return {'passed': True, 'sequence': sequence}
    def broken_sink(*args):
        raise OSError('inert disk failure')
    observed_backend.analyze = analyze
    errors = []
    result = asyncio.run(pr100_filter.develop(['A', 'B'], 2, on_result=broken_sink,
                                             on_result_error=errors.append))
    assert all(v['passed'] is True and 'error' not in v for v in result)
    assert errors == ['OSError', 'OSError']
    with pytest.raises(RuntimeError, match='observation failed'):
        asyncio.run(pr100_filter.develop(['A'], 1, on_result=broken_sink))


def test_cancelled_unfinished_profile_is_not_recorded_as_failure(observed_backend):
    async def check():
        began = asyncio.Event()
        seen = []
        async def analyze(sequence):
            began.set()
            await asyncio.Event().wait()
        observed_backend.analyze = analyze
        task = asyncio.create_task(pr100_filter.develop(['A'], 1,
                                   on_result=lambda *args: seen.append(args)))
        await asyncio.wait_for(began.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert seen == []
    asyncio.run(check())
    assert observed_backend.shutdowns == 1


class StubNovelty:
    """Archive novelty stand-in; records what it was asked to search."""
    def __init__(self, novel=True):
        self.novel, self.searched = novel, []

    def _init_worker(self, frame, sequences, all_archive=False):
        self.rows = len(frame)

    def _search_one(self, row):
        self.searched.append(row['sequence'])
        return {'validator_novel': self.novel, 'exact_historical': False,
                'coarse_candidates': 0, 'maximum_identity': 0.1}


def profile(psh, *, ok=True):
    return {'surface_hydrophobic_patches': psh, 'ok': ok}


def native_job(**over):
    job = {'phase': 'native', 'challenge': {'epoch': 1}, 'native_tools': '/tools',
           'cpu_slots': 8}
    job.update(over)
    return job


def write_input(directory, sequences, **over):
    payload = {'challenge': {'epoch': 1}, 'target': 'T', 'epoch': 1,
               'archive_sha256': 'x', 'archive_checked_at_unix': 0,
               'candidates': [{'sequence': s, 'sequence_hash': s.lower(),
                               'pr100_annotation': {'a': 1}} for s in sequences]}
    payload.update(over)
    (directory / 'input.json').write_text(json.dumps(payload))


@pytest.fixture
def native_env(monkeypatch, tmp_path):
    """Isolate the scheduling logic from real profiling and real gate config."""
    monkeypatch.setattr(worker.pr100_filter, 'helper',
                        lambda _: types.SimpleNamespace(tool_paths=None))
    monkeypatch.setattr(worker.rules, 'pinned_config',
                        lambda epoch: {'nanobody_requirements': {'psh_strict_rejection_threshold': 89}})
    monkeypatch.setattr(worker.rules, 'validate_evidence', lambda *a, **k: None)
    monkeypatch.setattr(worker.rules, 'FILTER_SHA', 'sha')
    # Keep the real decision rule; stub only the per-measurement gate verdict so
    # the fixture does not have to synthesise full A3D provenance.
    original = worker.selection.decide
    def decide(measurements, requirements, **kwargs):
        kwargs.pop('reason_for', None)
        return original(measurements, requirements,
                        reason_for=lambda v: '' if v.get('ok') else 'developability_red_flags',
                        **kwargs)
    monkeypatch.setattr(worker.selection, 'decide', decide)
    return tmp_path


class TestWorkerCount:
    def test_default_is_48_when_resources_allow(self):
        assert worker.worker_count({}, cores=224, memory=160 * 1024 ** 3) == 48

    def test_hard_cap_key_bounds_a_larger_request(self):
        assert worker.worker_count({'workers': 200, 'max_workers': 64},
                                   cores=224, memory=400 * 1024 ** 3) == 64

    def test_memory_bounds_concurrency_at_about_1_4_gb_each(self):
        assert worker.worker_count({'workers': 48}, cores=224, memory=14 * 1024 ** 3) == 10

    def test_cores_bound_concurrency(self):
        assert worker.worker_count({'workers': 48}, cores=8, memory=160 * 1024 ** 3) == 8

    def test_configurable_below_the_default(self):
        assert worker.worker_count({'workers': 12}, cores=224, memory=160 * 1024 ** 3) == 12

    @pytest.mark.parametrize('bad', [{'workers': 0}, {'workers': True}, {'max_workers': -1},
                                     {'workers': 1.5}])
    def test_invalid_worker_counts_refuse(self, bad):
        with pytest.raises(ValueError):
            worker.worker_count(bad)


class TestQueueTimeout:
    def test_budget_is_measured_from_slot_acquisition_not_enqueue(self):
        now = [0.0]
        def clock():
            return now[0]
        async def develop(sequences, workers):
            now[0] += 7.0          # real profiling time, slot already held
            return [profile(70) for _ in sequences]
        def queued_clock():
            now[0] += 100.0        # time spent before the batch starts
            return now[0]
        values, acquired, elapsed = worker.measure_batch(
            ['A', 'B'], 2, clock=clock, develop=develop)
        assert elapsed == pytest.approx(7.0)
        assert acquired == pytest.approx(0.0)
        assert len(values) == 2

    def test_never_enqueues_more_profiles_than_slots(self, native_env):
        seen = []
        async def develop(sequences, workers):
            seen.append(len(sequences))
            assert len(sequences) <= workers
            return [profile(70) for _ in sequences]
        def measure(sequences, workers, *, clock=None):
            return asyncio.run(develop(sequences, workers)), 0.0, 1.0
        write_input(native_env, [f'S{i}' for i in range(10)])
        worker.native_phase(native_job(workers=4, max_workers=4, repeats=1),
                            native_env, measure=measure)
        assert max(seen) <= 4

    def test_candidate_that_never_got_a_slot_is_not_run_not_failed(self, native_env):
        now = [0.0]
        def clock():
            return now[0]
        def measure(sequences, workers, *, clock=None):
            now[0] += 300.0
            return [profile(70) for _ in sequences], now[0], 300.0
        write_input(native_env, [f'S{i}' for i in range(12)])
        job = native_job(workers=4, max_workers=4, repeats=1,
                         candidate_budget_seconds=300, phase_budget_seconds=650)
        result = worker.native_phase(job, native_env, clock=clock, measure=measure)
        assert result['not_run'] > 0
        assert result['rejection_counts']['not_run'] == result['not_run']
        assert 'failed' not in result['rejection_counts']
        assert result['status'] == 'complete'


class TestRepeatsAndTiers:
    def measure_for(self, table):
        def measure(sequences, workers, *, clock=None):
            return [profile(table[s].pop(0)) for s in sequences], 0.0, 1.0
        return measure

    def test_survivor_gets_k_minus_one_more_measurements(self, native_env):
        table = {'A': [78, 81, 84]}
        write_input(native_env, ['A'])
        result = worker.native_phase(native_job(repeats=3), native_env,
                                     measure=self.measure_for(table))
        row = result['candidates'][0]
        assert row['repeats'] == 3
        assert row['psh_values'] == [78, 81, 84]
        assert row['psh_mean'] == pytest.approx(81.0)
        assert row['psh_sd'] == pytest.approx(3.0)
        assert row['tier'] == 'A' and row['margin_safe'] is True

    def test_single_high_measurement_is_never_tier_a(self, native_env):
        table = {'A': [88, 88, 88]}
        write_input(native_env, ['A'])
        result = worker.native_phase(native_job(repeats=3), native_env,
                                     measure=self.measure_for(table))
        assert [r['tier'] for r in result['candidates']] == ['B']

    def test_results_are_ordered_by_margin_tier_a_first(self, native_env):
        table = {'A': [86, 86, 86], 'B': [75, 75, 75], 'C': [80, 80, 80]}
        write_input(native_env, ['A', 'B', 'C'])
        result = worker.native_phase(native_job(repeats=3), native_env,
                                     measure=self.measure_for(table))
        assert [r['sequence'] for r in result['candidates']] == ['B', 'C', 'A']
        assert [r['tier'] for r in result['candidates']] == ['A', 'A', 'B']
        assert result['tier_counts'] == {'A': 2, 'B': 1}

    def test_allow_marginal_false_withholds_tier_b(self, native_env):
        table = {'A': [86, 86, 86], 'B': [75, 75, 75]}
        write_input(native_env, ['A', 'B'])
        result = worker.native_phase(native_job(repeats=3, allow_marginal=False),
                                     native_env, measure=self.measure_for(table))
        assert [r['sequence'] for r in result['candidates']] == ['B']

    def test_first_pass_failure_is_not_repeated(self, native_env):
        calls = []
        def measure(sequences, workers, *, clock=None):
            calls.append(list(sequences))
            return [profile(70, ok=False) for _ in sequences], 0.0, 1.0
        write_input(native_env, ['A'])
        result = worker.native_phase(native_job(repeats=3), native_env, measure=measure)
        assert calls == [['A']]
        assert result['candidates'] == []
        assert result['rejection_counts']['developability_red_flags'] == 1

    def test_no_eight_candidate_cap_on_output(self, native_env):
        table = {f'S{i}': [70, 70, 70] for i in range(30)}
        write_input(native_env, list(table))
        result = worker.native_phase(native_job(repeats=3, workers=8, max_workers=8),
                                     native_env, measure=self.measure_for(table))
        assert len(result['candidates']) == 30


class TestFunnelOrder:
    def test_novelty_runs_only_on_native_passers(self, native_env):
        table = {'A': [70, 70, 70], 'B': [70, 70, 70], 'C': [70, 70, 70]}
        table['C'] = [70, 70, 70]
        def measure(sequences, workers, *, clock=None):
            return [profile(70, ok=s != 'B') for s in sequences], 0.0, 1.0
        write_input(native_env, ['A', 'B', 'C'])
        (native_env / 'archive.csv').write_text('sequence\nZZZ\n')
        stub = StubNovelty()
        job = native_job(repeats=3, archive_sha256=worker.file_digest(native_env / 'archive.csv'))
        result = worker.native_phase(job, native_env, measure=measure, novelty=stub)
        assert sorted(set(stub.searched)) == ['A', 'C']   # never the native failure
        assert result['novelty_checked'] is True
        assert len(result['candidates']) == 2

    def test_archive_change_between_phases_refuses(self, native_env):
        def measure(sequences, workers, *, clock=None):
            return [profile(70) for _ in sequences], 0.0, 1.0
        write_input(native_env, ['A'])
        (native_env / 'archive.csv').write_text('sequence\nZZZ\n')
        job = native_job(repeats=3, archive_sha256='stale')
        with pytest.raises(ValueError):
            worker.native_phase(job, native_env, measure=measure, novelty=StubNovelty())

    def test_non_novel_passer_is_dropped(self, native_env):
        def measure(sequences, workers, *, clock=None):
            return [profile(70) for _ in sequences], 0.0, 1.0
        write_input(native_env, ['A'])
        (native_env / 'archive.csv').write_text('sequence\nZZZ\n')
        job = native_job(repeats=3, archive_sha256=worker.file_digest(native_env / 'archive.csv'))
        result = worker.native_phase(job, native_env, measure=measure,
                                     novelty=StubNovelty(novel=False))
        assert result['candidates'] == []


def test_interrupted_batch_preserves_completed_observation(native_env, monkeypatch):
    write_input(native_env, ['A', 'B'])
    def interrupted(sequences, workers, *, clock, on_result, on_result_error):
        on_result('B', profile(70))
        raise KeyboardInterrupt()
    monkeypatch.setattr(worker, 'measure_batch', interrupted)
    with pytest.raises(KeyboardInterrupt):
        worker.native_phase(native_job(repeats=3), native_env)
    assert (native_env / 'native-observations.jsonl').stat().st_size > 0
    running = json.loads((native_env / 'native-progress.json').read_text())
    assert running['status'] == 'running' and running['completed'] == 1
    assert running['observation_journal']['observations_durable'] == 1
    assert running['native_classification_complete'] is False
    assert running['not_started'] == 0 and running['not_completed'] == 1 and running['in_flight'] == 1
    assert 'native_passed' not in running and 'not_run' not in running


def test_out_of_order_callbacks_and_batch_fallback_count_each_repeat_once(native_env, monkeypatch):
    write_input(native_env, ['A', 'B', 'C'])
    def complete(sequences, workers, *, clock, on_result, on_result_error):
        for sequence in reversed(sequences):
            on_result(sequence, profile(70))
        return [profile(70) for _ in sequences], 0.0, 1.0
    monkeypatch.setattr(worker, 'measure_batch', complete)
    result = worker.native_phase(native_job(repeats=3), native_env)
    assert result['completed'] == 9
    assert result['observation_journal']['observations_durable'] == 9
    assert result['native_passed'] == 3
    assert result['native_classification_complete'] is True
    assert all(r['repeats'] == 3 for r in result['candidates'])


def test_journal_failure_does_not_block_passing_candidates(native_env, monkeypatch):
    from nanobody_opt import pr100_native_journal
    def broken_save(*args):
        raise OSError('inert journal failure')
    monkeypatch.setattr(pr100_native_journal.Journal, 'save', broken_save)
    write_input(native_env, ['A'])
    result = worker.native_phase(native_job(repeats=1), native_env,
                                 measure=lambda *a, **k: ([profile(70)], 0.0, 1.0))
    assert result['completed'] == 1 and result['native_passed'] == 1
    assert result['observation_journal']['status'] == 'degraded'
    assert result['status'] == 'complete'


def test_retry_measures_fresh_and_preserves_old_attempt(native_env):
    write_input(native_env, ['A'])
    calls = []
    def measure(*args, **kwargs):
        calls.append(1)
        return [profile(70)], 0.0, 1.0
    first = worker.native_phase(native_job(repeats=1), native_env, measure=measure)
    old_bytes = (native_env / 'native-observations.jsonl').read_bytes()
    second = worker.native_phase(native_job(repeats=1), native_env, measure=measure)
    assert calls == [1, 1]
    assert first['observation_journal']['attempt_id'] != second['observation_journal']['attempt_id']
    assert (native_env / 'native-observations.jsonl').read_bytes().startswith(old_bytes)
