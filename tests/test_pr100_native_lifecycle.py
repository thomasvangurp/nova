"""Actual pinned orchestration with inert profile work; no native/GPU calls."""
import asyncio
from collections import deque
import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest
from nanobody_opt import pr100_filter, pr100_native_refill as refill

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / 'NOVA-nanobody-filter'


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(sys, 'path', list(sys.path))
    # Execute actual pinned config/manager/scheduler/service modules. Replace
    # only hardware probes and the scientific worker boundary with inert work.
    for name in ('metanano', 'metanano.services', 'metanano.utils', 'metanano.filters'):
        module = ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, PINNED / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module); spec.loader.exec_module(module)
        return module

    config = load('metanano.config', 'metanano/config.py')
    gpu = load('metanano.utils.gpu_scheduler', 'metanano/utils/gpu_scheduler.py')
    load('metanano.utils.tnp_wrapper', 'metanano/utils/tnp_wrapper.py')
    state = SimpleNamespace(configs=[], scientific_configs=[], managers=[], profiles=0,
        active=0, maximum=0, shutdowns=0, gpu_gets=0, detections=0, barrier=None,
        fail_shutdown=False, block=False, entered=threading.Event(), released=threading.Event())
    original_get = gpu.get_gpu_scheduler
    def get_gpu(*args, **kwargs):
        state.gpu_gets += 1
        return original_get(*args, **kwargs)
    monkeypatch.setattr(gpu, 'get_gpu_scheduler', get_gpu)
    async def detect(_self):
        state.detections += 1; await asyncio.sleep(0); return [0]
    async def stats(_self, _index): return None
    monkeypatch.setattr(gpu.GPUScheduler, '_detect_gpus', detect)
    monkeypatch.setattr(gpu.GPUScheduler, '_get_gpu_stats', stats)
    manager_module = load('metanano.services.async_manager', 'metanano/services/async_manager.py')
    original_init = manager_module.AsyncServiceManager.__init__
    original_shutdown = manager_module.AsyncServiceManager.shutdown
    def manager_init(self, cfg):
        original_init(self, cfg); state.managers.append(self)
    async def shutdown(self):
        await original_shutdown(self); state.shutdowns += 1
        if state.fail_shutdown: raise RuntimeError('inert teardown failure')
    monkeypatch.setattr(manager_module.AsyncServiceManager, '__init__', manager_init)
    monkeypatch.setattr(manager_module.AsyncServiceManager, 'shutdown', shutdown)
    lock = threading.Lock()
    class Filter:
        def __init__(self, cfg): state.scientific_configs.append(cfg.model_dump())
        def compute_tnp_profile(self, sequence, *, execution):
            with lock:
                state.profiles += 1; state.active += 1; state.maximum = max(state.maximum, state.active)
            try:
                if state.barrier: state.barrier.wait(timeout=10)
                state.entered.set()
                if state.block:
                    while not execution._cancelled.wait(.001): pass
                    state.released.set()
                else: time.sleep(.001)
                return {'flags': {}, 'value': sequence}
            finally:
                with lock: state.active -= 1
        def analyze_profile(self, sequence, profile):
            return SimpleNamespace(to_dict=lambda: {'passed': True, 'value': sequence})
    filter_module = ModuleType('metanano.filters.developability')
    filter_module.DevelopabilityFilter = Filter; filter_module.DevelopabilityResult = dict
    monkeypatch.setitem(sys.modules, filter_module.__name__, filter_module)
    service = load('metanano.services.developability_service', 'metanano/services/developability_service.py')
    spec = importlib.util.spec_from_file_location('inert_aggregation_config', ROOT / 'utils/aggrescan3d.py')
    aggregation = importlib.util.module_from_spec(spec); spec.loader.exec_module(aggregation)
    factory = aggregation.developability_config
    state.expected_science = factory().developability.model_dump()
    state.expected_async = factory().async_config.model_dump()
    def configured():
        cfg = factory(); state.configs.append(cfg); return cfg
    monkeypatch.setattr(pr100_filter, 'activate_filter', lambda: None)
    monkeypatch.setattr(pr100_filter, 'helper', lambda _name: SimpleNamespace(developability_config=configured))
    state.gpu = gpu; state.manager_module = manager_module; state.service = service; state.config = config
    return state


def calls(count, *, timeout=130):
    queue = deque(str(i) for i in range(count))
    return lambda free: [{'sequence': queue.popleft(), 'timeout_seconds': timeout,
        'record': lambda *_args: None, 'record_error': lambda *_args: None}
        for _ in range(min(free, len(queue)))]


def assert_drained(state):
    assert state.active == 0
    assert not any(t.name.startswith('pr100-native-refill') for t in threading.enumerate())
    assert all(not m._initialized and m._gpu_scheduler is None for m in state.managers)


@pytest.mark.parametrize('width', [4, 32, 96])
def test_repeated_concurrent_real_managers_never_touch_global_scheduler(backend, width):
    backend.barrier = threading.Barrier(width)
    for _ in range(2):  # Three waves in each of two Runner lifetimes, same process.
        def completed(*_args):
            assert not any('_health_check_loop' in getattr(t.get_coro(), '__qualname__', '')
                           for t in asyncio.all_tasks())
        result = refill.run(calls(width * 3), completed, width)
        assert result['completed_calls'] == width * 3 and result['maximum_in_flight'] == width
        assert_drained(backend)
    assert backend.profiles == backend.shutdowns == width * 6
    assert backend.maximum == width
    assert backend.gpu_gets == backend.detections == 0
    assert all(cfg == backend.expected_science for cfg in backend.scientific_configs)
    for cfg in backend.configs:
        actual = cfg.async_config.model_dump(); expected = dict(backend.expected_async)
        assert 129 < actual.pop('task_timeout') <= 130
        expected.pop('task_timeout'); expected['max_concurrent_tnp'] = expected['batch_size'] = 1
        expected['gpu_scheduler'] = {**expected['gpu_scheduler'], 'enabled': False}
        assert actual == expected


def test_existing_foreign_gpu_owner_is_untouched(backend):
    async def scenario():
        owner = backend.gpu.get_gpu_scheduler(backend.config.GPUSchedulerConfig())
        await owner.initialize(); task = owner._health_check_task
        try:
            assert task and not task.done()
            before = (backend.gpu_gets, backend.detections)
            await asyncio.gather(*[pr100_filter.develop([str(i)], 1, cpu_only_manager=True) for i in range(32)])
            assert owner._initialized and owner._health_check_task is task and not task.done()
            assert (backend.gpu_gets, backend.detections) == before
        finally: await owner.shutdown()
        assert task.done()
    asyncio.run(scenario())


def test_default_adapter_keeps_original_gpu_configuration(backend):
    result = asyncio.run(pr100_filter.develop(['one'], 1))
    assert result[0]['passed'] and backend.gpu_gets == backend.detections == 1
    assert backend.configs[0].async_config.gpu_scheduler.enabled is True
    assert backend.managers[0]._gpu_scheduler._health_check_task.done()


@pytest.mark.parametrize('bad', [None, 0, 1, 'false'])
def test_cpu_manager_flag_is_strict_before_backend_import(backend, bad):
    with pytest.raises(ValueError, match='boolean'):
        asyncio.run(pr100_filter.develop(['one'], 1, cpu_only_manager=bad))
    assert not backend.managers and backend.profiles == 0


def test_cancellation_waits_for_original_profile_thread_cleanup(backend):
    backend.block = True
    async def scenario():
        task = asyncio.create_task(pr100_filter.develop(['one'], 1, cpu_only_manager=True))
        while not backend.entered.is_set(): await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert backend.released.is_set() and backend.active == 0
        assert backend.shutdowns == 1
    asyncio.run(scenario())
    assert backend.gpu_gets == backend.detections == 0


def test_teardown_failure_propagates_after_threads_and_observations_finish(backend):
    backend.fail_shutdown = True; observed = []
    with pytest.raises(RuntimeError, match='teardown'):
        refill.run(calls(1), lambda *_args: None, 4,
            develop=lambda sequences, concurrency, **kw: pr100_filter.develop(sequences, concurrency,
                **{**kw, 'on_result': lambda sequence, value: observed.append(value)}))
    assert len(observed) == 1 and observed[0]['passed']
    assert_drained(backend)
    assert backend.gpu_gets == backend.detections == 0



def leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in leaves(child)]
    return [error]


@pytest.mark.parametrize('width', [4, 32, 96])
def test_concurrent_teardown_failures_are_retained_and_all_threads_drained(backend, width):
    backend.fail_shutdown = True; backend.barrier = threading.Barrier(width)
    with pytest.raises(BaseExceptionGroup) as raised:
        refill.run(calls(width), lambda *_args: None, width)
    errors = leaves(raised.value)
    teardown = [e for e in errors if isinstance(e, RuntimeError) and 'teardown' in str(e)]
    assert len(teardown) == backend.shutdowns == width
    assert all(isinstance(e, (RuntimeError, asyncio.CancelledError)) for e in errors)
    assert_drained(backend)
    assert backend.gpu_gets == backend.detections == 0


def test_callback_failure_is_retained_alongside_shutdown_failures(backend):
    backend.fail_shutdown = True; backend.barrier = threading.Barrier(4)
    def completed(*_args): raise ValueError('inert observation failure')
    def report_failed(*_args): raise OSError('inert observation reporting failure')
    def admitted(free):
        rows = next_calls(free)
        for row in rows: row.update(record=completed, record_error=report_failed)
        return rows
    next_calls = calls(4)
    with pytest.raises(BaseExceptionGroup) as raised:
        refill.run(admitted, lambda *_args: None, 4)
    errors = leaves(raised.value)
    assert any(isinstance(e, RuntimeError) and 'observation failed' in str(e) for e in errors)
    assert sum(isinstance(e, RuntimeError) and 'teardown' in str(e) for e in errors) == 4
    assert_drained(backend)



def test_local_initialization_failure_remains_explicit_profile_error(backend, monkeypatch):
    async def failed(_self): raise RuntimeError('inert local manager initialization failure')
    monkeypatch.setattr(backend.manager_module.AsyncServiceManager, 'initialize', failed)
    values = asyncio.run(pr100_filter.develop(['one'], 1, cpu_only_manager=True))
    assert values[0]['error'] is True and values[0]['passed'] is False
    assert values[0]['error_kind'] == 'profile_exception'
    assert backend.profiles == 0 and backend.shutdowns == 1
    assert backend.gpu_gets == backend.detections == 0


from tests.test_pr100_native_execution_identity import identity


@pytest.mark.parametrize('observations', [1, 3])
def test_old_adapter_checkpoint_is_retained_deferred_and_never_redrawn(identity, monkeypatch, tmp_path, observations):
    import hashlib
    from nanobody_opt import pr100_live_worker as worker, pr100_native_journal as journal, rules
    from tests.test_pr100_native_screen_integration import CHALLENGE, profile
    requirements = rules.pinned_config(CHALLENGE['epoch'])['nanobody_requirements']
    context = journal._context(None, None, None)
    context['source_sha256']['adapter'] = '46cdaa528defef30f43b67d433e71e7e5d16c7be28d52d18f89ab2166cf6c6b4'
    previous = identity(requirements=requirements, challenge=CHALLENGE)
    row = {'sequence': 'AAAA', 'pr100_annotation': {}, 'native_resume': {
        'schema': 'pr100-native-confirmation-checkpoint-v1', 'sequence_hash': rules.sequence_hash('AAAA'),
        'identity': previous, 'measurements': [profile('AAAA') for _ in range(observations)],
        'required_cpu_observations': 3, 'execution_host': 'slave'}}
    pr100_filter.atomic(tmp_path / 'input.json', {'challenge': CHALLENGE, 'epoch': CHALLENGE['epoch'],
        'target': CHALLENGE['target'], 'archive_sha256': 'a'*64, 'archive_checked_at_unix': 1, 'candidates': [row]})
    context['source_sha256']['adapter'] = hashlib.sha256((ROOT / 'nanobody_opt/pr100_filter.py').read_bytes()).hexdigest()
    assert previous != identity(requirements=requirements, challenge=CHALLENGE)
    monkeypatch.setattr(pr100_filter, 'helper', lambda _: SimpleNamespace(tool_paths=None))
    result = worker.native_phase({'challenge': CHALLENGE, 'native_tools': '/inert', 'workers': 1,
        'native_host': 'slave', 'repeats': 3, 'native_interleave_confirmations': True, 'native_work_conserving': True},
        tmp_path, refill=lambda *a, **kw: refill.run(*a, **kw,
            develop=lambda *_a, **_k: pytest.fail('old bound checkpoint reached profiling')))
    assert result['completed'] == result['fresh_evaluations'] == result['repeat_evaluations'] == 0
    assert result['terminal_native_outcomes'] == {}
    assert result['blocked_confirmations'][0]['continuation_blocked'] == 'native_runtime_identity_mismatch'
    assert result['blocked_confirmations'][0]['native_resume'] == row['native_resume']



def test_handled_caller_exception_is_not_grouped_as_adapter_failure(backend):
    backend.fail_shutdown = True
    async def scenario():
        try: raise ValueError('unrelated handled caller exception')
        except ValueError:
            with pytest.raises(RuntimeError, match='inert teardown failure'):
                await pr100_filter.develop(['one'], 1, cpu_only_manager=True)
    asyncio.run(scenario())
    assert backend.shutdowns == 1
    assert_drained(backend)


def test_handled_caller_exception_is_not_grouped_as_refill_failure(backend):
    backend.fail_shutdown = True; backend.barrier = threading.Barrier(4)
    try: raise ValueError('unrelated handled caller exception')
    except ValueError:
        with pytest.raises(BaseExceptionGroup) as raised:
            refill.run(calls(4), lambda *_args: None, 4)
    errors = leaves(raised.value)
    assert not any(isinstance(error, ValueError) for error in errors)
    assert sum(isinstance(error, RuntimeError) and 'teardown' in str(error) for error in errors) == 4
    assert_drained(backend)
