"""Execution-framework tests using the pinned async service, with inert profiles.

No models, biological calculations or remote resources are used. The rendezvous
test exercises the actual upstream to_thread/semaphore method: a default pool
smaller than the advertised profile concurrency cannot complete the rendezvous.
"""
import ast
import asyncio
from pathlib import Path
import threading
import types

import pytest

from nanobody_opt import pr100_filter, pr100_live_worker


def upstream_service(profile, width):
    path = Path(__file__).resolve().parents[1] / 'NOVA-nanobody-filter/metanano/services/developability_service.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DevelopabilityService')
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == 'compute_tnp_profile_async')
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None

    class Execution:
        def __init__(self, timeout):
            self.timeout = timeout

        def remaining(self):
            return self.timeout

        def cancel(self):
            pass

    scope = {'asyncio': asyncio, 'ProfileExecution': Execution}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(path), 'exec'), scope)

    class Manager:
        task_timeout = 10

        def __init__(self):
            self.tnp_semaphore = asyncio.Semaphore(width)

        async def initialize(self):
            pass

    return types.SimpleNamespace(
        manager=Manager(), _filter=types.SimpleNamespace(compute_tnp_profile=profile)), scope[method.name]


def test_every_admitted_upstream_profile_can_start_without_a_second_queue():
    width = 48
    rendezvous = threading.Barrier(width, timeout=5)
    threads = set()
    lock = threading.Lock()

    def profile(value, *, execution):
        with lock:
            threads.add(threading.current_thread())
        rendezvous.wait()
        return {'input': value, 'unchanged_payload': [1.25, 'recorded-result']}

    async def backend(values, workers):
        service, method = upstream_service(profile, workers)
        return await asyncio.gather(*(method(service, value) for value in values))

    inputs = [str(i) for i in range(width)]
    rows = pr100_filter.run_develop(inputs, width, backend=backend)
    assert [row['input'] for row in rows] == inputs
    assert all(row['unchanged_payload'] == [1.25, 'recorded-result'] for row in rows)
    assert len(threads) == width
    assert all(not thread.is_alive() for thread in threads)


def test_executor_and_loop_are_owned_without_changing_the_callers_loop():
    seen = []

    async def backend(values, workers):
        seen.append(asyncio.get_running_loop())
        return await asyncio.gather(*(asyncio.to_thread(lambda v=v: v) for v in values))

    async def caller():
        outer = asyncio.get_running_loop()
        result = await asyncio.to_thread(pr100_filter.run_develop, ['x'], 48, backend=backend)
        assert result == ['x']
        assert asyncio.get_running_loop() is outer and seen[0] is not outer
        assert seen[0].is_closed()
        return await asyncio.to_thread(lambda: 'outer executor still usable')

    assert asyncio.run(caller()) == 'outer executor still usable'


def test_backend_error_propagates_after_owned_threads_finish():
    threads = []

    def fail():
        threads.append(threading.current_thread())
        raise RuntimeError('inert backend failure')

    async def backend(values, workers):
        return await asyncio.to_thread(fail)

    with pytest.raises(RuntimeError, match='inert backend failure'):
        pr100_filter.run_develop(['x'], 48, backend=backend)
    assert threads and all(not thread.is_alive() for thread in threads)


def test_repeated_measurements_are_fresh_backend_calls_with_no_cache():
    observations = []

    async def backend(values, workers):
        observations.append(tuple(values))
        return [{'observation': len(observations)} for _ in values]

    first, _, _ = pr100_live_worker.measure_batch(['x'], 48, develop=backend)
    second, _, _ = pr100_live_worker.measure_batch(['x'], 48, develop=backend)
    assert observations == [('x',), ('x',)]
    assert first == [{'observation': 1}] and second == [{'observation': 2}]


@pytest.mark.parametrize('width', [0, -1, True, 2.5])
def test_invalid_width_never_calls_backend(width):
    async def backend(*args):
        raise AssertionError('must not run')

    with pytest.raises(ValueError, match='positive integer'):
        pr100_filter.run_develop(['x'], width, backend=backend)


def test_empty_batch_still_preserves_backend_contract():
    async def backend(values, width):
        assert values == [] and width == 48
        return []

    assert pr100_filter.run_develop([], 48, backend=backend) == []
