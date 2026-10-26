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
