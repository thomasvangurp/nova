"""Behavioral bugs 1/2/10 guards against the actual SHIPPING producer module."""
import pytest
from nanobody_opt_v2 import run_live_producer as producer


@pytest.fixture
def pass_context(tmp_path, monkeypatch):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.once = False
    args.budget_mode = 'fixed'
    args.poll_interval = 7
    state = {'epoch': 100, 'block': 100 * 361 + 1, 'epoch_length': 361}
    monkeypatch.setattr(producer, '_chain_state', lambda *a: state)
    monkeypatch.setattr(producer, 'maintain_runtime_storage', lambda *a, **k: {'disk_reserve_violated': False})
    monkeypatch.setattr(producer, 'refresh_prescreen_inputs', lambda *a: None)
    sleeps = []
    monkeypatch.setattr(producer.time, 'sleep', sleeps.append)
    def forbidden(*a, **k):
        pytest.fail('unmocked GPU/subprocess operation')
    monkeypatch.setattr(producer, '_run_iteration', forbidden)
    monkeypatch.setattr(producer.subprocess, 'run', forbidden)
    return args, {}, set(), set(), sleeps


def test_one_pass_system_exit_returns_producer_retry(pass_context, monkeypatch):
    """Bugs 1/10: empty graft/gate SystemExit cannot terminate the producer."""
    args, completed, exhausted, finished, sleeps = pass_context
    def empty_graft(**kwargs):
        raise SystemExit('empty graft fixture')
    monkeypatch.setattr(producer, '_run_iteration', empty_graft)
    assert producer._one_pass(args, completed, exhausted, finished) == producer.PRODUCER_RETRY
    assert completed == {}
    assert sleeps == [7]


def test_disk_reserve_degrade_recovers_and_advances(pass_context, monkeypatch):
    """Bug 2: low disk pass sleeps; recovered pass re-enters work and advances."""
    args, completed, exhausted, finished, sleeps = pass_context
    reports = iter([{'disk_reserve_violated': True}, {'disk_reserve_violated': False}])
    monkeypatch.setattr(producer, 'maintain_runtime_storage', lambda *a, **k: next(reports))
    gpu_calls = []
    monkeypatch.setattr(producer, '_run_iteration', lambda **kwargs: gpu_calls.append(kwargs['iteration']))
    assert producer._one_pass(args, completed, exhausted, finished) == producer.PRODUCER_RETRY
    assert not gpu_calls and not completed
    assert sleeps == [7]
    assert producer._one_pass(args, completed, exhausted, finished) == producer.PRODUCER_ADVANCED
    assert gpu_calls == [1]
    assert completed == {100: 1}
    assert sleeps == [7, 7]  # deterministic spy; no wall-clock deadlines


@pytest.mark.parametrize('existing', [False, True])
def test_atomic_write_leaves_no_partial_file_on_crash(tmp_path, monkeypatch, existing):
    """Bug 4: failed publication cannot leave a truncated or orphan artifact."""
    target = tmp_path / 'result.json'
    if existing:
        target.write_bytes(b'old-complete-artifact')
    def crash(source, destination):
        assert destination == target
        assert source != str(target)
        raise OSError('simulated replacement failure')
    monkeypatch.setattr(producer.os, 'replace', crash)
    with pytest.raises(OSError, match='replacement failure'):
        producer._atomic_bytes(target, b'new-complete-artifact')
    assert sorted(p.name for p in tmp_path.iterdir()) == (['result.json'] if existing else [])
    if existing:
        assert target.read_bytes() == b'old-complete-artifact'


def test_one_pass_preserves_keyboard_interrupt(pass_context, monkeypatch):
    """Bug 1 positive control: recover worker exits without swallowing stop."""
    args, completed, exhausted, finished, sleeps = pass_context
    def stop(**kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(producer, '_run_iteration', stop)
    with pytest.raises(KeyboardInterrupt):
        producer._one_pass(args, completed, exhausted, finished)
    assert sleeps == []
