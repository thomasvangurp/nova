"""Checkpoints bind numerical runtime settings without recording raw environment."""
import json
from types import SimpleNamespace
import pytest
from nanobody_opt import pr100_live_worker as worker, pr100_native_journal as journal


@pytest.fixture
def identity(monkeypatch):
    context = {'runtime': {'python_binary_sha256': 'a'*64,
        'packages': {k: {'version_sha256': 'b'*64} for k in ('torch', 'numpy', 'openmm')},
        'tools': {k: {'sha256': 'c'*64} for k in ('aggrescan3d-rs', 'freesasa', 'TNP')}},
        'source_sha256': {'adapter': 'd'*64, 'worker': 'e'*64}}
    monkeypatch.setattr(journal, '_context', lambda *a: context)
    monkeypatch.setattr('importlib.util.find_spec', lambda _: SimpleNamespace(origin='/inert/__init__.py'))
    monkeypatch.setattr(worker, 'file_digest', lambda _: 'f'*64)
    return lambda requirements=None, **changes: worker.native_resume_identity(
        {'challenge': {'epoch': 25965}, **changes}, None, requirements or {})


@pytest.mark.parametrize('key', worker.NATIVE_EXECUTION_ENV_KEYS)
def test_scientific_environment_change_invalidates_checkpoint_without_raw_value(identity, monkeypatch, key):
    first = identity(); assert first['resume_eligible'] is True
    marker = 'inert-execution-setting-change'
    monkeypatch.setenv(key, marker)
    second = identity()
    assert first != second
    assert first['execution_environment_sha256'] != second['execution_environment_sha256']
    assert marker not in json.dumps(second)


def test_224_to_one_thread_change_cannot_reuse_old_partial_cpu_evidence(identity, monkeypatch):
    monkeypatch.setenv('OPENMM_CPU_THREADS', '224'); old = identity()
    monkeypatch.setenv('OPENMM_CPU_THREADS', '1'); current = identity()
    assert old != current
    assert len(current['execution_environment_sha256']) == 64
    assert 'OPENMM_CPU_THREADS' not in json.dumps(current)


def test_scheduling_only_release_and_concurrency_changes_remain_portable(identity, monkeypatch):
    before = identity(source_sha='a'*40, workers=32, cpu_slots=40)
    monkeypatch.setenv('PYTHONPATH', '/different-scheduling-release')
    after = identity(source_sha='b'*40, workers=96, cpu_slots=104)
    assert before == after


def test_partial_cpu_checkpoint_is_deferred_after_thread_change(identity, monkeypatch, tmp_path):
    from nanobody_opt import rules, pr100_filter
    from tests.test_pr100_native_screen_integration import CHALLENGE, profile
    requirements = rules.pinned_config(CHALLENGE['epoch'])['nanobody_requirements']
    monkeypatch.setenv('OPENMM_CPU_THREADS', '224')
    old = identity(requirements=requirements, challenge=CHALLENGE)
    row = {'sequence': 'AAAA', 'pr100_annotation': {}, 'native_resume': {
        'schema': 'pr100-native-confirmation-checkpoint-v1', 'sequence_hash': rules.sequence_hash('AAAA'),
        'identity': old, 'measurements': [profile('AAAA')], 'required_cpu_observations': 3}}
    pr100_filter.atomic(tmp_path / 'input.json', {'challenge': CHALLENGE, 'epoch': CHALLENGE['epoch'],
        'target': CHALLENGE['target'], 'archive_sha256': 'a'*64, 'archive_checked_at_unix': 1, 'candidates': [row]})
    monkeypatch.setenv('OPENMM_CPU_THREADS', '1')
    monkeypatch.setattr(pr100_filter, 'helper', lambda _: SimpleNamespace(tool_paths=None))
    result = worker.native_phase({'challenge': CHALLENGE, 'native_tools': '/inert', 'workers': 1, 'repeats': 3},
        tmp_path, measure=lambda *a, **kw: pytest.fail('incompatible partial evidence was resumed'))
    assert result['completed'] == 0 and result['terminal_native_outcomes'] == {}
    assert result['blocked_confirmations'][0]['continuation_blocked'] == 'native_runtime_identity_mismatch'
    assert result['blocked_confirmations'][0]['native_resume'] == row['native_resume']
