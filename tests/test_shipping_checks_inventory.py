"""Release gates include changed suites and test the actual submit-host layout."""
from pathlib import Path

import pytest

from scripts import shipping_checks as checks


def test_a_new_suite_cannot_be_omitted_from_release_checks():
    with pytest.raises(ValueError, match='changed regression suites omitted'):
        checks.validate_test_inventory(['tests/test_future_regression.py'])
    checks.validate_test_inventory(['tests/test_future_regression.py', 'tests/fixtures/values.json'],
                                   suites=['tests/test_future_regression.py'])


def test_required_suite_duplicates_are_refused():
    with pytest.raises(ValueError, match='duplicate'):
        checks.validate_test_inventory([], suites=['tests/test_a.py', 'tests/test_a.py'])


def test_submit_host_layout_uses_exact_packaged_bytes(tmp_path):
    source, target = tmp_path / 'release', tmp_path / 'pi'
    names = ('deploy/submit_host/pi_submit.py', 'nanobody_opt/rules.py',
             'nanobody_opt/commit_submission.py', 'nanobody_opt/pr100_filter.py',
             'nanobody_opt/publication_gates.py', 'compound_miner/nanobody_provider.py',
             'config/nanobody_pr100.json')
    for name in names:
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    checks.stage_submit_host(source, target, names)
    for name in names:
        installed = 'pi_submit.py' if name == 'deploy/submit_host/pi_submit.py' else name
        assert (target / installed).read_bytes() == (source / name).read_bytes()
    assert not (target / 'deploy').exists()


def test_incomplete_submit_host_runtime_is_refused(tmp_path):
    with pytest.raises(ValueError, match='incomplete'):
        checks.stage_submit_host(tmp_path / 'release', tmp_path / 'pi', [])
