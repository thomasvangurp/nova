import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import release_manifest as release
from scripts.check_launch_contract import check
from scripts.shipping_checks import TEST_FILES
from sn68_source import ROOT


def fixture_release(root):
    files = {}
    for name, content in [('pixi.lock', 'locked'), ('nanobody_opt_v2/run_live_producer.py', 'pass\n')]:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'schema': 1, 'git_sha': 'a'*40, 'pixi_lock_hash': files['pixi.lock'],
            'submodules': {'NOVA-nanobody-filter': 'fe2d5868126406a0de8953d26ecd8bdccbdbd029'},
            'files': files, 'runtime_files': sorted(files),
            'validation': {'passed': True, 'tests': 100, 'failures': 0, 'errors': 0,
                           'skipped': 0, 'test_files': list(TEST_FILES)}}


def test_release_accepts_verified_bytes_and_rejects_tampering(tmp_path):
    manifest = fixture_release(tmp_path)
    release.verify(tmp_path, manifest, expected_sha='a'*40)
    (tmp_path/'nanobody_opt_v2/run_live_producer.py').write_text('changed')
    with pytest.raises(ValueError, match='byte mismatch'):
        release.verify(tmp_path, manifest, expected_sha='a'*40)


@pytest.mark.parametrize('runtime_only', [False, True])
@pytest.mark.parametrize('pin', [None, 'b'*40])
def test_release_refuses_missing_or_substituted_filter_pin(tmp_path, runtime_only, pin):
    manifest = fixture_release(tmp_path)
    from nanobody_opt import rules
    assert manifest['submodules']['NOVA-nanobody-filter'] == rules.FILTER_SHA
    manifest['submodules'] = {} if pin is None else {'NOVA-nanobody-filter': pin}
    with pytest.raises(ValueError, match='filter submodule'):
        release.verify(tmp_path, manifest, expected_sha='a'*40, runtime_only=runtime_only)


@pytest.mark.parametrize('kind', ['sha', 'lock', 'runtime', 'zero', 'skip', 'failure', 'missing_suite'])
def test_release_refuses_incomplete_or_wrong_test_evidence(tmp_path, kind):
    manifest = fixture_release(tmp_path)
    if kind == 'sha': manifest['git_sha'] = 'b'*40
    if kind == 'lock': manifest['pixi_lock_hash'] = 'b'*64
    if kind == 'runtime': manifest['runtime_files'].pop()
    if kind == 'zero': manifest['validation']['tests'] = 0
    if kind == 'skip': manifest['validation']['skipped'] = 1
    if kind == 'failure': manifest['validation']['failures'] = 1
    if kind == 'missing_suite': manifest['validation']['test_files'].pop()
    with pytest.raises(ValueError):
        release.verify(tmp_path, manifest, expected_sha='a'*40)


@pytest.mark.parametrize('name', ['../escape', '/absolute', '.env', 'config/.npmrc', 'identity.key'])
def test_release_rejects_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError): release.safe_path(tmp_path, name)


def test_actual_launch_contract_and_mutated_budget(tmp_path):
    check(ROOT)
    path = tmp_path/'deploy/producer/producer_argv.txt'; path.parent.mkdir(parents=True)
    source = (ROOT/'deploy/producer/producer_argv.txt').read_text()
    path.write_text(source.replace('--epoch-candidate-budget\n4800', '--epoch-candidate-budget\n8000'))
    with pytest.raises(ValueError, match='epoch_candidate_budget'): check(tmp_path)


def test_deployment_requires_same_workflow_ci_artifact_and_v2_install():
    workflow = yaml.load((ROOT/'.github/workflows/deploy-nanobody.yml').read_text(), Loader=yaml.BaseLoader)
    assert workflow['jobs']['deploy']['needs'] == 'checks'
    assert workflow['jobs']['checks']['uses'] == './.github/workflows/ci.yml'
    assert 'continue-on-error' not in json.dumps(workflow)
    sync = (ROOT/'ansible/tasks/sync_repo.yml').read_text()
    assert '{name: nanobody_opt_v2}' in sync
    assert sync.index('prepare_tested_release.yml') < sync.index('rsync')
    for name, stop in [('nanobody.yml', '- name: Stop prior autonomous nanobody producer'),
                       ('nanobody_services.yml', 'Restart the checkpoint publisher')]:
        source = (ROOT/'ansible'/name).read_text()
        assert 'tasks/verify_tested_runtime.yml' in source
        assert source.index('tasks/verify_tested_runtime.yml') < source.index(stop)
