import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('activation', Path(__file__).resolve().parents[1] / 'scripts/pr100_safe_activate.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
SHA = 'd' * 40
CLOCK = {'epoch': 25964, 'epoch_length': 361, 'block': 25964*361+100}
HEARTBEAT = {**CLOCK, 'status': 'watching', 'updated_at_unix': 1000}
STATE = {'challenge': {'epoch': 25964}, 'active': None, 'native_rows': ['preserved']}


@pytest.mark.parametrize('change', ['active', 'missing_active', 'uncertain', 'epoch', 'stale', 'running'])
def test_readiness_uses_continuation_ownership_not_legacy_receipt(change):
    clock, heartbeat, state = copy.deepcopy(CLOCK), copy.deepcopy(HEARTBEAT), copy.deepcopy(STATE)
    if change == 'active': state['active'] = {'number': 2, 'phase': 'native'}
    if change == 'missing_active': state.pop('active')
    if change == 'uncertain': state['legacy_uncertain_phase'] = 'native'
    if change == 'epoch': state['challenge']['epoch'] -= 1
    if change == 'stale': heartbeat['updated_at_unix'] -= 31
    if change == 'running': heartbeat['status'] = 'running'
    assert not a.ready_for_activation(clock, heartbeat, state, 1000)


def test_idle_continuation_is_ready_without_legacy_receipt():
    assert a.ready_for_activation(CLOCK, HEARTBEAT, STATE, 1000)


@pytest.mark.parametrize('remaining', [35, 20, 1])
def test_quiet_cutoff_gap_allows_deployment_without_admitting_scientific_work(remaining):
    clock = {**CLOCK, 'block': 25965*361-remaining}
    assert a.ready_for_activation(clock, {**HEARTBEAT, **clock}, STATE, 1000)


@pytest.fixture
def deployment(tmp_path):
    release = tmp_path / 'releases' / SHA; release.mkdir(parents=True)
    unit = tmp_path / 'controller.service'; unit.write_text('original-unit')
    target = release / 'deploy/submit_host' / a.UNIT_NAME
    target.parent.mkdir(parents=True); target.write_text('cwd=<RELEASE_DIR> config=<CONFIG_PATH> python=<CONTROLLER_PYTHON>')
    root = tmp_path / 'state'; root.mkdir()
    a.atomic(root / 'heartbeat.json', HEARTBEAT)
    a.atomic(root / str(CLOCK['epoch']) / 'continuation.json', STATE)
    config = {'source_sha': a.OLD, 'start_epoch': 25964, 'state_root': str(root),
        'publication_path': str(tmp_path / 'publication.json'), 'worker': {'workers': 48, 'native_tools': '/tools'},
        'hosts': {name: {'release': '/root/releases/' + a.OLD, 'annotation_python': '/annotation',
                        'path': '/annotation/bin', 'python_path': '/annotation-shim'} for name in ('master', 'slave')}}
    a.atomic(tmp_path / 'pr100-controller.json', config)
    a.atomic(tmp_path / 'publication.json', {'preserved': True})
    return tmp_path, release, unit, root, config


def service_factory(unit, events, stop=None, validation=None, fail_install=False,
                    start_hook=None, fail_start=False, fail_verify=False):
    class Service:
        def __init__(self, release, path, backup): self.unit = unit
        def validate(self, candidate):
            events.append('validate')
            if validation: validation()
        def stop(self):
            events.append('stop')
            if stop: stop()
        def start(self):
            events.append('start')
            if start_hook: start_hook()
            if fail_start: raise RuntimeError('inert start result ambiguous')
        def install(self, path):
            events.append('install')
            if fail_install and 'new-controller' in str(path): raise RuntimeError('inert install failure')
            unit.write_bytes(path.read_bytes())
        def verify(self):
            events.append('verify')
            if fail_verify: raise RuntimeError('inert verification failed')
    return Service


def test_source_activation_preserves_state_publication_and_annotation(deployment):
    base, release, unit, root, old = deployment; events = []
    state = (root / '25964/continuation.json').read_bytes(); publication = (base / 'publication.json').read_bytes()
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=service_factory(unit, events))
    new = a.read(base / 'pr100-controller.json')
    assert result['status'] == 'activated' and events == ['validate', 'stop', 'install', 'start', 'verify']
    assert new['native_hosts'] == ['master'] and new['native_worker_overrides']['master']['workers'] == 96
    assert new['worker']['workers'] == 48
    assert new['worker']['native_screen']['enabled'] is False
    assert new['hosts']['slave']['path'] == '/annotation/bin'
    assert (root / '25964/continuation.json').read_bytes() == state
    assert (base / 'publication.json').read_bytes() == publication


def test_attempt_started_during_preparation_is_not_stopped(deployment):
    base, release, unit, root, old = deployment; events = []
    def changed(): a.atomic(root / '25964/continuation.json', {**STATE, 'active': {'number': 3}})
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=service_factory(unit, events, validation=changed))
    assert result['status'] == 'waiting_for_clean_attempt' and events == ['validate']
    assert a.read(base / 'pr100-controller.json') == old


def test_stop_race_restarts_old_without_config_or_unit_change(deployment):
    base, release, unit, root, old = deployment; events = []
    def changed(): a.atomic(root / '25964/continuation.json', {**STATE, 'active': {'number': 3}})
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=service_factory(unit, events, stop=changed))
    assert result['status'] == 'boundary_race_old_service_restarted'
    assert events == ['validate', 'stop', 'start']
    assert a.read(base / 'pr100-controller.json') == old and unit.read_text() == 'original-unit'


def test_failed_install_restores_old_configuration_and_unit(deployment):
    base, release, unit, root, old = deployment; events = []
    with pytest.raises(RuntimeError):
        a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
            now=lambda: 1000, service_factory=service_factory(unit, events, fail_install=True))
    assert a.read(base / 'pr100-controller.json') == old and unit.read_text() == 'original-unit'
    assert events[-1] == 'start'


def test_same_release_both_config_preserves_annotation_fields(deployment):
    _, _, _, _, old = deployment
    master = a.proposed_config(old, SHA, CLOCK['epoch'], 'master')
    qualified = {'runtime': {**a.NATIVE, 'native_tools': '/native-tools'}}
    both = a.proposed_config(master, SHA, CLOCK['epoch'], 'both', qualified)
    assert both['source_sha'] == SHA and both['native_hosts'] == ['master', 'slave']
    assert both['hosts']['slave']['annotation_python'] == '/annotation'
    assert both['hosts']['slave']['path'] == '/annotation/bin'
    assert all(both['hosts']['slave'][k] == v for k, v in a.NATIVE.items())


def test_alternate_native_prefix_keeps_annotation_runtime_unchanged(deployment):
    _, _, _, _, old = deployment
    qualified = {'runtime': {**a.NATIVE_ALTERNATE, 'native_tools': a.ALTERNATE_PREFIX + '/bin'},
                 'runtime_identity': {'tools': {'native_python_wrapper': a.NATIVE_WRAPPER_SHA256}}}
    both = a.proposed_config(old, SHA, CLOCK['epoch'], 'both', qualified)
    assert all(both['hosts']['slave'][k] == v for k, v in a.NATIVE_ALTERNATE.items())
    assert both['hosts']['slave']['annotation_python'] == '/annotation'
    assert both['hosts']['slave']['path'] == '/annotation/bin'
    assert both['hosts']['slave']['python_path'] == '/annotation-shim'


def test_unapproved_runtime_prefix_cannot_change_slave_config(deployment):
    _, _, _, _, old = deployment
    qualified = {'runtime': {**a.NATIVE_ALTERNATE, 'native_python': '/unapproved/python', 'native_tools': '/tools'}}
    with pytest.raises(ValueError, match='runtime_layout'):
        a.proposed_config(old, SHA, CLOCK['epoch'], 'both', qualified)


@pytest.mark.parametrize('digest', [None, '0'*64, '84e9dc2dd849d725d478b08ba9656264dc59195b984787b7cb1571ef68cd03fb'])
def test_alternate_wrapper_requires_exact_attested_hash(deployment, digest):
    _, _, _, _, old = deployment
    qualified = {'runtime': {**a.NATIVE_ALTERNATE, 'native_tools': '/native-tools'},
                 'runtime_identity': {'tools': {'native_python_wrapper': digest}}}
    with pytest.raises(ValueError, match='wrapper_identity'):
        a.proposed_config(old, SHA, CLOCK['epoch'], 'both', qualified)


def test_bare_alternate_python_is_no_longer_an_approved_runtime(deployment):
    _, _, _, _, old = deployment
    qualified = {'runtime': {**a.NATIVE_ALTERNATE, 'native_python': a.ALTERNATE_PREFIX + '/bin/python3.12',
                            'native_tools': '/native-tools'}}
    with pytest.raises(ValueError, match='runtime_layout'):
        a.proposed_config(old, SHA, CLOCK['epoch'], 'both', qualified)


def test_cap_code_rollout_accepts036_and_preserves_qualified_slave(deployment):
    _, _, _, _, old = deployment
    qualified = {'runtime': {**a.NATIVE_ALTERNATE, 'native_tools': a.ALTERNATE_PREFIX + '/bin'},
                 'runtime_identity': {'tools': {'native_python_wrapper': a.NATIVE_WRAPPER_SHA256}}}
    both = a.proposed_config(old, a.THROUGHPUT_RELEASE, CLOCK['epoch'], 'both', qualified)
    upgraded = a.proposed_config(both, SHA, CLOCK['epoch'], 'preserve', qualified)
    assert upgraded['native_hosts'] == ['master', 'slave'] and upgraded['source_sha'] == SHA
    assert upgraded['hosts']['slave']['native_python'] == both['hosts']['slave']['native_python']
    assert upgraded['hosts']['slave']['path'] == both['hosts']['slave']['path']
    with pytest.raises(ValueError, match='cannot_disable'):
        a.proposed_config(both, SHA, CLOCK['epoch'], 'master')


def test_cap_code_rollout_accepts036_and_preserves_master_only(deployment):
    _, _, _, _, old = deployment
    master = a.proposed_config(old, a.THROUGHPUT_RELEASE, CLOCK['epoch'], 'master')
    upgraded = a.proposed_config(master, SHA, CLOCK['epoch'], 'preserve')
    assert upgraded['native_hosts'] == ['master'] and upgraded['source_sha'] == SHA


def test_repeated_activation_does_not_restart_or_rewrite_start_epoch(deployment):
    base, release, unit, root, old = deployment; events = []
    configured = a.proposed_config(old, SHA, CLOCK['epoch']-1, 'master')
    a.atomic(base / 'pr100-controller.json', configured)
    unit.write_bytes(a.rendered_unit(release, base / 'pr100-controller.json'))
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == 'already_configured' and events == ['verify']
    assert a.read(base / 'pr100-controller.json')['start_epoch'] == CLOCK['epoch']-1


def test_epoch_rollover_after_stop_restarts_old_service(deployment):
    base, release, unit, root, old = deployment; events = []
    clocks = iter([CLOCK, {**CLOCK, 'epoch': 25965, 'block': 25965*361}])
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: next(clocks),
        now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == 'boundary_race_old_service_restarted'
    assert events == ['validate', 'stop', 'start']
    assert a.read(base / 'pr100-controller.json') == old


def test_lost_clock_after_stop_restarts_old_service(deployment):
    base, release, unit, root, old = deployment; events = []; calls = []
    def clock():
        calls.append(1)
        if len(calls) == 2: raise a.ClockUnavailable('inert unavailable clock')
        return CLOCK
    with pytest.raises(a.ClockUnavailable):
        a.handoff(base, SHA, 'master', release, None, clock_reader=clock, now=lambda: 1000,
                  service_factory=service_factory(unit, events))
    assert events == ['validate', 'stop', 'start']
    assert a.read(base / 'pr100-controller.json') == old


@pytest.fixture
def prerequisites(tmp_path):
    release = tmp_path / 'release'; release.mkdir()
    (release / 'inert.py').write_text('# inert\n')
    manifest = {'inert.py': hashlib.sha256((release / 'inert.py').read_bytes()).hexdigest()}
    stamp = {'git_sha': SHA, 'files': manifest, 'validation': {'passed': True, 'tests': 1000,
        'failures': 0, 'errors': 0, 'skipped': 0,
        'test_files': ['tests/test_pr100_native_throughput.py', 'tests/test_pr100_native_dispatch_recovery.py',
                       'tests/test_pr100_fresh_native_budget.py', 'tests/test_pr100_native_execution_identity.py']}}
    a.atomic(release / 'build_stamp.json', stamp)
    stage_path = tmp_path / 'staging.json'
    stage = {'source_sha': SHA, 'hosts': [{'host': host, 'source_sha': SHA,
        'manifest_files': 1, 'mismatch_files': []} for host in ('pi', 'master', 'slave')]}
    a.atomic(stage_path, stage)
    q_path = tmp_path / 'qualification.json'
    qualified = {'schema': 'pr100-native-runtime-qualification-v1', 'status': 'passed',
        'application_source_sha': a.OLD, 'host': 'slave', 'workers': 32, 'completed_profiles': 32,
        'returncode': 0, 'runtime_inconclusive_count': 0,
        'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True},
        'runtime_manifest_verified': True, 'cpu_affinity_verified': True,
        'runtime': {**a.NATIVE, 'native_tools': '/native-tools'}}
    identity = {'manifest_sha256': '1'*64, 'scientific_sources': {'TNP': '2'*64},
                'model_weights': {str(n): '3'*64 for n in range(4)}, 'tools': {'TNP': '4'*64}}
    qualified.update(runtime_identity=copy.deepcopy(identity), reference_runtime_identity=copy.deepcopy(identity))
    a.atomic(q_path, qualified)
    return release, stage_path, q_path, stamp, stage, qualified


@pytest.mark.parametrize('invalid', ['ci', 'skipped', 'suites', 'manifest', 'stage', 'runtime', 'workers'])
def test_activation_gates_reject_unverified_evidence(prerequisites, invalid):
    release, staging, qualification, stamp, stage, q = prerequisites
    if invalid == 'ci': stamp['validation']['passed'] = False
    if invalid == 'skipped': stamp['validation']['skipped'] = 1
    if invalid == 'suites': stamp['validation']['test_files'] = []
    if invalid == 'manifest': (release / 'inert.py').write_text('# changed')
    if invalid == 'stage': stage['hosts'] = stage['hosts'][:2]
    if invalid == 'runtime': q['runtime_manifest_verified'] = False
    if invalid == 'workers': q['workers'] = 16
    a.atomic(release / 'build_stamp.json', stamp); a.atomic(staging, stage); a.atomic(qualification, q)
    with pytest.raises(ValueError):
        a.verify_prerequisites(SHA, release, staging, mode='both', qualification_path=qualification)


def test_verified_three_host_staging_and_qualification_pass(prerequisites):
    release, staging, qualification, *_ = prerequisites
    q = a.verify_prerequisites(SHA, release, staging, mode='both', qualification_path=qualification)
    assert q['workers'] == 32


def test_runtime_difference_requires_root_approval_bound_to_actual_identity(prerequisites):
    _, _, _, _, _, q = prerequisites
    assert a.qualification_identity_verified(q)
    q['runtime_identity']['manifest_sha256'] = 'b'*64
    assert not a.qualification_identity_verified(q)
    q['runtime_differences'] = ['runtime library package differs']
    q['root_runtime_difference_approval'] = {'approved': True, 'runtime_identity_sha256': hashlib.sha256(
        json.dumps(q['runtime_identity'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    assert a.qualification_identity_verified(q)
    q['runtime_identity']['tools']['TNP'] = 'c'*64
    assert not a.qualification_identity_verified(q)


def test_no_slow_chain_lookup_between_readiness_and_stop(deployment):
    base, release, unit, root, old = deployment; events = []
    def current():
        events.append('clock')
        return CLOCK
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=current,
        now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == 'activated'
    assert events == ['clock', 'validate', 'stop', 'clock', 'install', 'start', 'verify']


@pytest.mark.parametrize('failure', ['start', 'verify'])
def test_poststart_failure_never_rolls_old_config_under_owned_new_work(deployment, failure):
    base, release, unit, root, old = deployment; events = []
    new_active = {**STATE, 'active': {'number': 4, 'phase': 'native', 'native_unresolved': True}}
    def admitted(): a.atomic(root / '25964/continuation.json', new_active)
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=service_factory(unit, events, start_hook=admitted,
            fail_start=failure == 'start', fail_verify=failure == 'verify'))
    assert result['status'] == 'activation_unverified_new_release_preserved'
    assert result['recovery_required'] and result['new_service_start_attempted']
    assert a.read(base / 'pr100-controller.json')['source_sha'] == SHA
    assert unit.read_bytes() == a.rendered_unit(release, base / 'pr100-controller.json')
    assert a.read(root / '25964/continuation.json') == new_active
    assert events.count('stop') == events.count('install') == events.count('start') == 1


def test_same_config_does_not_hide_unverified_service(deployment):
    base, release, unit, root, old = deployment; events = []
    a.atomic(base / 'pr100-controller.json', a.proposed_config(old, SHA, CLOCK['epoch'], 'master'))
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=service_factory(unit, events, fail_verify=True))
    assert result['recovery_required'] and not result['new_service_start_attempted']
    assert events == ['verify']


def test_same_config_does_not_hide_wrong_unit(deployment):
    base, release, unit, root, old = deployment; events = []
    a.atomic(base / 'pr100-controller.json', a.proposed_config(old, SHA, CLOCK['epoch'], 'master'))
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['recovery_required'] and events == ['verify']
    assert unit.read_text() == 'original-unit'


def test_controller_lock_race_restarts_old_without_install(deployment):
    base, release, unit, root, old = deployment; events = []
    with (root / 'controller.lock').open('a') as owner:
        a.fcntl.flock(owner, a.fcntl.LOCK_EX | a.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
                now=lambda: 1000, service_factory=service_factory(unit, events))
    assert events == ['validate', 'stop', 'start']
    assert a.read(base / 'pr100-controller.json') == old


def test_failed_rollback_cannot_restart_with_uncertain_unit_config_pair(deployment):
    base, release, unit, root, old = deployment; events = []
    base_service = service_factory(unit, events)
    class Service(base_service):
        def install(self, path):
            events.append('install')
            raise RuntimeError('inert install and restore failure')
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=Service)
    assert result['status'] == 'rollback_failed_service_stopped' and result['recovery_required']
    assert 'start' not in events and events == ['validate', 'stop', 'install', 'install']


def test_configuration_changed_during_preparation_is_not_stopped(deployment):
    base, release, unit, root, old = deployment; events = []
    changed = {**old, 'operator_revision': 1}
    def update(): a.atomic(base / 'pr100-controller.json', changed)
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=service_factory(unit, events, validation=update))
    assert result['status'] == 'waiting_for_clean_attempt' and events == ['validate']
    assert a.read(base / 'pr100-controller.json') == changed


@pytest.mark.parametrize('name', ['../outside.py', '/absolute.py', '.env', '.env.test', '.ssh/key', 'a.pem'])
def test_manifest_rejects_unsafe_paths_before_reading(prerequisites, name):
    release, staging, qualification, stamp, *_ = prerequisites
    stamp['files'] = {name: 'f' * 64}
    a.atomic(release / 'build_stamp.json', stamp)
    with pytest.raises(ValueError, match='manifest_path'):
        a.verify_prerequisites(SHA, release, staging, mode='master')


def test_rollout_from_qualified224_preserves_both_hosts(deployment):
    _, _, _, _, old = deployment
    q = {'runtime': {**a.NATIVE, 'native_tools': '/native-tools'}}
    current = a.proposed_config(old, a.QUALIFIED_BOTH_RELEASE, CLOCK['epoch'], 'both', q)
    updated = a.proposed_config(current, SHA, CLOCK['epoch'], 'preserve', q)
    assert updated['native_hosts'] == ['master', 'slave']
    assert updated['hosts']['slave']['annotation_python'] == current['hosts']['slave']['annotation_python']


@pytest.mark.parametrize('change', ['missing_cleanup', 'remaining_child', 'unstable_cleanup',
    'inconclusive', 'failed_exit', 'missing_inconclusive', 'boolean_count'])
def test_qualification_requires_clean_completed_native_work(prerequisites, change):
    release, staging, qualification, stamp, stage, q = prerequisites
    if change == 'missing_cleanup': q.pop('cleanup')
    elif change == 'remaining_child': q['cleanup']['remaining_descendants'] = 1
    elif change == 'unstable_cleanup': q['cleanup']['fixed_point_reached'] = False
    elif change == 'inconclusive': q['runtime_inconclusive_count'] = 1
    elif change == 'failed_exit': q['returncode'] = 1
    elif change == 'missing_inconclusive': q.pop('runtime_inconclusive_count')
    else: q['cleanup']['remaining_descendants'] = False
    a.atomic(qualification, q)
    with pytest.raises(ValueError, match='not_qualified'):
        a.verify_prerequisites(SHA, release, staging, mode='both', qualification_path=qualification)


@pytest.mark.parametrize('status', ['activation_unverified_new_release_preserved', 'rollback_failed_service_stopped'])
def test_recovery_required_is_terminal_failure_not_a_retry_or_success(deployment, monkeypatch, capsys, status):
    base, release, unit, root, old = deployment
    stage = base / 'stage.json'; stage.write_text('{}')
    monkeypatch.setattr(a, 'BASE', base)
    monkeypatch.setattr(a, 'verify_prerequisites', lambda *_a, **_k: None)
    monkeypatch.setattr(a, 'verify_novelty_qualification', lambda *_a, **_k: {})
    calls = []
    def handoff(*_a, **_k):
        calls.append(1)
        return {'status': status, 'recovery_required': True, 'source_sha': SHA}
    monkeypatch.setattr(a, 'handoff', handoff)
    monkeypatch.setattr(a.time, 'sleep', lambda *_: pytest.fail('cannot retry unresolved activation'))
    assert a.main([SHA, 'master', '--staging-receipt', str(stage), '--novelty-qualification-receipt', str(stage), '--wait-seconds', '5']) == 1
    assert len(calls) == 1 and json.loads(capsys.readouterr().out)['status'] == status


def test_historical_used_script_and_receipts_remain_exact():
    evidence = Path(__file__).resolve().parents[1] / 'docs/evidence/pr100_activation_20261027'
    manifest = a.read(evidence / 'retention-manifest.json')
    assert hashlib.sha256((evidence / 'activate-used-v2.py').read_bytes()).hexdigest() == manifest['historical_script_sha256']
    assert hashlib.sha256((evidence / 'test-activation-used-v2.py').read_bytes()).hexdigest() == manifest['historical_tests_sha256']
    actual = a.read(evidence / 'activation-verified.json')
    assert actual['status'] == 'activated' and actual['source_sha'] == a.QUALIFIED_BOTH_RELEASE
    assert actual['state_preserved'] and actual['publication_preserved']
    batch = a.read(evidence / 'disjoint-first-batch.json')
    assert batch['shared_inputs'] == 0
    assert [batch['hosts'][host]['fresh_evaluations'] for host in ('master', 'slave')] == [96, 32]


def reseal_novelty(value):
    for item in value['hosts'].values():
        item['artifact_sha256'] = {name: a.digest(item[name]) for name in ('identity', 'result', 'supervisor')}


@pytest.fixture
def novelty_receipt(tmp_path):
    release = tmp_path / 'novelty-release'; release.mkdir()
    sources = {}
    for name in a.NOVELTY_SOURCES:
        path = release / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('qualified scientific source: ' + name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {'schema': 'pr100-qualified-novelty-v1', 'sources': sources,
        'python_binary_sha256': 'a'*64, 'execution_environment_sha256': 'b'*64,
        'annotation_identity': 'c'*64,
        'assets': {name: 'e'*64 for name in ('assets-manifest.json', 'pydantic-assets-manifest.json', 'hmmer-assets-manifest.json')}}
    verdicts = [{'sequence_hash': hashlib.sha256(str(i).encode()).hexdigest(), 'validator_novel': True,
                 'exact_historical': False, 'coarse_candidates': i, 'maximum_identity': 0.9} for i in range(22)]
    item = {'runtime': {**a.NOVELTY_PATHS, 'identity_sha256': a.digest(identity)}, 'identity': identity,
        'result': {'status': 'complete', 'runtime_sha256': a.digest(identity), 'input_count': 22,
            'input_sha256': a.NOVELTY_INPUT_SHA256,
            'snapshots': {name: {'archive_sha256': sha, 'archive_rows': count,
                'archive_cdr_available': count, 'kept_count': 22, 'verdicts': copy.deepcopy(verdicts)}
                for name, (sha, count) in a.NOVELTY_ARCHIVES.items()}},
        'supervisor': {'reason': 'completed', 'exit_code': 0,
            'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}}}
    value = {'schema': 'pr100-novelty-runtime-qualification-v1', 'artifact_hash_encoding': 'canonical-json-v1',
             'hosts': {host: copy.deepcopy(item) for host in ('master', 'slave')}}
    reseal_novelty(value)
    path = tmp_path / 'novelty.json'; a.atomic(path, value)
    return release, path, value


def test_both_completed_novelty_replays_bind_final_scientific_sources(novelty_receipt):
    release, path, value = novelty_receipt
    assert a.verify_novelty_qualification(release, path) == {h: v['runtime'] for h, v in value['hosts'].items()}
    (release / 'nanobody_opt/pr100_novelty_runtime.py').write_text('changed after qualification')
    with pytest.raises(ValueError, match='source_mismatch'):
        a.verify_novelty_qualification(release, path)


@pytest.mark.parametrize('fault', ['missing_host', 'running', 'missing_cdrs', 'incomplete', 'wrong_archive',
    'duplicate', 'changed_cohort', 'failed_verdict', 'nan', 'owned_child', 'failed_cleanup',
    'wrong_runtime', 'wrong_artifact', 'unapproved_path', 'identity_changed', 'unknown_encoding'])
def test_novelty_qualification_rejects_partial_unbound_or_unclean_evidence(novelty_receipt, fault):
    release, path, value = novelty_receipt; item = value['hosts']['slave']
    snapshot = item['result']['snapshots']['new']
    if fault == 'missing_host': del value['hosts']['slave']
    elif fault == 'running': item['result']['status'] = 'running'
    elif fault == 'missing_cdrs': snapshot['archive_cdr_available'] = 0
    elif fault == 'incomplete': snapshot['kept_count'] = 21
    elif fault == 'wrong_archive': snapshot['archive_sha256'] = 'f'*64
    elif fault == 'duplicate': snapshot['verdicts'][1] = copy.deepcopy(snapshot['verdicts'][0])
    elif fault == 'changed_cohort': snapshot['verdicts'][0]['sequence_hash'] = 'f'*64
    elif fault == 'failed_verdict': snapshot['verdicts'][0]['validator_novel'] = False
    elif fault == 'nan': snapshot['verdicts'][0]['maximum_identity'] = 'nan'
    elif fault == 'owned_child': item['supervisor']['cleanup']['remaining_descendants'] = 1
    elif fault == 'failed_cleanup': item['supervisor']['cleanup']['fixed_point_reached'] = False
    elif fault == 'wrong_runtime': item['result']['runtime_sha256'] = 'f'*64
    elif fault == 'unapproved_path': item['runtime']['python'] = '/unqualified/python'
    elif fault == 'identity_changed': item['identity']['execution_environment_sha256'] = 'f'*64
    elif fault == 'unknown_encoding': value['artifact_hash_encoding'] = 'raw-remote-files'
    reseal_novelty(value)
    if fault == 'wrong_artifact': item['artifact_sha256']['result'] = 'f'*64
    a.atomic(path, value)
    with pytest.raises(ValueError): a.verify_novelty_qualification(release, path)


def test_novelty_handoff_preserves_all_native_settings_and_current_work(deployment, novelty_receipt):
    base, release, unit, root, old = deployment; events = []
    old['native_hosts'] = ['master', 'slave']
    old['native_worker_overrides'] = {'master': {'workers': 96}, 'slave': {'workers': 32}}
    old['worker'].update(native_screen={'enabled': False}, repeats=3, candidate_budget_seconds=300)
    old['epoch_fresh_native_limit'] = 4800
    for host in old['hosts'].values(): host.update(native_python='/existing/native', native_tools='/existing/tools', gpus=[0, 1])
    a.atomic(base / 'pr100-controller.json', old)
    native_release, receipt, _ = novelty_receipt
    runtimes = a.verify_novelty_qualification(native_release, receipt)
    prior_state = (root / '25964/continuation.json').read_bytes(); prior_publication = (base / 'publication.json').read_bytes()
    result = a.handoff(base, SHA, 'preserve', release, {'runtime': {'unused': True}}, novelty_runtimes=runtimes,
        clock_reader=lambda: CLOCK, now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == 'activated'
    new = a.read(base / 'pr100-controller.json'); expected = copy.deepcopy(old)
    expected.update(source_sha=SHA, start_epoch=CLOCK['epoch'])
    for host in expected['hosts']:
        expected['hosts'][host]['release'] = '/root/releases/' + SHA
        expected['hosts'][host]['novelty_runtime'] = runtimes[host]
    assert new == expected
    assert (root / '25964/continuation.json').read_bytes() == prior_state
    assert (base / 'publication.json').read_bytes() == prior_publication


def test_novelty_rollout_cannot_also_change_native_topology(deployment, novelty_receipt):
    *_, old = deployment
    release, receipt, _ = novelty_receipt
    with pytest.raises(ValueError, match='native_topology'):
        a.proposed_config(old, SHA, CLOCK['epoch'], 'both', {}, novelty_runtimes=a.verify_novelty_qualification(release, receipt))


def test_cli_requires_explicit_novelty_qualification_before_any_operations(monkeypatch):
    monkeypatch.setattr(a, 'handoff', lambda *_a, **_k: pytest.fail('must not mutate'))
    with pytest.raises(SystemExit) as error:
        a.main([SHA, 'preserve', '--staging-receipt', '/unused'])
    assert error.value.code == 2


def pending_migration(deployment):
    base, release, unit, root, old = deployment
    state = {**STATE, 'active': {'number': 6, 'phase': 'native', 'native_unresolved': True}}
    a.atomic(root / '25964/continuation.json', state)
    directory = root / '25964/attempts/00006'
    a.atomic(directory / 'native-dispatch.json', {'challenge': STATE['challenge'],
        'source_sha': old['source_sha'], 'hosts': ['master']})
    operation = directory / 'novelty-rechecks/operation'
    a.atomic(operation / 'novelty-request.json', {'host': 'master', 'request': {'spec': {'epoch': 25964}}})
    a.atomic(operation / 'novelty-supervisor.json', {'reason': 'completed',
        'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}})
    a.atomic(operation / 'novelty-collected.json', {'files': ['result.json']})
    return directory, operation, state


def test_qualified_migration_can_handoff_clean_pending_prelaunch_failure(deployment, novelty_receipt):
    base, release, unit, root, old = deployment; events = []
    directory, operation, state = pending_migration(deployment)
    native_release, receipt, _ = novelty_receipt
    runtimes = a.verify_novelty_qualification(native_release, receipt)
    before = (root / '25964/continuation.json').read_bytes()
    assert not a.boundary(old, CLOCK, 1000)
    result = a.handoff(base, SHA, 'preserve', release, None, novelty_runtimes=runtimes,
        clock_reader=lambda: CLOCK, now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == 'activated'
    assert (root / '25964/continuation.json').read_bytes() == before
    assert a.read(root / '25964/continuation.json')['active']['native_unresolved'] is True


@pytest.mark.parametrize('fault', ['uncollected', 'owned_native', 'wrong_epoch', 'changed_plan', 'running', 'legacy'])
def test_pending_migration_never_overrides_unproven_ownership(deployment, fault):
    base, release, unit, root, old = deployment
    directory, operation, state = pending_migration(deployment)
    if fault == 'uncollected': (operation / 'novelty-collected.json').unlink()
    elif fault == 'owned_native':
        a.atomic(directory / 'native-hosts/master/native-request.json', {'host': 'master', 'request': {'spec': {'epoch': 25964}}})
    elif fault == 'wrong_epoch':
        a.atomic(operation / 'novelty-request.json', {'host': 'master', 'request': {'spec': {'epoch': 25963}}})
    elif fault == 'changed_plan':
        plan = a.read(directory / 'native-dispatch.json'); plan['source_sha'] = 'f'*40
        a.atomic(directory / 'native-dispatch.json', plan)
    elif fault == 'running': a.atomic(root / 'heartbeat.json', {**HEARTBEAT, 'status': 'running'})
    else: a.atomic(root / '25964/continuation.json', {**state, 'legacy_uncertain_phase': 'native'})
    assert not a.boundary(old, CLOCK, 1000, novelty_migration=True)


def test_pending_migration_rechecks_new_owner_after_stop(deployment, novelty_receipt):
    base, release, unit, root, old = deployment; events = []
    directory, operation, state = pending_migration(deployment)
    native_release, receipt, _ = novelty_receipt
    def stop():
        a.atomic(directory / 'native-hosts/master/native-request.json', {'host': 'master', 'request': {'spec': {'epoch': 25964}}})
    result = a.handoff(base, SHA, 'preserve', release, None,
        novelty_runtimes=a.verify_novelty_qualification(native_release, receipt), clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=service_factory(unit, events, stop=stop))
    assert result['status'] == 'boundary_race_old_service_restarted'
    assert a.read(base / 'pr100-controller.json') == old
    assert 'install' not in events


def test_same_novelty_identity_code_upgrade_cannot_claim_pending_migration_exception(deployment, novelty_receipt):
    base, release, unit, root, old = deployment; events = []
    pending_migration(deployment)
    native_release, receipt, _ = novelty_receipt
    runtimes = a.verify_novelty_qualification(native_release, receipt)
    for host in old['hosts']: old['hosts'][host]['novelty_runtime'] = runtimes[host]
    a.atomic(base / 'pr100-controller.json', old)
    result = a.handoff(base, SHA, 'preserve', release, None, novelty_runtimes=runtimes,
        clock_reader=lambda: CLOCK, now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == 'waiting_for_clean_attempt'
    assert events == [] and a.read(base / 'pr100-controller.json') == old


@pytest.mark.parametrize('predecessor', [a.QUALIFIED_NOVELTY_RELEASE, a.ARCHIVE_CACHE_RELEASE,
                                        a.TERMINAL_INVALIDATION_RELEASE])
def test_qualified_novelty_release_is_approved_predecessor(deployment, novelty_receipt, predecessor):
    *_, old = deployment
    old['source_sha'] = predecessor
    for host in old['hosts'].values(): host['release'] = '/root/releases/' + predecessor
    release, receipt, _ = novelty_receipt
    value = a.proposed_config(old, SHA, CLOCK['epoch'], 'preserve', novelty_runtimes=a.verify_novelty_qualification(release, receipt))
    assert value['source_sha'] == SHA


@pytest.mark.parametrize('enabled', [False, True])
def test_interleave_activation_is_explicit_and_changes_only_one_worker_field(deployment, novelty_receipt, enabled):
    *_, old = deployment
    release, receipt, _ = novelty_receipt
    runtimes = a.verify_novelty_qualification(release, receipt)
    unchanged = a.proposed_config(old, SHA, CLOCK['epoch'], 'preserve', novelty_runtimes=runtimes)
    value = a.proposed_config(old, SHA, CLOCK['epoch'], 'preserve', novelty_runtimes=runtimes,
                              interleave_confirmations=enabled)
    if enabled: unchanged['worker']['native_interleave_confirmations'] = True
    assert value == unchanged
    assert 'native_interleave_confirmations' not in old['worker']


def test_interleave_optin_refuses_accelerated_screen(deployment, novelty_receipt):
    *_, old = deployment
    old['worker']['native_screen'] = {'enabled': True}
    release, receipt, _ = novelty_receipt
    with pytest.raises(ValueError, match='original_cpu'):
        a.proposed_config(old, SHA, CLOCK['epoch'], 'preserve',
            novelty_runtimes=a.verify_novelty_qualification(release, receipt), interleave_confirmations=True)


def test_slave_refill_and_donors_preserve_other_qualified_settings(deployment, novelty_receipt):
    *_, old = deployment
    old['source_sha'] = a.TERMINAL_INVALIDATION_RELEASE
    for host in old['hosts'].values():
        host['release'] = '/root/releases/' + old['source_sha']
    old.update(native_hosts=['master', 'slave'], epoch_fresh_native_limit=4800,
               native_worker_overrides={'master': {'workers': 96, 'max_workers': 96, 'cpu_slots': 104},
                                        'slave': {'workers': 32, 'max_workers': 32, 'cpu_slots': 32}})
    old['worker'].update(native_interleave_confirmations=True, native_screen={'enabled': False})
    release, receipt, _ = novelty_receipt
    runtimes = a.verify_novelty_qualification(release, receipt)
    baseline = a.proposed_config(old, SHA, CLOCK['epoch'], 'preserve', {}, novelty_runtimes=runtimes)
    canary = a.proposed_config(old, SHA, CLOCK['epoch'], 'preserve', {}, novelty_runtimes=runtimes,
                               native_refill_hosts=['slave'], missing_cell_donors=True)
    baseline['native_worker_overrides']['slave']['native_work_conserving'] = True
    baseline['worker']['missing_cell_donor_fallback'] = True
    assert canary == baseline
    assert 'native_work_conserving' not in old['native_worker_overrides']['slave']
    assert 'missing_cell_donor_fallback' not in old['worker']


@pytest.mark.parametrize('hosts', ['slave', ['unknown'], ['slave', 'slave'], [True]])
def test_refill_host_selection_rejects_ambiguous_requests(hosts):
    with pytest.raises(ValueError, match='invalid_native_refill_hosts'):
        a.rollout_options({'worker': {}}, native_refill_hosts=hosts)


@pytest.mark.parametrize('interleave,screen,hosts,match', [
    (False, False, ['master'], 'original_cpu'),
    (True, True, ['master'], 'original_cpu'),
    (True, False, ['slave'], 'host_not_enabled'),
])
def test_refill_requires_existing_native_ownership(interleave, screen, hosts, match):
    value = {'native_hosts': ['master'], 'worker': {
        'native_interleave_confirmations': interleave, 'native_screen': {'enabled': screen}}}
    with pytest.raises(ValueError, match=match):
        a.rollout_options(value, native_refill_hosts=hosts)


def test_donor_activation_requires_boolean():
    with pytest.raises(ValueError, match='must_be_boolean'):
        a.rollout_options({'worker': {}}, missing_cell_donors=1)


@pytest.mark.parametrize('expired', [False, True])
def test_refill_adapter_cutover_preserves_live_partial_confirmations(deployment, expired):
    base, release, unit, root, old = deployment
    events = []
    old['worker']['native_interleave_confirmations'] = True
    a.atomic(base / 'pr100-controller.json', old)
    state = {**STATE, 'pending_confirmation': [{'sequence_hash': 'retained-inert-checkpoint'}]}
    a.atomic(root / str(CLOCK['epoch']) / 'continuation.json', state)
    clock = {**CLOCK, 'block': (CLOCK['epoch'] + 1) * CLOCK['epoch_length'] - 20} if expired else CLOCK
    a.atomic(root / 'heartbeat.json', {**HEARTBEAT, **clock})
    before = (root / str(CLOCK['epoch']) / 'continuation.json').read_bytes()
    result = a.handoff(base, SHA, 'master', release, None, native_refill_hosts=['master'],
        clock_reader=lambda: clock, now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == ('activated' if expired else 'waiting_for_clean_attempt')
    assert (root / str(CLOCK['epoch']) / 'continuation.json').read_bytes() == before
    if not expired:
        assert events == [] and a.read(base / 'pr100-controller.json') == old


@pytest.mark.parametrize('feature', ['refill', 'donors'])
@pytest.mark.parametrize('listed,staged', [(False, False), (True, False), (False, True), (True, True)])
def test_rollout_requires_executed_and_staged_feature_suites(prerequisites, feature, listed, staged):
    release, stage_path, qualification, stamp, stage, _ = prerequisites
    suites = {'tests/test_pr100_safe_activate.py'}
    kwargs = {'mode': 'both', 'qualification_path': qualification}
    if feature == 'refill':
        suites.update({'tests/test_pr100_native_refill.py', 'tests/test_pr100_native_lifecycle.py'})
        kwargs['native_refill_hosts'] = ['slave']
    else:
        suites.add('tests/test_pr100_reference_donors.py')
        kwargs['missing_cell_donors'] = True
    if listed:
        stamp['validation']['test_files'].extend(sorted(suites))
    if staged:
        for name in suites:
            path = release / name; path.parent.mkdir(exist_ok=True)
            path.write_text('# inert rollout suite fixture')
            stamp['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for host in stage['hosts']:
            host['manifest_files'] = len(stamp['files'])
    a.atomic(release / 'build_stamp.json', stamp); a.atomic(stage_path, stage)
    if listed and staged:
        assert a.verify_prerequisites(SHA, release, stage_path, **kwargs)
    else:
        with pytest.raises(ValueError, match='shipping_tests_missing|feature_suites_not_staged'):
            a.verify_prerequisites(SHA, release, stage_path, **kwargs)


@pytest.mark.parametrize('listed,staged', [(False, False), (True, False), (False, True), (True, True)])
def test_interleave_requires_tested_and_hash_bound_deadline_suite(prerequisites, listed, staged):
    release, stage_path, qualification, stamp, stage, _ = prerequisites
    suite = 'tests/test_pr100_confirmation_deadline_review.py'
    if listed: stamp['validation']['test_files'].append(suite)
    if staged:
        path = release / suite; path.parent.mkdir(); path.write_text('# inert deadline regression fixture')
        stamp['files'][suite] = hashlib.sha256(path.read_bytes()).hexdigest()
        for host in stage['hosts']: host['manifest_files'] += 1
    a.atomic(release / 'build_stamp.json', stamp); a.atomic(stage_path, stage)
    if listed and staged:
        a.verify_prerequisites(SHA, release, stage_path, mode='master', interleave_confirmations=True)
    else:
        with pytest.raises(ValueError):
            a.verify_prerequisites(SHA, release, stage_path, mode='master', interleave_confirmations=True)


def test_cli_forwards_optin_to_both_shipping_gate_and_handoff(deployment, monkeypatch, capsys):
    base, *_ = deployment; stage = base / 'stage.json'; stage.write_text('{}'); calls = []
    monkeypatch.setattr(a, 'BASE', base)
    def verify(*args, **kwargs): calls.append(('verify', kwargs['interleave_confirmations']))
    def handoff(*args, **kwargs):
        calls.append(('handoff', kwargs['interleave_confirmations'])); return {'status': 'activated'}
    monkeypatch.setattr(a, 'verify_prerequisites', verify)
    monkeypatch.setattr(a, 'verify_novelty_qualification', lambda *_a: {})
    monkeypatch.setattr(a, 'handoff', handoff)
    assert a.main([SHA, 'preserve', '--staging-receipt', str(stage),
        '--novelty-qualification-receipt', str(stage), '--interleave-confirmations']) == 0
    assert calls == [('verify', True), ('handoff', True)]
