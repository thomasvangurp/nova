import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('activation', '/tmp/pr100-native-throughput-safe-activate.py')
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


def service_factory(unit, events, stop=None, validation=None, fail_install=False):
    class Service:
        def __init__(self, release, path, backup): self.unit = unit
        def validate(self, candidate):
            events.append('validate')
            if validation: validation()
        def stop(self):
            events.append('stop')
            if stop: stop()
        def start(self): events.append('start')
        def install(self, path):
            events.append('install')
            if fail_install and 'new-controller' in str(path): raise RuntimeError('inert install failure')
            unit.write_bytes(path.read_bytes())
        def verify(self): events.append('verify')
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
    result = a.handoff(base, SHA, 'master', release, None, clock_reader=lambda: CLOCK,
        now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == 'already_configured' and not events
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
