import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import pytest

spec = importlib.util.spec_from_file_location('activation', Path(__file__).resolve().parents[1] / 'scripts/pr100_safe_activate.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
SHA = 'd' * 40
CLOCK = {'epoch': 25964, 'epoch_length': 361, 'block': 25964*361+100}
HEARTBEAT = {**CLOCK, 'status': 'watching', 'updated_at_unix': 1000}
STATE = {'challenge': {'epoch': 25964}, 'active': None, 'native_rows': ['preserved']}


@pytest.mark.parametrize('fault', [None, 'unqualified', 'proof_changed', 'resident_changed',
                                 'policy_changed', 'loader_changed', 'unknown', 'cpu_only'])
def test_explicit_resident_upgrade_preserves_runtime_and_allocations(tmp_path, fault):
    # Synthetic activation fixture; this is not scientific qualification evidence.
    source = Path(__file__).resolve().parents[1]
    old = next(r for r in a.read(source/'config/pr100_native_reference.json')['registrations']
               if r['policy']['final_refinement']['platform'] == 'CUDA')
    replacement = copy.deepcopy(old)
    module = tmp_path/'nanobody_opt/pr100_registered_resident.py'
    module.parent.mkdir();module.write_text('synthetic resident fixture')
    proof = tmp_path/'config/proof.json';proof.parent.mkdir();proof.write_text('{"fixture":true}')
    replacement.update(name='synthetic-replacement',
        resident_module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
        qualification_report_file='config/proof.json',
        qualification_report_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
    if fault == 'unqualified':replacement['qualification_status']='research_unqualified'
    if fault == 'proof_changed':proof.write_text('{}')
    if fault == 'resident_changed':module.write_text('different bytes')
    if fault == 'policy_changed':replacement['policy']['preparation_threads']=2
    if fault == 'loader_changed':replacement['cuda_plugin']['loader']['nvrtc_sha256']='f'*64
    a.atomic(tmp_path/'config/pr100_native_reference.json', {'registrations':[old,replacement]})
    settings={'enabled':True,'resident':True,'registration_sha256':a.digest(old),
              'workers_per_gpu':4,'gpus':[0,1],'resident_batch_draws':16}
    config={'native_hosts':['master','slave'], 'worker':{},
            'native_worker_overrides':{'master':{'native_reference':settings},'slave':{'cpu_slots':32}}}
    if fault == 'cpu_only':settings['enabled']=False
    original=copy.deepcopy(config)
    identity='f'*64 if fault == 'unknown' else a.digest(replacement)
    if fault:
        with pytest.raises(ValueError):a.registered_native_rollout(config,tmp_path,identity)
    else:
        result=a.registered_native_rollout(config,tmp_path,identity)
        expected=copy.deepcopy(config)
        expected['native_worker_overrides']['master']['native_reference']['registration_sha256']=identity
        assert result==expected
    assert config==original


@pytest.fixture
def cuda_slave_rollout(tmp_path,monkeypatch):
    """Synthetic proof exercises the guard; real canaries are separate artifacts."""
    source=Path(__file__).resolve().parents[1]
    files=set(a.CUDA_HOST_SOURCE_FILES)|{'config/pr100_native_reference.json'}
    for name in files:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,target)
    rows=a.read(tmp_path/'config/pr100_native_reference.json')['registrations']
    resident_hash=hashlib.sha256((tmp_path/'nanobody_opt/pr100_registered_resident.py').read_bytes()).hexdigest()
    row=next(r for r in rows if r.get('resident_module_sha256')==resident_hash)
    proof_file=tmp_path/row['qualification_report_file']
    shutil.copy2(source/row['qualification_report_file'],proof_file)
    identity=a.digest(row)
    settings={'enabled':True,'resident':True,'registration_sha256':identity,
        'gpus':list(range(8)),'workers_per_gpu':4,'resident_batch_draws':16,
        'environment':'/sealed/env','bundle':'/sealed/bundle','cuda_plugin_root':'/sealed/plugin'}
    config={'native_hosts':['master','slave'],'worker':{'native_interleave_confirmations':True},
        'native_gpu_by_host':{'master':list(range(8)),'slave':[]},
        'native_worker_overrides':{
            'master':{'cpu_slots':48,'workers':32,'max_workers':32,'native_reference':settings,
                      'native_fleet_policy':{'kind':'pr100-observed-fleet-v1','minimum_smoothed_fraction':.4}},
            'slave':{'cpu_slots':32,'workers':32,'max_workers':32,'native_work_conserving':True,
                     'native_screen':{'enabled':False,'backend':'resident_cpu'}}}}
    proof={'schema':'pr100-cuda-native-host-qualification-v1','host':'slave','status':'complete',
        'gpus':[0,1,2,3],'workers_per_gpu':4,'resident_batch_draws':8,'cpu_slots':32,
        'registration_sha256':identity,'source_files_sha256':{
            name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in a.CUDA_HOST_SOURCE_FILES},
        'canaries':[]}
    from nanobody_opt import pr100_native_reference as native
    monkeypatch.setattr(native,'authoritative_measurement',lambda value:value.get('synthetic_valid') is True)
    for gpu in range(4):
        groups=[{'profiles':8,'worker_pids':[100+gpu*4+i for i in range(4)],
            'genuine_context_policy_verified':True,'model_postchecks_passed':True,
            'draw_ids':[f'/synthetic/gpu-{gpu}/group-{group}/{i}' for i in range(8)]} for group in range(2)]
        result={'status':'complete','source_sha':'e'*40,'complete_draws':16,
            'fresh_prediction_events':16,'unique_draw_ids':16,'same_four_worker_pids':True,
            'same_coordinator_session':True,'distinct_batch_UUIDs':2,'errors':[],
            'cleanup':{'remaining_descendants':0,'fixed_point_reached':True},'groups':groups,
            'final_refinement':row['policy']['final_refinement']}
        request={'source_sha':'e'*40,'gpu':gpu,'final_refinement':row['policy']['final_refinement'],
            'native_reference':{**settings,'gpus':[gpu],'gpu_allowlist':[gpu],'resident_batch_draws':8}}
        manifest={'schema':'pr100-integrated-resident-canary-manifest-v1','ok':True,
            'source_sha':'e'*40,'registration_sha256':identity,'files_sha256':{k:'a'*64 for k in
                ('request.json','result/result.json','result/measurements.json','leases')}}
        canary={'gpu':gpu,'request':request,'result':result,'manifest':manifest,
            'lease_summaries':[{'gpu':gpu,'status':'process_finished','released':True,'child_exit_code':0,
                'cleanup':{'remaining_descendants':0,'fixed_point_reached':True}}]}
        canary['raw_artifacts']={'request.json':json.dumps(request),'result/result.json':json.dumps(result),
            'leases':json.dumps(canary['lease_summaries'])}
        for name,text in canary['raw_artifacts'].items():manifest['files_sha256'][name]=hashlib.sha256(text.encode()).hexdigest()
        measurements={'measurements':[{'synthetic_valid':True,'native_reference':{
            'registration_sha256':identity,'receipt':{'source_sha':'e'*40,'request':{'draw_id':draw},
                'trace':[{'event':'context_created','role':'sidechain_repair','platform':'CUDA'}]}}}
            for group in groups for draw in group['draw_ids']]}
        raw=json.dumps(measurements).encode()
        canary.update(measurement_artifact_gzip=a.base64.b64encode(a.gzip.compress(raw)).decode(),
            measurement_artifact_uncompressed_bytes=len(raw),measurement_artifact_encoding='base64-gzip-v1',
            context_inventory=[{'role':'sidechain_repair','platform':'CUDA','count':16}])
        manifest['files_sha256']['result/measurements.json']=hashlib.sha256(raw).hexdigest()
        proof['canaries'].append(canary)
    return config,proof,tmp_path


def test_qualified_all_device_rollout_uses_gpu_without_expanding_cpu_allocation(cuda_slave_rollout):
    config,proof,release=cuda_slave_rollout;before=copy.deepcopy(config)
    result=a.cuda_native_host_rollout(config,release,proof)
    slave=result['native_worker_overrides']['slave']
    assert result['native_gpu_by_host']['slave']==[0,1,2,3]
    assert slave['cpu_slots']==slave['workers']==slave['max_workers']==32
    assert slave['native_reference']['enabled'] is True and slave['native_reference']['resident'] is True
    assert slave['native_reference']['gpus']==[0,1,2,3]
    assert slave['native_reference']['workers_per_gpu']==4
    assert slave['native_reference']['resident_batch_draws']==8
    assert slave['native_screen']=={'enabled':False}
    assert slave['native_fleet_policy']==before['native_worker_overrides']['master']['native_fleet_policy']
    assert result['native_worker_overrides']['master']==before['native_worker_overrides']['master']
    assert config==before


@pytest.mark.parametrize('fault',['missing_gpu','source_changed','registration','incomplete','dirty',
    'retired_error','different_models','duplicate_draw','untested_queue','expanded_cpu','disabled_reference',
    'digest_changed','inventory_changed','invalid_scientific_measurement'])
def test_gpu_host_rollout_refuses_incomplete_or_changed_qualification(cuda_slave_rollout,fault):
    config,proof,release=cuda_slave_rollout;canary=proof['canaries'][0]
    if fault=='missing_gpu':proof['canaries'].pop()
    if fault=='source_changed':(release/'nanobody_opt/pr100_registered_resident.py').write_text('changed')
    if fault=='registration':proof['registration_sha256']='f'*64
    if fault=='incomplete':canary['result']['complete_draws']=15
    if fault=='dirty':canary['lease_summaries'][0]['cleanup']['remaining_descendants']=1
    if fault=='retired_error':canary['result']['errors']=[{'error_type':'RuntimeError'}]
    if fault=='different_models':canary['result']['groups'][1]['worker_pids']=[1,2,3,4]
    if fault=='duplicate_draw':canary['result']['groups'][1]['draw_ids']=canary['result']['groups'][0]['draw_ids']
    if fault=='untested_queue':proof['resident_batch_draws']=16
    if fault=='expanded_cpu':proof['cpu_slots']=64
    if fault=='disabled_reference':canary['request']['native_reference']['enabled']=False
    if fault=='digest_changed':canary['manifest']['files_sha256']['leases']='f'*64
    if fault=='inventory_changed':canary['context_inventory'][0]['count']=15
    if fault=='invalid_scientific_measurement':
        packed=a.base64.b64decode(canary['measurement_artifact_gzip'])
        raw=a.gzip.decompress(packed);values=json.loads(raw)
        values['measurements'][0]['synthetic_valid']=False
        raw=json.dumps(values).encode()
        canary['measurement_artifact_gzip']=a.base64.b64encode(a.gzip.compress(raw)).decode()
        canary['measurement_artifact_uncompressed_bytes']=len(raw)
        canary['manifest']['files_sha256']['result/measurements.json']=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):a.cuda_native_host_rollout(config,release,proof)


def test_cli_passes_explicit_cuda_host_proof_to_guarded_handoff(deployment,monkeypatch):
    base,*_=deployment;stage=base/'stage.json';stage.write_text('{}')
    receipt=base/'cuda-host.json';proof={'schema':'synthetic-cuda-host-fixture'};a.atomic(receipt,proof)
    monkeypatch.setattr(a,'BASE',base)
    monkeypatch.setattr(a,'verify_prerequisites',lambda *args,**kwargs:None)
    monkeypatch.setattr(a,'verify_novelty_qualification',lambda *args,**kwargs:{})
    calls=[]
    def handoff(*args,**kwargs):
        calls.append(kwargs['cuda_native_host_receipt']);return {'status':'activated'}
    monkeypatch.setattr(a,'handoff',handoff)
    assert a.main([SHA,'preserve','--staging-receipt',str(stage),
        '--novelty-qualification-receipt',str(stage),'--cuda-native-host-receipt',str(receipt)])==0
    assert calls==[proof]


def test_activation_preserves_service_config_permissions_before_validation(deployment):
    base,release,unit,root,_=deployment
    path=base/'pr100-controller.json';path.chmod(0o640)
    previous=path.stat();observed=[];events=[]
    def validate():
        candidate=base/'pr100-staging'/('throughput-candidate-'+SHA+'-master.json')
        now=candidate.stat()
        observed.append((now.st_uid,now.st_gid,now.st_mode&0o777))
    result=a.handoff(base,SHA,'master',release,None,clock_reader=lambda:CLOCK,
        now=lambda:1000,service_factory=service_factory(unit,events,validation=validate))
    assert result['status']=='activated'
    assert observed==[(previous.st_uid,previous.st_gid,0o640)]
    current=path.stat()
    assert (current.st_uid,current.st_gid,current.st_mode&0o777)==observed[0]


@pytest.mark.parametrize('launcher_uid',[1000,0])
def test_privileged_activation_validates_as_actual_service_user(tmp_path,monkeypatch,launcher_uid):
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setattr(a.os,'geteuid',lambda:launcher_uid)
    monkeypatch.setattr(a.pwd,'getpwnam',lambda _:SimpleNamespace(pw_uid=1000))
    monkeypatch.setattr(a,'command',lambda args,**kwargs:calls.append(args))
    monkeypatch.setattr(a,'verify_consumer_reference',lambda *args:None)
    candidate=tmp_path/'config.json';a.atomic(candidate,{})
    service=a.Service(tmp_path,candidate,None)
    service.validate(candidate)
    expected=[] if launcher_uid==1000 else ['sudo','-n','-u','thomasg']
    assert calls[0][:len(expected)]==expected
    assert calls[0][len(expected)]==a.PYTHON


@pytest.mark.parametrize('fault', [None, 'missing_cuda_registration', 'old_verifier'])
def test_actual_submitter_must_accept_selected_cuda_registration(tmp_path, monkeypatch, fault):
    release = Path(__file__).resolve().parents[1]
    consumer = tmp_path / 'consumer'
    files = ('nanobody_opt/pr100_native_reference.py',
             'nanobody_opt/pr100_validator_reference.py',
             'config/pr100_validator_reference.json', 'config/pr100_native_reference.json')
    for name in files:
        target = consumer / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(release / name, target)
    (consumer / 'nanobody_opt/__init__.py').write_text('')
    registry_path = consumer / 'config/pr100_native_reference.json'
    registry = a.read(registry_path)
    row = next(r for r in registry['registrations']
               if r['policy']['final_refinement']['platform'] == 'CUDA')
    identity = a.digest(row)
    config = {'native_hosts': ['master', 'slave'], 'worker': {},
              'native_worker_overrides': {'master': {'native_reference': {
                  'enabled': True, 'registration_sha256': identity}}}}
    if fault == 'missing_cuda_registration':
        registry['registrations'].remove(row)
        a.atomic(registry_path, registry)
    elif fault == 'old_verifier':
        verifier = consumer / 'nanobody_opt/pr100_validator_reference.py'
        verifier.write_text(verifier.read_text().replace(
            "set(plugin) in (required, required | {'loader'})", 'set(plugin) == required'))
    monkeypatch.setattr(a, 'PYTHON', sys.executable)
    if fault:
        with pytest.raises(ValueError, match='consumer_(native_registration_missing|reference_source_mismatch)'):
            a.verify_consumer_reference(config, release, consumer)
    else:
        a.verify_consumer_reference(config, release, consumer)


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
    assert new['native_hosts'] == ['master']
    assert new['worker']['workers'] == 48
    assert 'native_screen' not in new['worker']
    assert 'native_worker_overrides' not in new
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


@pytest.mark.parametrize('stage', ['before', 'prepare', 'stop'])
def test_force_never_bypasses_active_attempt_ownership(deployment, stage):
    base, release, unit, root, old = deployment
    events = []
    def activate_attempt():
        a.atomic(root / '25964/continuation.json', {**STATE, 'active': {'number': 9, 'phase': 'native'}})
    if stage == 'before':
        activate_attempt()
    result = a.handoff(base, SHA, 'master', release, None, force=True,
        clock_reader=lambda: CLOCK, now=lambda: 1000,
        service_factory=service_factory(unit, events,
            validation=activate_attempt if stage == 'prepare' else None,
            stop=activate_attempt if stage == 'stop' else None))
    assert result['status'] == ('boundary_race_old_service_restarted' if stage == 'stop'
                                else 'waiting_for_clean_attempt')
    assert a.read(base / 'pr100-controller.json') == old
    assert unit.read_text() == 'original-unit'
    assert 'install' not in events


def test_force_never_bypasses_unqualified_predecessor(deployment):
    *_, old = deployment
    old['source_sha'] = 'f' * 40
    with pytest.raises(ValueError, match='unexpected_current_release'):
        a.proposed_config(old, SHA, CLOCK['epoch'], 'master', force=True)


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
                                        a.TERMINAL_INVALIDATION_RELEASE, a.RECOVERY_RELEASE])
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


@pytest.fixture
def portable_cuda_deployment(deployment):
    """Inert staged files test compatibility guards, not scientific qualification."""
    base, release, unit, root, old = deployment
    source = Path(__file__).resolve().parents[1]
    previous = release.parent / old['source_sha']
    files = a.PORTABLE_NATIVE_SOURCES | {
        'nanobody_opt/pr100_live_worker.py', 'config/pr100_native_reference.json'}
    for name in files:
        origin = source / name
        data = origin.read_bytes() if origin.is_file() else b'inert upstream source fixture\n'
        for staged in (previous, release):
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    rows = a.read(release / 'config/pr100_native_reference.json')['registrations']
    module_hash = hashlib.sha256((release / 'nanobody_opt/pr100_registered_resident.py').read_bytes()).hexdigest()
    row = next(row for row in rows if row.get('resident_module_sha256') == module_hash
               and row['policy']['final_refinement']['platform'] == 'CUDA')
    identity = a.digest(row)
    for staged in (previous, release):
        target = staged / row['qualification_report_file']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / row['qualification_report_file']).read_bytes())
    old.update(native_hosts=['master', 'slave'], native_gpu_by_host={'master': [0, 1], 'slave': [0]},
               native_worker_overrides={})
    old['worker']['native_fleet_policy'] = {
        'kind': 'pr100-observed-fleet-v1', 'minimum_smoothed_fraction': .4}
    runtimes = {}
    for host, gpus in old['native_gpu_by_host'].items():
        old['native_worker_overrides'][host] = {
            'workers': 32, 'max_workers': 32, 'cpu_slots': 32,
            'native_reference': {'enabled': True, 'resident': True,
                'registration_sha256': identity, 'gpus': gpus, 'workers_per_gpu': 4,
                'resident_batch_draws': 8, 'environment': '/sealed/env', 'bundle': '/sealed/bundle'}}
        runtimes[host] = {**a.NOVELTY_PATHS, 'identity_sha256': '7' * 64}
        old['hosts'][host]['novelty_runtime'] = runtimes[host]
    proposed = a.proposed_config(old, SHA, CLOCK['epoch'], 'preserve', {}, novelty_runtimes=runtimes)
    from nanobody_opt import rules
    state = copy.deepcopy(STATE)
    requirements = a.read(release / 'config/nanobody_pr100.json')['nanobody_requirements']
    resume = {'challenge_sha256': a.digest(state['challenge']), 'resume_eligible': True,
        'requirements_sha256': a.digest(requirements), 'filter_sha': rules.FILTER_SHA,
        'backend': 'native_reference', 'registration_sha256': identity,
        'runtime_identity_sha256': row['runtime_sha256'], 'policy_sha256': a.digest(row['policy']),
        'runtime_identity_status': 'registered_native_reference'}
    state['pending_confirmation'] = [{'sequence': 'ACDEFGHIK', 'native_resume': {
        'schema': 'pr100-native-confirmation-checkpoint-v1',
        'sequence_hash': rules.sequence_hash('ACDEFGHIK'), 'identity': resume,
        'fleet_policy': copy.deepcopy(old['worker']['native_fleet_policy']), 'execution_host': 'slave',
        'measurements': [{'inert_fixture': True}], 'required_cpu_observations': 2}}]
    a.atomic(base / 'pr100-controller.json', old)
    a.atomic(root / str(CLOCK['epoch']) / 'continuation.json', state)
    return base, release, unit, root, old, proposed, runtimes, state


def test_portable_cuda_generation_successor_keeps_pending_state(portable_cuda_deployment):
    base, release, unit, root, old, proposed, runtimes, state = portable_cuda_deployment
    worker = release / 'nanobody_opt/pr100_live_worker.py'
    worker.write_text(worker.read_text().replace('def sequence_phase(job, directory):\n',
        'def sequence_phase(job, directory):\n    portable_generation_fixture = True\n'))
    assert a.portable_native_successor(old, proposed, release, state=state)
    checkpoint = root / str(CLOCK['epoch']) / 'continuation.json'
    before, published = checkpoint.read_bytes(), (base / 'publication.json').read_bytes()
    events = []
    result = a.handoff(base, SHA, 'preserve', release, {}, novelty_runtimes=runtimes,
        clock_reader=lambda: CLOCK, now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == 'activated'
    assert events == ['validate', 'stop', 'install', 'start', 'verify']
    assert a.read(base / 'pr100-controller.json') == proposed
    assert checkpoint.read_bytes() == before and (base / 'publication.json').read_bytes() == published


@pytest.mark.parametrize('fault', ['registration', 'resident', 'cpu_backend', 'host_removed',
    'cpu_slots', 'workers', 'gpus', 'batch', 'fleet_policy', 'native_tools', 'novelty_runtime',
    'environment', 'preparation_threads', 'requirements_source', 'scientific_source',
    'novelty_source', 'native_gate', 'missing_old_source', 'checkpoint_registration',
    'checkpoint_policy', 'checkpoint_host', 'checkpoint_challenge', 'checkpoint_requirements',
    'checkpoint_sequence', 'screen_checkpoint'])
def test_portable_cuda_successor_fails_closed(portable_cuda_deployment, fault):
    base, release, unit, root, old, proposed, runtimes, state = portable_cuda_deployment
    native = proposed['native_worker_overrides']['slave']
    settings = native['native_reference']
    if fault == 'registration': settings['registration_sha256'] = 'f' * 64
    if fault == 'resident': settings['resident'] = False
    if fault == 'cpu_backend': settings['enabled'] = False
    if fault == 'host_removed': proposed['native_hosts'] = ['master']
    if fault == 'cpu_slots': native['cpu_slots'] += 1
    if fault == 'workers': native['workers'] += 1
    if fault == 'gpus': proposed['native_gpu_by_host']['slave'] = [1]
    if fault == 'batch': settings['resident_batch_draws'] += 1
    if fault == 'fleet_policy': proposed['worker']['native_fleet_policy']['minimum_smoothed_fraction'] = .5
    if fault == 'native_tools': proposed['worker']['native_tools'] = '/other/tools'
    if fault == 'novelty_runtime': proposed['hosts']['slave']['novelty_runtime']['identity_sha256'] = '8' * 64
    if fault == 'environment': settings['environment'] = '/other/sealed/env'
    if fault == 'preparation_threads': settings['preparation_threads'] = 2
    source = {'requirements_source': 'config/nanobody_pr100.json',
        'scientific_source': 'nanobody_opt/pr100_registered_resident.py',
        'novelty_source': 'NOVA-nanobody-filter/metanano/utils/alignment.py'}.get(fault)
    if source: (release / source).write_bytes(b'changed scientific bytes')
    if fault == 'native_gate':
        path = release / 'nanobody_opt/pr100_live_worker.py'
        path.write_text(path.read_text().replace('def native_resume_identity(job, directory, requirements):\n',
            'def native_resume_identity(job, directory, requirements):\n    changed_gate = True\n'))
    if fault == 'missing_old_source':
        (release.parent / old['source_sha'] / 'nanobody_opt/rules.py').unlink()
    saved = state['pending_confirmation'][0]['native_resume']
    if fault == 'checkpoint_registration': saved['identity']['registration_sha256'] = 'f' * 64
    if fault == 'checkpoint_policy': saved['fleet_policy']['minimum_smoothed_fraction'] = .5
    if fault == 'checkpoint_host': saved['execution_host'] = 'retired-host'
    if fault == 'checkpoint_challenge': saved['identity']['challenge_sha256'] = 'f' * 64
    if fault == 'checkpoint_requirements': saved['identity']['requirements_sha256'] = 'f' * 64
    if fault == 'checkpoint_sequence': saved['sequence_hash'] = 'f' * 64
    if fault == 'screen_checkpoint': saved['accelerated_screen'] = {'inert': True}
    assert not a.portable_native_successor(old, proposed, release, state=state)


@pytest.mark.parametrize('race', ['already_active', 'during_validation', 'after_stop', 'closure_during_validation',
                                  'closure_after_stop', 'checkpoint_after_stop'])
def test_portable_activation_rechecks_ownership_and_compatibility(portable_cuda_deployment, race):
    base, release, unit, root, old, proposed, runtimes, state = portable_cuda_deployment
    path = root / str(CLOCK['epoch']) / 'continuation.json'
    def active():
        changed = copy.deepcopy(state); changed['active'] = {'number': 4, 'phase': 'native'}
        a.atomic(path, changed)
    def closure(): (release / 'nanobody_opt/rules.py').write_bytes(b'changed requirements implementation')
    def checkpoint():
        changed = copy.deepcopy(state)
        changed['pending_confirmation'][0]['native_resume']['identity']['registration_sha256'] = 'f' * 64
        a.atomic(path, changed)
    if race == 'already_active': active()
    events = []
    result = a.handoff(base, SHA, 'preserve', release, {}, novelty_runtimes=runtimes,
        clock_reader=lambda: CLOCK, now=lambda: 1000,
        service_factory=service_factory(unit, events,
            validation=active if race == 'during_validation' else closure if race == 'closure_during_validation' else None,
            stop=active if race == 'after_stop' else closure if race == 'closure_after_stop'
                else checkpoint if race == 'checkpoint_after_stop' else None))
    assert result['status'] == ('boundary_race_old_service_restarted' if race in ('after_stop', 'checkpoint_after_stop', 'closure_after_stop')
                                else 'waiting_for_clean_attempt')
    assert events == ([] if race == 'already_active' else ['validate', 'stop', 'start']
                      if race in ('after_stop', 'checkpoint_after_stop', 'closure_after_stop') else ['validate'])
    assert a.read(base / 'pr100-controller.json') == old


@pytest.mark.parametrize('fault', ['disabled', 'nonresident', 'cpu_refinement', 'mixed_registrations',
                                  'empty_gpus', 'missing_novelty'])
def test_unchanged_nonportable_runtime_cannot_use_exception(portable_cuda_deployment, fault):
    base, release, unit, root, old, proposed, runtimes, state = portable_cuda_deployment
    for config in (old, proposed):
        settings = config['native_worker_overrides']['slave']['native_reference']
        if fault == 'disabled': settings['enabled'] = False
        if fault == 'nonresident': settings['resident'] = False
        if fault == 'empty_gpus':
            settings['gpus'] = []; config['native_gpu_by_host']['slave'] = []
        if fault == 'missing_novelty': config['hosts']['slave'].pop('novelty_runtime')
    if fault in ('cpu_refinement', 'mixed_registrations'):
        registry = a.read(release / 'config/pr100_native_reference.json')
        selected = old['native_worker_overrides']['slave']['native_reference']['registration_sha256']
        row = copy.deepcopy(next(row for row in registry['registrations'] if a.digest(row) == selected))
        if fault == 'cpu_refinement': row['policy']['final_refinement'] = {'platform': 'CPU', 'threads': 1}
        else: row['name'] += '-second-test-registration'
        registry['registrations'].append(row)
        for staged in (release, release.parent / old['source_sha']):
            a.atomic(staged / 'config/pr100_native_reference.json', registry)
        for config in (old, proposed):
            config['native_worker_overrides']['slave']['native_reference']['registration_sha256'] = a.digest(row)
    assert not a.portable_native_successor(old, proposed, release, state=state)


def test_pending_same_release_registration_change_still_waits(portable_cuda_deployment, monkeypatch):
    base, release, unit, root, old, proposed, runtimes, state = portable_cuda_deployment
    old = copy.deepcopy(proposed)
    a.atomic(base / 'pr100-controller.json', old)
    def transition(value, *_):
        value = copy.deepcopy(value)
        value['native_worker_overrides']['slave']['native_reference']['registration_sha256'] = 'f' * 64
        return value
    monkeypatch.setattr(a, 'registered_native_rollout', transition)
    events = []
    result = a.handoff(base, SHA, 'preserve', release, {}, novelty_runtimes=runtimes,
        native_registration='f' * 64, clock_reader=lambda: CLOCK, now=lambda: 1000,
        service_factory=service_factory(unit, events))
    assert result['status'] == 'waiting_for_clean_attempt' and events == []
    assert a.read(base / 'pr100-controller.json') == old


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


@pytest.mark.parametrize('backend', ['cuda_refine', 'resident_cpu'])
def test_native_screen_activation_sets_backend_without_refill_or_interleave(deployment, backend):
    *_, old = deployment
    value = a.proposed_config(old, SHA, CLOCK['epoch'], 'master', native_screen=backend)
    assert value['worker']['native_screen'] == {'enabled': True, 'backend': backend}
    assert not value['worker'].get('native_interleave_confirmations')
    for override in value.get('native_worker_overrides', {}).values():
        assert 'native_work_conserving' not in override


def test_native_screen_handoff_threads_backend_through_proposal(deployment):
    base, release, unit, root, old = deployment; events = []
    result = a.handoff(base, SHA, 'master', release, None, native_screen='cuda_refine',
        clock_reader=lambda: CLOCK, now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == 'activated'
    new = a.read(base / 'pr100-controller.json')
    assert new['worker']['native_screen'] == {'enabled': True, 'backend': 'cuda_refine'}
    assert not new['worker'].get('native_interleave_confirmations')
    for override in new.get('native_worker_overrides', {}).values():
        assert 'native_work_conserving' not in override


def test_native_screen_activation_clears_existing_refill_and_interleave(deployment, novelty_receipt):
    *_, old = deployment
    old['source_sha'] = a.TERMINAL_INVALIDATION_RELEASE
    for host in old['hosts'].values():
        host['release'] = '/root/releases/' + old['source_sha']
    old.update(native_hosts=['master', 'slave'], epoch_fresh_native_limit=4800,
               native_worker_overrides={'master': {'workers': 96, 'native_work_conserving': True},
                                        'slave': {'workers': 32}})
    old['worker'].update(native_interleave_confirmations=True, native_screen={'enabled': False})
    release, receipt, _ = novelty_receipt
    runtimes = a.verify_novelty_qualification(release, receipt)
    value = a.proposed_config(old, SHA, CLOCK['epoch'], 'preserve', {}, novelty_runtimes=runtimes,
                              native_screen='resident_cpu')
    assert value['worker']['native_screen'] == {'enabled': True, 'backend': 'resident_cpu'}
    assert value['worker'].get('native_interleave_confirmations') is False
    assert 'native_work_conserving' not in value['native_worker_overrides']['master']


def test_native_screen_with_refill_host_raises(deployment):
    *_, old = deployment
    with pytest.raises(ValueError, match='mutually_exclusive'):
        a.proposed_config(old, SHA, CLOCK['epoch'], 'master', native_screen='cuda_refine',
                          native_refill_hosts=['slave'])


def test_native_screen_with_interleave_raises(deployment):
    *_, old = deployment
    with pytest.raises(ValueError, match='mutually_exclusive'):
        a.proposed_config(old, SHA, CLOCK['epoch'], 'master', native_screen='cuda_refine',
                          interleave_confirmations=True)


@pytest.mark.parametrize('conflict', [['--interleave-confirmations'], ['--native-refill-host', 'slave']])
def test_cli_rejects_native_screen_with_conflicting_modes(conflict):
    with pytest.raises(SystemExit) as error:
        a.main([SHA, 'master', '--staging-receipt', '/unused',
                '--novelty-qualification-receipt', '/unused', '--native-screen', 'cuda_refine'] + conflict)
    assert error.value.code == 2


def test_non_screen_activation_preserves_native_screen(deployment):
    *_, old = deployment
    value = a.proposed_config(old, SHA, CLOCK['epoch'], 'master')
    assert 'native_screen' not in value['worker']


@pytest.fixture
def recovery_receipt(deployment):
    from nanobody_opt import pr100_native_screen as screen
    base, _release, _unit, _root, old = deployment
    old['hosts']['slave']['novelty_runtime'] = {'identity_sha256': 'b' * 64}
    a.atomic(base / 'pr100-controller.json', old)
    settings = screen.settings({'native_screen': {
        'enabled': True, 'backend': 'cuda_refine', 'workers': 32, 'workers_per_gpu': 2,
        'gpus': [1, 2], 'runtime_versions': {**screen.PINNED_VERSIONS, 'torch': '2.7.1+cu126'}}})
    return {'schema': 'pr100-recovery-rollout-v1', 'source_sha': SHA,
        'native_screen_hosts': {'master': {'settings': settings, 'smoke': {
            'source_sha': SHA, 'host': 'master', 'status': 'complete',
            'settings_sha256': screen.digest(settings),
            'protocol_sha256': screen.digest(screen.protocol(settings['backend'], settings['runtime_versions'])),
            'measured_count': 2, 'runtime_inconclusive_count': 0,
            'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}}}},
        'reference_bank': {'path': '/root/pr100-rival-bank/input.json', 'sha256': 'c' * 64, 'rows': 12,
            'annotation_qualification': {'source_sha': SHA, 'host': 'slave', 'status': 'complete',
                'bank_sha256': 'c' * 64, 'annotated_rows': 12, 'runtime_sha256': 'b' * 64}},
        'edit_generation': {'settings': {'enabled': False, 'limit': 300}}}


def test_measured_recovery_configuration_is_applied_inside_atomic_handoff(deployment, recovery_receipt):
    base, release, unit, root, old = deployment; events = []
    state = (root / '25964/continuation.json').read_bytes()
    publication = (base / 'publication.json').read_bytes()
    result = a.handoff(base, SHA, 'master', release, None, recovery_rollout_receipt=recovery_receipt,
        clock_reader=lambda: CLOCK, now=lambda: 1000, service_factory=service_factory(unit, events))
    assert result['status'] == 'activated'
    assert events == ['validate', 'stop', 'install', 'start', 'verify']
    new = a.read(base / 'pr100-controller.json')
    assert new['native_worker_overrides']['master']['native_screen'] == recovery_receipt['native_screen_hosts']['master']['settings']
    assert new['native_gpu_by_host']['master'] == [1, 2]
    assert new['hosts']['slave']['references'] == '/root/pr100-rival-bank/input.json'
    assert new['hosts']['slave']['reference_bank_sha256'] == 'c' * 64
    assert new['worker']['edit_generation'] == {'enabled': False, 'limit': 300}
    assert (root / '25964/continuation.json').read_bytes() == state
    assert (base / 'publication.json').read_bytes() == publication


@pytest.mark.parametrize('bad', ['source', 'unknown', 'smoke_host', 'count', 'inconclusive',
    'cleanup', 'settings_binding', 'protocol_binding', 'runtime', 'bank_count', 'bank_runtime',
    'bank_path', 'edit_unqualified'])
def test_unqualified_recovery_receipt_never_stops_or_reconfigures_service(deployment, recovery_receipt, bad):
    base, release, unit, root, old = deployment; events = []
    receipt = copy.deepcopy(recovery_receipt)
    smoke = receipt['native_screen_hosts']['master']['smoke']
    bank = receipt['reference_bank']
    if bad == 'source': receipt['source_sha'] = 'f' * 40
    if bad == 'unknown': receipt['gpu'] = [0]
    if bad == 'smoke_host': smoke['host'] = 'slave'
    if bad == 'count': smoke['measured_count'] = 0
    if bad == 'inconclusive': smoke['runtime_inconclusive_count'] = 1
    if bad == 'cleanup': smoke['cleanup']['remaining_descendants'] = 1
    if bad == 'settings_binding': smoke['settings_sha256'] = 'f' * 64
    if bad == 'protocol_binding': smoke['protocol_sha256'] = 'f' * 64
    if bad == 'runtime': receipt['native_screen_hosts']['master']['settings']['runtime_versions']['torch'] = 'unqualified'
    if bad == 'bank_count': bank['rows'] += 1
    if bad == 'bank_runtime': bank['annotation_qualification']['runtime_sha256'] = 'f' * 64
    if bad == 'bank_path': bank['path'] = '/root/../unapproved.json'
    if bad == 'edit_unqualified': receipt['edit_generation']['settings']['enabled'] = True
    before = (base / 'pr100-controller.json').read_bytes()
    with pytest.raises(ValueError):
        a.handoff(base, SHA, 'master', release, None, recovery_rollout_receipt=receipt,
            clock_reader=lambda: CLOCK, now=lambda: 1000, service_factory=service_factory(unit, events))
    assert events == []
    assert (base / 'pr100-controller.json').read_bytes() == before


def test_recovery_receipt_does_not_bypass_a_stop_race(deployment, recovery_receipt):
    base, release, unit, root, old = deployment; events = []
    def changed(): a.atomic(root / '25964/continuation.json', {**STATE, 'active': {'number': 3}})
    before = (base / 'pr100-controller.json').read_bytes()
    result = a.handoff(base, SHA, 'master', release, None, recovery_rollout_receipt=recovery_receipt,
        clock_reader=lambda: CLOCK, now=lambda: 1000,
        service_factory=service_factory(unit, events, stop=changed), force=True)
    assert result['status'] == 'boundary_race_old_service_restarted'
    assert events == ['validate', 'stop', 'start']
    assert (base / 'pr100-controller.json').read_bytes() == before


def test_edit_rollout_requires_measured_novelty_with_exact_settings_and_bank(deployment, recovery_receipt):
    from nanobody_opt import rules
    *_, old = deployment
    candidate = a.proposed_config(old, SHA, CLOCK['epoch'], 'master')
    edits = recovery_receipt['edit_generation']
    edits['settings']['enabled'] = True
    edits['qualification'] = {'source_sha': SHA, 'status': 'complete', 'bank_sha256': 'c' * 64,
        'settings_sha256': a.digest(edits['settings']), 'archive_sha256': 'd' * 64,
        'filter_sha': rules.FILTER_SHA, 'tested_candidates': 20, 'novelty_passed_count': 3,
        'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}}
    assert a.recovery_rollout(candidate, recovery_receipt, SHA)['worker']['edit_generation']['enabled']
    edits['settings']['limit'] += 1
    with pytest.raises(ValueError, match='novelty_not_qualified'):
        a.recovery_rollout(candidate, recovery_receipt, SHA)


def test_chemistry_policy_cannot_reuse_balanced_qualification(deployment, recovery_receipt):
    from nanobody_opt import rules, pr100_edit_generation as edit
    *_, old = deployment
    candidate = a.proposed_config(old, SHA, CLOCK['epoch'], 'master')
    edits = recovery_receipt['edit_generation']
    edits['settings'] = {'enabled': True, 'limit': 60}
    edits['qualification'] = {'source_sha': SHA, 'status': 'complete', 'bank_sha256': 'c' * 64,
        'settings_sha256': a.digest(edits['settings']), 'archive_sha256': 'd' * 64,
        'filter_sha': rules.FILTER_SHA, 'tested_candidates': 60, 'novelty_passed_count': 37,
        'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}}
    edits['settings']['policy'] = edit.CHEMISTRY_POLICY
    with pytest.raises(ValueError, match='novelty_not_qualified'):
        a.recovery_rollout(candidate, recovery_receipt, SHA)
    edits['qualification']['settings_sha256'] = a.digest(edits['settings'])
    activated = a.recovery_rollout(candidate, recovery_receipt, SHA)
    assert activated['worker']['edit_generation'] == edits['settings']
    edits['settings']['policy'] = 'unrecognized'
    with pytest.raises(ValueError, match='policy'):
        a.recovery_rollout(candidate, recovery_receipt, SHA)


def test_cli_loads_recovery_receipt_for_handoff(deployment, recovery_receipt, monkeypatch):
    base, *_ = deployment; stage = base / 'stage.json'; stage.write_text('{}')
    receipt = base / 'recovery.json'; a.atomic(receipt, recovery_receipt)
    calls = []
    monkeypatch.setattr(a, 'BASE', base)
    monkeypatch.setattr(a, 'verify_prerequisites', lambda *args, **kwargs: None)
    monkeypatch.setattr(a, 'verify_novelty_qualification', lambda *args: {})
    def handoff(*args, **kwargs):
        calls.append(kwargs['recovery_rollout_receipt']); return {'status': 'activated'}
    monkeypatch.setattr(a, 'handoff', handoff)
    assert a.main([SHA, 'preserve', '--staging-receipt', str(stage),
        '--novelty-qualification-receipt', str(stage), '--recovery-rollout-receipt', str(receipt)]) == 0
    assert calls == [recovery_receipt]

