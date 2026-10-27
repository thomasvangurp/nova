import copy
import hashlib
import json
import sys
import types
from types import SimpleNamespace
import pytest
from nanobody_opt import pr100_native_screen as screen

def test_screen_profile_survives_installed_scripts_package(tmp_path,monkeypatch):
    from nanobody_opt import rules
    from nanobody_opt.pr100_native_benchmark import FIELDS,A3D_FIELDS
    sequence='A'*100
    evidence={'calls':[],'platforms':[]}
    measured={**dict.fromkeys(FIELDS,1.0),'surface_hydrophobic_patches':80.0,'passed':True,
        'aggrescan3d':{**dict.fromkeys(A3D_FIELDS,1.0),'status':'ok','passed':True,
            'input_structure_sha256':'b'*64,'normalized_structure_sha256':'c'*64,
            'sequence_hash':hashlib.sha256(sequence.encode()).hexdigest(),
            'ph_scale':'default-scale','surface_method':'FreeSASA-2.0.1-NACCESS-LR100',
            'mode':'static-no-minimization','radius_angstrom':10}}
    def profile_one(*_):
        evidence['calls'].append({'native_success':True})
        evidence['platforms'].append({'name':'CPU'})
        return measured
    wrapper=types.ModuleType('metanano.utils.tnp_wrapper')
    wrapper.ProfileExecution=lambda **_:SimpleNamespace(remaining=lambda:1)
    monkeypatch.setitem(sys.modules,'metanano.utils.tnp_wrapper',wrapper)
    monkeypatch.setitem(sys.modules,'scripts',types.ModuleType('scripts'))
    monkeypatch.delitem(sys.modules,'scripts.pr100_native_probe',raising=False)
    monkeypatch.setattr(rules,'developability_reason',lambda *a,**k:None)
    request={'settings':{'backend':'resident_cpu'},'challenge':{'epoch':25962},
        'request_sha256':'a'*64,'candidate_timeout_seconds':30,'requirements':{},
        'results_directory':str(tmp_path)}
    monkeypatch.setattr(screen,'_STATE',{'request':request,'profiler':SimpleNamespace(profile_one=profile_one),
        'evidence':evidence,'scorer':SimpleNamespace(profile=lambda *a:None),'identity':{},
        'analyzer':SimpleNamespace(analyze_profile=lambda seq,value:SimpleNamespace(to_dict=lambda:value))})
    screen.profile_sanitizer.cache_clear()
    try:
        key=screen._screen_one(sequence)
        record=json.loads((tmp_path/(key+'.json')).read_text())
        assert record['screen_outcome']=='passed'
        assert record['profile']['surface_hydrophobic_patches']==80.0
        assert record['authoritative'] is False
    finally:screen.profile_sanitizer.cache_clear()

def fixture(backend='resident_cpu'):
    sequence = 'A' * 100
    request = {'settings': {'backend': backend}, 'challenge': {'epoch': 25962}, 'request_sha256': 'f'*64}
    request['expected_model_weights']={str(i):'a'*64 for i in range(1,5)}
    identity = dict(sources=dict(screen.PINNED_SOURCES), versions=dict(screen.PINNED_VERSIONS),
        model_weights={str(i): 'a'*64 for i in range(1,5)}, python=[3,12,14], prediction_device='cpu',
        accelerator_source_sha256=screen.file_hash(screen.__file__), hydrogen_context_platform_observed=False,
        model_hash_scope='coordinator_hash_worker_path_stat_before_after_load')
    platform = {'name': 'CPU', 'properties': {'Threads': '1'}} if backend == 'resident_cpu' else {
        'name': 'CUDA', 'properties': {'Precision': 'single', 'DeviceIndex': '0'}}
    key = hashlib.sha256(sequence.encode()).hexdigest()
    record = dict(schema=screen.SCHEMA, evidence_backend=backend, authoritative=False, sequence_hash=key,
        epoch=25962, challenge_sha256=screen.digest(request['challenge']), request_sha256='f'*64,
        protocol_sha256=screen.digest(screen.protocol(backend)), runtime_identity_sha256=screen.digest(identity),
        screen_outcome='passed', profile={'aggrescan3d': {'sequence_hash': key}},
        provenance={**identity, 'final_refinement_success': True,
                    'refinement': {'calls': [{'native_success': True}], 'platforms': [platform]}})
    return sequence, request, record

@pytest.mark.parametrize('backend',screen.BACKENDS)
def test_valid_screen_is_explicitly_non_authoritative(backend):
    sequence,request,record=fixture(backend)
    assert screen.valid_record(record,request=request,sequence=sequence)
    record['authoritative']=True
    assert not screen.valid_record(record,request=request,sequence=sequence)

@pytest.mark.parametrize('field,value',[('epoch',25961),('challenge_sha256','0'*64),
    ('protocol_sha256','0'*64),('runtime_identity_sha256','0'*64),('request_sha256','0'*64),
    ('evidence_backend','cpu'),('sequence_hash','0'*64),('screen_outcome','inconclusive')])
def test_stale_or_untrusted_record_rejected(field,value):
    sequence,request,record=fixture()
    record[field]=value
    assert not screen.valid_record(record,request=request,sequence=sequence)

@pytest.mark.parametrize('change',['version','prediction','weights','platform','calls','missing','precision'])
def test_self_consistent_digest_is_not_semantic_attestation(change):
    sequence,request,record=fixture('cuda_refine')
    p=record['provenance']
    if change=='version':p['versions']['openmm']='9.0'
    elif change=='prediction':p['prediction_device']='cuda'
    elif change=='weights':p['model_weights']={'1':'bad'}
    elif change=='platform':p['refinement']['platforms'][0]['name']='CPU'
    elif change=='calls':p['refinement']['calls'][-1]['native_success']=False
    elif change=='missing':p['refinement']['platforms']=[]
    elif change=='precision':p['refinement']['platforms'][0]['properties']['Precision']='mixed'
    identity={k:v for k,v in p.items() if k not in ('final_refinement_success','refinement')}
    record['runtime_identity_sha256']=screen.digest(identity)
    assert not screen.valid_record(record,request=request,sequence=sequence)

@pytest.mark.parametrize('bad',[None,[],{'provenance':None},{'provenance':{'refinement':None}}])
def test_malformed_records_fall_back(bad):
    sequence,request,_=fixture()
    assert not screen.valid_record(bad,request=request,sequence=sequence)

def test_bounded_devices_and_contexts():
    value=screen.settings({'gpus':[1,2,3,4],'native_screen':{'enabled':True,'backend':'cuda_refine','gpus':[5,6,7]}})
    assert value['gpus']==[5,6,7] and value['workers_per_gpu']==4
    with pytest.raises(ValueError):screen.settings({'native_screen':{'workers_per_gpu':48}})

def test_disabled_or_short_window_never_launches(tmp_path,monkeypatch):
    monkeypatch.setattr(screen.subprocess,'Popen',lambda *a,**k:pytest.fail('must not launch'))
    assert screen.screen_batch(['A'],job={},directory=tmp_path/'disabled',timeout_seconds=60)['profiles']=={}
    job={'native_screen':{'enabled':True,'backend':'resident_cpu'}}
    assert screen.screen_batch(['A'],job=job,directory=tmp_path/'short',timeout_seconds=24)['status']=='unavailable'

def native_fixture():
    calls=[]
    class Platform:
        def __init__(self,name):self.name=name
        def getName(self):return self.name
        def getPropertyNames(self):return ['Threads','Precision','DeviceIndex']
        def getPropertyValue(self,context,name):return {'Threads':'1','Precision':'single','DeviceIndex':'0'}[name]
    native=SimpleNamespace(Platform=SimpleNamespace(getPlatformByName=Platform))
    def simulation(*args,**kwargs):
        platform=kwargs.get('platform') or args[3]
        return SimpleNamespace(context=SimpleNamespace(getPlatform=lambda:platform))
    native.app=SimpleNamespace(Simulation=simulation)
    def refine(input_file,output_file,check_for_strained_bonds=True,tries=3,n=6,n_threads=-1):
        calls.append((check_for_strained_bonds,tries,n,n_threads))
        if n_threads>0:native.app.Simulation(1,2,3,Platform('CPU'),{'Threads':str(n_threads)})
        else:native.app.Simulation(1,2,3)
        return True
    native.refine=refine
    return native,SimpleNamespace(refine=refine),calls,simulation

@pytest.mark.parametrize('cuda',[False,True])
def test_only_routing_changes_and_hooks_restore(cuda):
    native,builder,calls,simulation=native_fixture()
    original=native.refine
    with screen.refinement_observer(native,builder,cuda=cuda) as evidence:
        assert builder.refine('in','out',n_threads=1)
        assert calls==[(True,3,6,-1 if cuda else 1)]
        assert evidence['platforms'][0]['name']==('CUDA' if cuda else 'CPU')
    assert native.refine is builder.refine is original
    assert native.app.Simulation is simulation

def test_disabled_scientific_check_rejected_and_restored():
    native,builder,calls,simulation=native_fixture()
    original=native.refine
    with pytest.raises(ValueError),screen.refinement_observer(native,builder,cuda=True):
        builder.refine('in','out',check_for_strained_bonds=False,n_threads=1)
    assert not calls and native.refine is original and native.app.Simulation is simulation

@pytest.mark.parametrize('fail',[False,True])
def test_guard_always_cleans_after_coordinator_failure(tmp_path,monkeypatch,fail):
    cleaned=[]
    class Control:
        def dedicated_controller(self):pass
        def stop_descendants(self,children,timeout):
            cleaned.extend(children)
            return {'remaining_descendants':0,'fixed_point_reached':True}
    class Child:
        def wait(self):
            if fail:raise RuntimeError('coordinator_failed')
            return 7
    child=Child()
    monkeypatch.setattr(screen.signal,'signal',lambda *a:None)
    monkeypatch.setattr(screen.signal,'setitimer',lambda *a:None)
    screen._guard({'directory':str(tmp_path),'timeout_seconds':30},launch=lambda *a,**k:child,control=Control())
    result=screen.json.loads((tmp_path/'guard.json').read_text())
    assert cleaned==[child] and result['cleanup']['remaining_descendants']==0
