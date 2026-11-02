"""Resident lifecycle and original CLI control flow, not scientific qualification."""
import argparse
import ast
import asyncio
import hashlib
import io
import json
import logging
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest

from nanobody_opt import pr100_registered_resident as resident

FIXTURE = Path(__file__).parent/'fixtures/pr100/resident_entrypoints'
POLICY = {'prediction':'auto_cuda','preparation_threads':1,
          'final_refinement':{'platform':'CUDA','precision':'mixed','deterministic_forces':True}}
CPU_POLICY = {'prediction':'auto_cuda','preparation_threads':1,
              'final_refinement':{'platform':'CPU','threads':1}}


def original_cli(monkeypatch):
    module = ModuleType('ImmuneBuilder.NanoBodyBuilder2')
    package = ModuleType('ImmuneBuilder');package.__path__=[]
    monkeypatch.setitem(sys.modules,'ImmuneBuilder',package)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    calls=[]
    class Prediction:
        def save(self,path,**kwargs):calls.append(('save',path,kwargs))
    class Builder:
        def __init__(self,**kwargs):calls.append(('constructor',kwargs))
        def predict(self,sequences):calls.append(('predict',sequences));return Prediction()
    module.__dict__.update(argparse=argparse,sys=sys,NanoBodyBuilder2=Builder,
                          torch=NS(set_num_threads=lambda n:calls.append(('threads',n))))
    tree=ast.parse((FIXTURE/'NanoBodyBuilder2.py').read_bytes())
    cli=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='command_line_interface')
    exec(compile(ast.Module(body=[cli],type_ignores=[]),str(FIXTURE/'NanoBodyBuilder2.py'),'exec'),module.__dict__)
    entry=compile((FIXTURE/'NanoBodyBuilder2').read_bytes(),str(FIXTURE/'NanoBodyBuilder2'),'exec')
    return module,Builder(),calls,entry


def test_actual_original_cli_constructor_reuse_keeps_every_predict_save(monkeypatch,tmp_path):
    module,model,calls,code=original_cli(monkeypatch)
    original_constructor=module.NanoBodyBuilder2
    argv,cwd,run,disabled=sys.argv,Path.cwd(),subprocess.run,logging.root.manager.disable
    for index,sequence in enumerate(('A'*60,'G'*60,'A'*60)):
        profiles=tmp_path/str(index);profiles.mkdir()
        bridge=resident.OriginalNBB2CLI(module,model,sequence,'name',profiles,entrypoint_code=code)
        with resident.original_program_context(['TNP'],profiles,bridge,
                preparation_threads=1,trace_path=profiles/'unused'):
            assert subprocess.run(bridge.command,check=True).returncode==0
        assert module.NanoBodyBuilder2 is original_constructor
    assert sum(x[0]=='constructor' for x in calls)==1
    assert [x[1] for x in calls if x[0]=='predict']==[{'H':'A'*60},{'H':'G'*60},{'H':'A'*60}]
    assert [x[1] for x in calls if x[0]=='threads']==[1,1,1]
    assert all(x[2]=={'check_for_strained_bonds':True,'n_threads':1} for x in calls if x[0]=='save')
    assert sys.argv is argv and Path.cwd()==cwd and subprocess.run is run
    assert logging.root.manager.disable==disabled


@pytest.mark.parametrize('failure,expected',[(AssertionError,subprocess.CalledProcessError),
                                            (ArithmeticError,ArithmeticError)])
def test_original_cli_error_restores_process_globals(monkeypatch,tmp_path,failure,expected):
    module,model,calls,code=original_cli(monkeypatch)
    model.predict=lambda _seq:(_ for _ in ()).throw(failure())
    constructor=module.NanoBodyBuilder2;argv=sys.argv;run=subprocess.run
    bridge=resident.OriginalNBB2CLI(module,model,'A'*60,'name',tmp_path,entrypoint_code=code)
    with pytest.raises(expected):
        with resident.original_program_context(['TNP'],tmp_path,bridge,
                preparation_threads=1,trace_path=tmp_path/'unused'):
            subprocess.run(bridge.command,check=True)
    assert module.NanoBodyBuilder2 is constructor and sys.argv is argv and subprocess.run is run


def test_router_uses_current_observer_phase_and_preserves_other_contexts(tmp_path):
    from nanobody_opt.pr100_context_observer import Observer
    calls=[]
    class Context:
        def __init__(self,*args,**kwargs):calls.append((args,kwargs))
    original=Context.__init__;mm=NS(Context=Context,Platform=NS(getPlatformByName=lambda name:name))
    observer=Observer(tmp_path/'trace.jsonl')
    with resident.route_contexts(observer,mm,POLICY,temporary_directory=tmp_path):
        Context('outside')
        for role in ('missing_atoms','hydrogens','minimize_energy','sidechain_repair'):
            with observer.phase(role):
                if role=='sidechain_repair':
                    with observer.phase('minimize_energy'):Context('system','integrator','CPU',{'Threads':'1'})
                else:Context('system','integrator')
    assert Context.__init__ is original
    assert calls[0]==(('outside',),{})
    assert calls[1][0]==('system','integrator','CPU',{'Threads':'1'})
    assert calls[2][0]==('system','integrator','CPU',{'Threads':'1'})
    for call in calls[3:]:
        assert call[0][2]=='CUDA'
        assert call[0][3]=={'Precision':'mixed','DeterministicForces':'true','DeviceIndex':'0','TempDirectory':str(tmp_path)}


def test_router_rejects_unqualified_policy_and_restores_on_failure(tmp_path):
    class Context:
        def __init__(self,*args):raise ArithmeticError()
    original=Context.__init__;mm=NS(Context=Context,Platform=NS(getPlatformByName=lambda name:name))
    observer=NS(local=threading.local())
    with pytest.raises(resident.AdapterError,match='registered_cuda_policy'):
        with resident.route_contexts(observer,mm,{'final_refinement':{'platform':'CPU'}},temporary_directory=tmp_path):pass
    with pytest.raises(ArithmeticError):
        with resident.route_contexts(observer,mm,POLICY,temporary_directory=tmp_path):Context('outside')
    assert Context.__init__ is original


def test_cpu_router_preserves_original_context_identity_and_all_arguments(tmp_path):
    calls=[]
    class Context:
        def __init__(self,*args,**kwargs):calls.append((args,kwargs))
    original=Context.__init__
    # Any platform lookup/sidecar route is a failure in the CPU path.
    mm=NS(Context=Context,Platform=NS(getPlatformByName=lambda *_:pytest.fail('CPU platform untouched')))
    with resident.route_contexts(NS(),mm,CPU_POLICY,temporary_directory=tmp_path):
        assert Context.__init__ is original
        Context('system','integrator','CPU',{'Threads':'1'})
        Context('preparation',platform='CPU')
    assert Context.__init__ is original
    assert calls==[(('system','integrator','CPU',{'Threads':'1'}),{}),
                   (('preparation',),{'platform':'CPU'})]


@pytest.mark.parametrize('cpu',[True,False])
def test_refinement_plugin_is_loaded_only_for_cuda_policy(monkeypatch,cpu):
    calls=[]
    monkeypatch.setattr(resident.native,'cuda_plugin_paths',lambda *args:
        calls.append('sidecar_verified') or {'plugin_directory':'/sealed/plugin'})
    mm=NS(Platform=NS(loadPluginsFromDirectory=lambda path:calls.append(('load',path)),
        getPlatformByName=lambda name:calls.append(('platform',name))))
    resident.load_refinement_plugin(mm,{}, {'policy':CPU_POLICY if cpu else POLICY})
    assert calls==([] if cpu else ['sidecar_verified',('load','/sealed/plugin'),('platform','CUDA')])


@pytest.mark.parametrize('change',[None,'hash','duplicate','compiled','cpp','missing_descriptor'])
def test_cuda_startup_binds_unique_actual_maps_before_any_context(tmp_path,change):
    core=tmp_path/'libOpenMM.so';core.write_bytes(b'core')
    nvrtc=tmp_path/'libnvrtc.so.12.6.20';nvrtc.write_bytes(b'nvrtc')
    loader={'core_sha256':resident.native.sha(core),'nvrtc_sha256':resident.native.sha(nvrtc),
            'torch_compiled_cuda':12060,'cpp_version':'8.5'}
    plugin={'distribution':'OpenMM-CUDA-12','version':'8.5.0',
        'wheel_sha256':resident.reference.CUDA_PLUGIN_WHEEL_SHA256,
        'manifest_sha256':'e'*64,'loader':loader}
    row={'policy':POLICY,'cuda_plugin':plugin}
    paths=[core,core,nvrtc]  # Multiple ELF segments are one actual mapped file.
    if change=='hash':nvrtc.write_bytes(b'changed')
    if change=='duplicate':
        other=tmp_path/'other/libnvrtc.so.12.6.20';other.parent.mkdir();other.write_bytes(b'nvrtc');paths.append(other)
    if change=='missing_descriptor':plugin.pop('loader')
    maps=tmp_path/'maps';maps.write_text('\n'.join(f'000-111 r-xp 000 00:00 1 {p}' for p in paths))
    mm=NS(Platform=NS(getOpenMMVersion=lambda:'8.6' if change=='cpp' else '8.5'),
          Context=lambda *_:pytest.fail('must refuse before Context construction'))
    torch=NS(_C=NS(_cuda_getCompiledVersion=lambda:12090 if change=='compiled' else 12060))
    if change:
        with pytest.raises(resident.AdapterError):resident.verify_cuda_loader(mm,torch,row,maps_path=maps)
    else:assert resident.verify_cuda_loader(mm,torch,row,maps_path=maps)==loader
    assert resident.verify_cuda_loader(None,None,{'policy':CPU_POLICY},maps_path=tmp_path/'absent') is None


def test_model_drift_prevents_any_authoritative_delivery(monkeypatch):
    engine=resident.Engine.__new__(resident.Engine);engine.model=object();engine.initial_state='a'*64
    monkeypatch.setattr(resident,'model_fingerprint',lambda _model:'b'*64)
    monkeypatch.setattr(resident.native,'authoritative_measurement',lambda _value:pytest.fail('must not certify drift'))
    with pytest.raises(resident.AdapterError,match='model_state_changed'):
        engine.finish_batch({},[{}],batch_id='a'*32)


def test_model_identity_includes_numbering_scheme_and_device():
    model=NS(device='cuda',scheme='imgt',models={})
    original=resident.model_fingerprint(model)
    model.scheme='kabat';assert resident.model_fingerprint(model)!=original
    model.scheme='imgt';model.device='cpu';assert resident.model_fingerprint(model)!=original


def test_model_checks_are_batch_boundary_not_each_draw(monkeypatch):
    calls=[]
    class Engine:
        def __init__(self,_request):calls.append('startup')
        def draw(self,_request,index,*,deadline):calls.append(('fresh_draw',index));return {'stub':True}
        def finish_batch(self,_request,pending,*,batch_id):
            calls.append(('postcheck',len(pending)));return [{'authoritative':False,'inert_test_only':True} for _ in pending]
    monkeypatch.setattr(resident,'Engine',Engine)
    command={'request':{},'indices':[0,1,2],'deadline':1.,'batch_id':'a'*32}
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(command)+'\n'+json.dumps(command)+'\n{"stop":true}\n'))
    channel=io.StringIO();assert resident.worker_main({},channel)==0
    assert calls==['startup',('fresh_draw',0),('fresh_draw',1),('fresh_draw',2),('postcheck',3),
                  ('fresh_draw',0),('fresh_draw',1),('fresh_draw',2),('postcheck',3)]
    assert len(channel.getvalue().splitlines())==3


def test_failed_draw_retains_checked_prefix_and_marks_only_actual_attempts(monkeypatch,tmp_path):
    class Engine:
        def __init__(self,_request):pass
        def draw(self,_request,index,*,deadline):
            (tmp_path/(str(index)+'.started.json')).write_text('{}')
            if index==1:raise ArithmeticError()
            return {'inert_test_only':True}
        def finish_batch(self,_request,pending,*,batch_id):
            return [{'authoritative':False,'inert_test_only':True} for _ in pending]
    monkeypatch.setattr(resident,'Engine',Engine)
    command={'request':{'directory':str(tmp_path)},'indices':[0,1,2],'deadline':1.,'batch_id':'a'*32}
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(command)+'\n'))
    channel=io.StringIO();assert resident.worker_main({},channel)==1
    reply=json.loads(channel.getvalue().splitlines()[1])
    assert reply['ok'] is False and reply['indices']==[0]
    assert reply['attempted']==[0,1] and len(reply['values'])==1


def test_expired_draw_never_writes_started_marker(monkeypatch,tmp_path):
    from nanobody_opt import pr100_live_gpu
    engine=resident.Engine.__new__(resident.Engine);engine.gpu=0
    monkeypatch.setattr(pr100_live_gpu,'inherited_lease_fd',lambda _gpus:4)
    with pytest.raises(resident.AdapterError,match='deadline_or_lease_lost'):
        engine.draw({'directory':str(tmp_path),'sequences':['A'*60]},0,deadline=time.monotonic()-1)
    assert not list(tmp_path.iterdir())


def test_slow_peer_reply_cannot_hide_completed_peer_or_exceed_cutoff():
    complete=subprocess.Popen([sys.executable,'-c',
        'import json;print(json.dumps({"ok":True,"inert_checked_peer":True}),flush=True)'],
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    slow=subprocess.Popen([sys.executable,'-c','import time;time.sleep(3)'],
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    try:
        start=time.monotonic();replies=resident.collect_worker_replies([slow,complete],start+.3)
        assert time.monotonic()-start<1.
        assert replies[complete]['inert_checked_peer'] is True and slow not in replies
    finally:
        for process in (complete,slow):
            if process.poll() is None:process.terminate()
            process.wait(timeout=3)


@pytest.mark.parametrize('late_value',[False,True])
def test_actual_server_retains_checked_peers_after_late_retirement(monkeypatch,tmp_path,late_value):
    """Exercise real coordinator ordering; inert protocol values are not science."""
    from nanobody_opt import pr100_live_gpu
    root=tmp_path/'session';root.mkdir();(root/'commands').mkdir();(root/'responses').mkdir()
    directory=root/'batch';directory.mkdir();(directory/'1.started.json').write_text('{}')
    settings={'registration_sha256':'a'*64,'workers_per_gpu':2,'resident_batch_draws':32}
    sequences=['A'*60,'G'*60,'C'*60,'D'*60]
    request={'settings':settings,'gpu':0,'source_sha':'a'*40,'directory':str(root),'deadline':1000.}
    batch={'settings':settings,'gpu':0,'source_sha':'a'*40,'directory':str(directory),'sequences':sequences}
    command={'request':batch,'deadline':1000.,'batch_id':'a'*32}
    (root/'commands'/('a'*32+'.json')).write_text(json.dumps(command))
    processes=[];clock=[900.];checks=[]
    class Process:
        def __init__(self,*_args,**_kwargs):
            self.stdout=io.StringIO('{"ready":true}\n');self.stdin=io.StringIO();self.done=False
            processes.append(self)
        def poll(self):return 0 if self.done else None
        def wait(self,timeout):self.done=True;return 0
        def terminate(self):self.done=True
        @property
        def pid(self):return 1000+processes.index(self)
    def collect(_processes,work_deadline):
        assert work_deadline==990.
        clock[0]=1001.  # Delayed collection crosses client cutoff, not draw completion.
        values=[{'sequence_hash':hashlib.sha256(sequences[i].encode()).hexdigest(),
            'inert_test_only':True,'authoritative':False,
            'native_reference':{'receipt':{'request':{
                'scientific_completed_monotonic':1001. if late_value and i==2 else 980.,
                'scientific_deadline_monotonic':990.}}}}
            for i in (0,2)]
        return {processes[0]:{'ok':True,'indices':[0,2],'values':values,'attempted':[0,2]}}
    monkeypatch.setattr(pr100_live_gpu,'inherited_lease_fd',lambda _gpus:4)
    monkeypatch.setattr(resident.native,'registration',lambda _identity:{})
    monkeypatch.setattr(resident.native,'verify_files',lambda *_args,**_kwargs:checks.append('verified'))
    monkeypatch.setattr(resident.native,'authoritative_measurement',lambda row:row.get('inert_test_only') is True)
    monkeypatch.setattr(resident.subprocess,'Popen',Process)
    monkeypatch.setattr(resident,'collect_worker_replies',collect)
    monkeypatch.setattr(resident.time,'monotonic',lambda:clock[0])
    assert resident.serve(request)==0
    reply=json.loads((root/'responses'/('a'*32+'.json')).read_text())['values']
    assert reply[0]['inert_test_only']
    if late_value:assert reply[2]['inconclusive'] is True and reply[2]['native_started'] is True
    else:assert reply[2]['inert_test_only']
    assert reply[1]['inconclusive'] is True and reply[1]['native_started'] is True
    assert reply[3]['inconclusive'] is True and reply[3]['native_started'] is False
    assert (directory/'0/measurement.json').exists()
    assert (directory/'2/measurement.json').exists() is not late_value
    assert checks==['verified','verified'] and (root/'RETIRING.json').exists()
    assert all(p.poll()==0 for p in processes)


def test_native_phase_128_queue_slots_keep_16_actual_process_allocation():
    from nanobody_opt import pr100_live_worker as worker
    settings={'resident':True,'gpus':[0,1,2,3],'workers_per_gpu':4,
              'registration':{'policy':{'preparation_threads':1}}}
    resources=worker.native_slot_resources(16,settings,NS(capacity=128),cores=16)
    assert resources=={'queue_slots':128,'scientific_processes':16,'executor_workers':16}
    with pytest.raises(ValueError,match='CPU allocation'):
        worker.native_slot_resources(8,settings,NS(capacity=128),cores=16)


def test_real_refill_admits_128_async_calls_without_128_executor_threads(monkeypatch):
    from nanobody_opt import pr100_native_refill as refill
    counts=[];original=refill.ThreadPoolExecutor;remaining=list(range(128));completed=[]
    def executor(*args,**kwargs):counts.append(kwargs['max_workers']);return original(*args,**kwargs)
    monkeypatch.setattr(refill,'ThreadPoolExecutor',executor)
    admitted=[0];ready=None
    def next_calls(free):
        selected=remaining[:free];del remaining[:free]
        return [{'sequence':str(i),'timeout_seconds':30,'minimum_window':10,'record':lambda *_:None,
                 'record_error':lambda *_:None} for i in selected]
    async def develop(sequences,_concurrency,**_kwargs):
        nonlocal ready
        if ready is None:ready=asyncio.Event()
        admitted[0]+=1
        if admitted[0]==128:ready.set()
        await ready.wait()
        return [{'inert_test_only':True}]
    result=refill.run(next_calls,lambda call,_value:completed.append(call['sequence']),128,
                      develop=develop,executor_workers=16)
    assert counts==[16] and result['maximum_in_flight']==128 and len(completed)==128


@pytest.mark.parametrize('enabled',[None,False,True])
def test_actual_native_phase_selects_resident_only_when_enabled(monkeypatch,tmp_path,enabled):
    from nanobody_opt import pr100_live_worker as worker,pr100_filter
    challenge={'epoch':26030}
    (tmp_path/'input.json').write_text(json.dumps({'challenge':challenge}))
    job={'challenge':challenge,'native_tools':str(tmp_path),'native_reference':{'enabled':True}}
    settings={'registration':{'policy':{'preparation_threads':1}}}
    if enabled is not None:settings['resident']=enabled
    monkeypatch.setattr(worker,'worker_count',lambda _job:4)
    monkeypatch.setattr(pr100_filter,'helper',lambda _name:NS(tool_paths=None))
    monkeypatch.setattr(resident.native,'settings_for',lambda _job:settings)
    class CPUChosen(BaseException):pass
    class ResidentChosen(BaseException):pass
    def cpu(*_args):raise CPUChosen()
    def fast(*_args):raise ResidentChosen()
    monkeypatch.setattr(resident.native,'Adapter',cpu)
    monkeypatch.setattr(resident,'RegisteredResidentAdapter',fast)
    with pytest.raises(ResidentChosen if enabled is True else CPUChosen):
        worker.native_phase(job,tmp_path)


def adapter(tmp_path):
    value=resident.RegisteredResidentAdapter.__new__(resident.RegisteredResidentAdapter)
    value.settings={'gpus':[0],'workers_per_gpu':4,'resident_batch_draws':32,'resident':True,
        'registration':{'policy':POLICY},'bundle':'/qualified/bundle','environment':'/qualified/env',
        'lock_directory':'/owned/locks','scorer':'/qualified/scorer.py'}
    value.source_sha='a'*40;value.directory=tmp_path;value.phase_deadline=time.monotonic()+300;value.sessions={}
    return value


@pytest.mark.parametrize('cpu,libraries',[(True,[]),(False,[]),(False,['/qualified/plugin/lib'])])
def test_models_survive_multiple_request_groups_under_one_lease(monkeypatch,tmp_path,cpu,libraries):
    value=adapter(tmp_path);processes=[]
    if cpu:value.settings['registration']={'policy':CPU_POLICY}
    class Process:
        def __init__(self,*args,**kwargs):self.args=args;self.kwargs=kwargs;processes.append(self)
        def poll(self):return None
    monkeypatch.setattr(resident.subprocess,'Popen',Process)
    def plugin(*_args):
        assert not cpu,'CPU resident must not read sidecar files'
        return {'library_directories':libraries}
    monkeypatch.setattr(resident.native,'cuda_plugin_paths',plugin)
    first=value._start(0);second=value._start(0)
    assert first is second and len(processes)==1
    lease=json.loads((first['root']/'lease-job.json').read_text())
    assert lease['command'][:3]==['/qualified/env/bin/python3.12','-s','-B']
    if cpu or not libraries:
        assert 'LD_LIBRARY_PATH' not in lease['env']
    else:
        assert lease['env']['LD_LIBRARY_PATH']==':'.join(libraries)
        assert '/qualified/env/lib' not in lease['env']['LD_LIBRARY_PATH']
    assert lease['minimum_free_mib']==(20480 if cpu else 10240)
    assert lease['ttl_seconds']<=300 and value.capacity==32
    assert value._batch_limit({'queue':list(range(50)),'waiting_lanes':1})==32


def test_async_groups_keep_fresh_UUIDs_and_never_cache_predictions(monkeypatch,tmp_path):
    value=adapter(tmp_path);root=tmp_path/'session';root.mkdir();(root/'commands').mkdir();(root/'responses').mkdir();(root/'READY.json').write_text('{}')
    session={'root':root,'process':NS(poll=lambda:None)};calls=[]
    monkeypatch.setattr(value,'_start',lambda _gpu:session)
    real_atomic=resident.native.atomic
    def respond(path,command):
        real_atomic(path,command);calls.append(command)
        rows=[{'sequence_hash':hashlib.sha256(s.encode()).hexdigest(),'authoritative':False,'inert_test_only':True} for s in command['request']['sequences']]
        real_atomic(root/'responses'/path.name,{'ok':True,'values':rows})
    monkeypatch.setattr(resident.native,'atomic',respond)
    monkeypatch.setattr(resident.native,'authoritative_measurement',lambda row:row.get('inert_test_only') is True)
    async def run():
        await value._launch(0,['A'*60],100)
        await value._launch(0,['A'*60],100)
    asyncio.run(run())
    assert len(calls)==2 and calls[0]['batch_id']!=calls[1]['batch_id']
    assert calls[0]['request']['directory']!=calls[1]['request']['directory']


def test_clean_exit_recovers_checked_files_and_exact_never_started_markers(monkeypatch,tmp_path):
    value=adapter(tmp_path);batch=tmp_path/'batch';batch.mkdir()
    sequences=['A'*60,'G'*60,'C'*60]
    measured={'sequence_hash':hashlib.sha256(sequences[0].encode()).hexdigest(),
              'authoritative':False,'inert_test_only':True}
    (batch/'0').mkdir();(batch/'0/measurement.json').write_text(json.dumps(measured))
    (batch/'1.started.json').write_text('{}')
    monkeypatch.setattr(resident.native,'authoritative_measurement',lambda row:row.get('inert_test_only') is True)
    values=value._completed_files(batch,sequences)
    assert values[0]==measured
    assert isinstance(values[1],resident.native.Inconclusive) and values[1]['native_started'] is True
    assert isinstance(values[2],resident.native.Inconclusive) and values[2]['native_started'] is False


@pytest.mark.parametrize('clean',[True,False])
def test_phase_shutdown_waits_for_owner_and_requires_fixed_point_cleanup(tmp_path,clean):
    value=adapter(tmp_path);root=tmp_path/'session';root.mkdir();calls=[]
    class Process:
        done=False
        def poll(self):return 0 if self.done else None
        def wait(self,timeout):calls.append(('wait',timeout));self.done=True;return 0
    (root/'lease.json').write_text(json.dumps({'status':'process_finished','released':True,
        'cleanup':{'remaining_descendants':0 if clean else 1,'fixed_point_reached':clean}}))
    value.sessions={0:{'root':root,'process':Process()}}
    if clean:value.shutdown()
    else:
        with pytest.raises(resident.AdapterError,match='unverified_reference_cleanup'):value.shutdown()
    assert (root/'STOP').exists() and calls==[('wait',2)]


def test_cancelled_registered_call_awaits_normal_owner_cleanup(monkeypatch,tmp_path):
    value=adapter(tmp_path);root=tmp_path/'session';root.mkdir()
    (root/'commands').mkdir();(root/'responses').mkdir();(root/'READY.json').write_text('{}')
    calls=[]
    class Process:
        done=False;terminated=False
        def poll(self):return 0 if self.done else None
        def terminate(self):self.terminated=True;calls.append('terminate_owned_lease')
        def wait(self,timeout):
            calls.append(('wait',timeout))
            if not self.terminated:raise subprocess.TimeoutExpired('owned-lease',timeout)
            self.done=True
            (root/'lease.json').write_text(json.dumps({'status':'failed','released':True,
                'cleanup':{'remaining_descendants':0,'fixed_point_reached':True}}))
            return 0
    session={'root':root,'process':Process()};value.sessions={0:session}
    value._states=weakref.WeakKeyDictionary();value._launcher=value._launch
    monkeypatch.setattr(value,'_start',lambda _gpu:session)
    async def run():
        task=asyncio.create_task(value.develop(['A'*60],1,task_timeout_seconds=30))
        while not list((root/'commands').glob('*.json')):await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert session['process'].done
    asyncio.run(run())
    assert calls==[('wait',2),'terminate_owned_lease',('wait',12)]
    assert (root/'STOP').exists()


def test_original_program_context_cwd_failure_does_not_leak_process_lock(monkeypatch, tmp_path):
    def broken_cwd():
        raise FileNotFoundError('cwd removed')
    monkeypatch.setattr(resident.Path, 'cwd', staticmethod(broken_cwd))
    with pytest.raises(FileNotFoundError):
        with resident.original_program_context(['TNP'], tmp_path, None, preparation_threads=1,
                                               trace_path=tmp_path/'trace.jsonl'):
            pass
    assert resident._process_lock.acquire(blocking=False)  # not left held
    resident._process_lock.release()
