"""Public-input caching must preserve bytes, launch identity and deadline ownership."""
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from nanobody_opt import pr100_archive_cache as cache, pr100_live_controller as c


@pytest.fixture(autouse=True)
def empty_hints():
    cache._ACKNOWLEDGED.clear()
    yield
    cache._ACKNOWLEDGED.clear()


@pytest.fixture
def case(tmp_path):
    raw = b'sequence,epoch\nPUBLIC_INERT_SEQUENCE,25969\n' * 128
    sha = hashlib.sha256(raw).hexdigest()
    root = tmp_path / 'controller'; release = tmp_path / 'release'; release.mkdir()
    source = 'a' * 40
    (release / 'build_stamp.json').write_text(json.dumps({'git_sha': source, 'files': {}}))
    request = {'operation': 'start', 'directory': str(root / '25969/attempts/00001/native'),
        'source_sha': source, 'python': sys.executable, 'supervisor': '/inert/supervisor.py',
        'transport_deadline_unix': time.time() + 120,
        'clock': {'epoch': 25969, 'block': 25969*361+10, 'epoch_length': 361, 'updated_at_unix': time.time()},
        'spec': {'epoch': 25969, 'cwd': str(release), 'environment': {}, 'wall_seconds': 60,
                 'deadline_unix': time.time()+100},
        'files': {'archive.csv': base64.b64encode(raw).decode(),
                  'job.json': base64.b64encode(json.dumps({'archive_sha256': sha}).encode()).decode(),
                  'input.json': base64.b64encode(b'{"unchanged":true}').decode()}}
    config = {'remote_root': str(root), 'hosts': {'master': {'address': 'inert.invalid', 'port': 22},
                                               'slave': {'address': 'other.invalid', 'port': 22}}}
    return config, request, raw, root / 'archive-cache-v1' / (sha + '.csv')


def hydrate(request):
    namespace = {}; exec(cache.RECEIVER, namespace)
    return namespace['hydrate_archive'](request)


def reference(config, request):
    descriptor = cache.prepare(config, 'master', request)
    cache.acknowledge(descriptor, {'archive_cache_verified_sha256': descriptor['descriptor']['sha256']})
    return cache.wire_request(request, cache.prepare(config, 'master', request))


def test_upload_then_hit_preserves_exact_archive_job_and_durable_intent(case):
    config, request, raw, path = case; original = copy.deepcopy(request)
    cold = cache.wire_request(request, cache.prepare(config, 'master', request))
    assert hydrate(cold)['verified'] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw and stat.S_IMODE(path.stat().st_mode) == 0o444
    hot = reference(config, request)
    assert 'archive.csv' not in hot['files'] and hot['spec'] == request['spec']
    assert hydrate(hot)['verified'] == hashlib.sha256(raw).hexdigest()
    assert hot['files'] == request['files'] and request == original


@pytest.mark.parametrize('fault', ['missing', 'changed', 'symlink', 'directory', 'wrong_size'])
def test_unusable_cached_artifact_is_a_miss_and_never_scientific_evidence(case, fault):
    config, request, raw, path = case; hot = reference(config, request)
    path.parent.mkdir(parents=True)
    if fault == 'changed': path.write_bytes(b'x' * len(raw))
    elif fault == 'symlink':
        target = path.parent / 'unrelated'; target.write_bytes(raw); path.symlink_to(target)
    elif fault == 'directory': path.mkdir()
    elif fault == 'wrong_size': path.write_bytes(raw + b'x')
    assert hydrate(hot) == {'miss': hashlib.sha256(raw).hexdigest()}
    assert 'archive.csv' not in hot['files']
    assert not Path(request['directory']).exists()


@pytest.mark.parametrize('fault', ['digest', 'size', 'job', 'epoch', 'root', 'deadline', 'cache_symlink'])
def test_binding_and_deadline_fail_closed_before_launch_or_materialization(case, fault):
    config, request, raw, path = case
    wire = cache.wire_request(request, cache.prepare(config, 'master', request))
    if fault == 'digest': wire['archive_cache']['sha256'] = 'b'*64
    elif fault == 'size': wire['archive_cache']['size'] += 1
    elif fault == 'job': wire['files']['job.json'] = base64.b64encode(b'{"archive_sha256":"wrong"}').decode()
    elif fault == 'epoch': wire['spec'] = {**wire['spec'], 'epoch': 25970}
    elif fault == 'root': wire['archive_cache']['root'] = str(path.parent / '.ssh')
    elif fault == 'deadline': wire['spec'] = {**wire['spec'], 'deadline_unix': time.time()-1}
    else:
        path.parent.parent.mkdir(parents=True); target=path.parent.parent/'other'; target.mkdir(); path.parent.symlink_to(target)
    with pytest.raises((ValueError, TimeoutError)): hydrate(wire)
    assert not Path(request['directory']).exists()


def test_concurrent_identical_uploads_are_atomic_and_leave_no_temporary_files(case):
    config, request, raw, path = case
    def upload(_): return hydrate(cache.wire_request(request, cache.prepare(config, 'master', request)))
    with ThreadPoolExecutor(max_workers=8) as pool: replies = list(pool.map(upload, range(16)))
    assert all(r.get('verified') == hashlib.sha256(raw).hexdigest() for r in replies)
    assert path.read_bytes() == raw
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_cache_write_failure_preserves_verified_full_payload_without_false_cache_hint(case, monkeypatch):
    import os
    config, request, raw, path = case
    wire = cache.wire_request(request, cache.prepare(config, 'master', request))
    def unavailable(*_a, **_k): raise PermissionError('inert cache write failure')
    monkeypatch.setattr(os, 'replace', unavailable)
    result = hydrate(wire)
    assert result == {'materialized': hashlib.sha256(raw).hexdigest()}
    assert base64.b64decode(wire['files']['archive.csv']) == raw
    assert not path.exists() and list(path.parent.iterdir()) == []


def test_attempt_copy_cannot_modify_reusable_archive(case):
    config, request, raw, path = case
    receiver(cache.wire_request(request, cache.prepare(config, 'master', request)))
    (Path(request['directory']) / 'archive.csv').write_bytes(b'inert per-attempt mutation')
    assert path.read_bytes() == raw
    hot = reference(config, request)
    assert hydrate(hot)['verified'] == hashlib.sha256(raw).hexdigest()


def test_acknowledgement_is_bound_to_host_root_and_exact_digest(case):
    config, request, *_ = case
    hot = reference(config, request)
    assert 'archive.csv' not in hot['files']
    assert cache.prepare(config, 'slave', request)['reference_only'] is False
    different = copy.deepcopy(request); different['files']['archive.csv'] = base64.b64encode(b'different').decode()
    digest = hashlib.sha256(b'different').hexdigest()
    different['files']['job.json'] = base64.b64encode(json.dumps({'archive_sha256': digest}).encode()).decode()
    assert cache.prepare(config, 'master', different)['reference_only'] is False
    assert cache.prepare({**config, 'archive_cache_enabled': False}, 'master', request) is None


def decode(raw): return json.loads(gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw)


def test_missing_or_corrupt_cache_retries_full_bytes_same_identity_and_deadline(case, monkeypatch):
    config, request, raw, path = case; reference(config, request); calls=[]; original=copy.deepcopy(request)
    def run(_command, *, input, **kwargs):
        wire = decode(input); calls.append(wire)
        result = hydrate(wire)
        reply = {'running': False, 'archive_cache_miss': result['miss']} if 'miss' in result else {
            'running': True, 'archive_cache_verified_sha256': result['verified']}
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply).encode(), stderr=b'')
    monkeypatch.setattr(c.subprocess, 'run', run)
    assert c.remote(config, 'master', request)['running'] is True
    assert len(calls) == 2 and 'archive.csv' not in calls[0]['files']
    assert base64.b64decode(calls[1]['files']['archive.csv']) == raw
    assert calls[0]['spec'] == calls[1]['spec'] == request['spec']
    assert calls[0]['transport_deadline_unix'] == calls[1]['transport_deadline_unix']
    assert request == original


def test_cache_fallback_cannot_borrow_time_after_absolute_deadline(case, monkeypatch):
    config, request, *_ = case; reference(config, request)
    now=[1000.];request['transport_deadline_unix']=1002.; calls=[]
    monkeypatch.setattr(c.time,'time',lambda:now[0]);monkeypatch.setattr(c.time,'monotonic',lambda:now[0])
    def run(_command, *, input, **kwargs):
        calls.append(decode(input));now[0]=1002.
        return SimpleNamespace(returncode=0,stdout=json.dumps({'archive_cache_miss':calls[-1]['archive_cache']['sha256']}).encode(),stderr=b'')
    monkeypatch.setattr(c.subprocess,'run',run)
    with pytest.raises(TimeoutError,match='epoch deadline'):c.remote(config,'master',request)
    assert len(calls)==1


def receiver(request):
    # Exercise the actual wire receiver while replacing only OS process launch
    # with an inert marker; no scientific process or remote connection exists.
    hook='''import subprocess,pathlib,json,types
def inert_launch(*args,**kwargs):
 spec=json.loads(pathlib.Path(args[0][-1]).read_text());root=pathlib.Path(spec['cwd']);p=root/'launch-count'
 p.write_text(str(int(p.read_text())+1 if p.exists() else 1));return types.SimpleNamespace(pid=123)
subprocess.Popen=inert_launch
'''
    result=subprocess.run([sys.executable,'-c',hook+c.TRANSPORT],input=json.dumps(request).encode(),capture_output=True,timeout=5)
    assert result.returncode==0, result.stderr.decode()
    return json.loads(result.stdout)


def test_actual_receiver_cache_hit_materializes_bytes_and_never_relaunches_owner(case):
    config, request, raw, path=case
    cold=cache.wire_request(request,cache.prepare(config,'master',request));receiver(cold)
    assert (Path(request['directory'])/'archive.csv').read_bytes()==raw
    next_request=copy.deepcopy(request);next_request['directory']=str(Path(request['directory']).parent.parent/'00002/native')
    receiver(reference(config,next_request))
    assert (Path(next_request['directory'])/'archive.csv').read_bytes()==raw
    receiver(reference(config,next_request))
    assert (Path(request['spec']['cwd'])/'launch-count').read_text()=='2'


def test_actual_receiver_cache_miss_and_ambiguous_owner_do_not_launch(case):
    config,request,raw,path=case;hot=reference(config,request)
    assert receiver(hot)['archive_cache_miss']==hashlib.sha256(raw).hexdigest()
    directory=Path(request['directory']);assert not (directory/'spec.json').exists()
    (directory/'spec.json').write_text('{}')
    answer=receiver(hot)
    assert answer['running'] is True and answer['launch_uncertain'] is True
    assert not (Path(request['spec']['cwd'])/'launch-count').exists()


def test_realistic_cache_reference_has_small_wire_bytes_without_changing_saved_input(case):
    config,request,raw,path=case
    reference_request=reference(config,request)
    assert 'archive.csv' in request['files'] and 'archive.csv' not in reference_request['files']
    assert reference_request['files']['job.json']==request['files']['job.json']
    assert len(json.dumps(reference_request))<3000
