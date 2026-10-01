#!/usr/bin/env python3
"""Isolated NIM 1.9 test instance control; credentials stay in memory only.

Never use this helper for production instances. All remote stdout/stderr is
captured; only this helper's explicitly structured scientific events are shown.
"""
import argparse
import base64
import json
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
import sys
import urllib.request

INSTANCE_ID = 51321895
LABEL = "nova-boltz2-nim19-test-20260917"
IMAGE = "nvcr.io/nim/mit/boltz2@sha256:a2eb69458fd9505ce9fdaebb16e206da909666ec8d94a297d81e7b57a2f6c090"
SSH_KEY = "/Users/tvgdeena/.ssh/id_ed25519"


def keychain(service):
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    if result.returncode:
        raise RuntimeError("credential_unavailable")
    return result.stdout.decode().strip()


def instance():
    credential = keychain("vast.ai")
    request = urllib.request.Request(
        "https://console.vast.ai/api/v0/instances/?owner=me",
        headers={"Authorization": "Bearer " + credential},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    credential = None
    found = next(x for x in data.get("instances", []) if x.get("id") == INSTANCE_ID)
    if found.get("label") != LABEL or found.get("num_gpus") != 1:
        raise RuntimeError("wrong_instance")
    return found


def remote(code, payload=None, timeout=3600):
    host = instance()
    mapping = host.get("ports", {}).get("22/tcp") or []
    if not mapping:
        print(json.dumps({"event": "vm_not_ssh_ready", "status": host.get("actual_status")}))
        return 2
    command = [
        "ssh", "-i", SSH_KEY, "-p", str(mapping[0]["HostPort"]),
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
        "root@" + host["public_ipaddr"], "python3 -c " + shlex.quote(code),
    ]
    result = subprocess.run(
        command, input=json.dumps(payload or {}).encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    # These operations emit JSON themselves, never subprocess text or raw HTTP.
    # Authentication-bearing operations are additionally allowlisted per field.
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(event, dict) and event.get("event") in SAFE_EVENTS:
            safe = {k: v for k, v in event.items() if k in SAFE_FIELDS}
            print(json.dumps(safe), flush=True)
    if result.returncode:
        print(json.dumps({"event": "remote_failed", "exit_code": result.returncode}))
    return result.returncode


SAFE_EVENTS = {"preflight", "pull_started", "pull_progress", "pull_complete", "pull_failed", "operation_failed", "service", "schema", "started", "assets", "source", "native_extract", "native_runtime", "benchmark_request", "benchmark_summary"}
SAFE_FIELDS = {"event", "passed", "gpus", "cpu_count", "ram_gb", "disk_free_gb", "docker_version", "gpu_runtime", "elapsed_s", "downloaded_bytes", "total_bytes", "digest_verified", "image_id", "http_status", "container_status", "exit_code", "ready", "request_fields", "checkpoint_files", "missing_files", "seed_env", "affinity_recycles_exposed", "file_count", "verified", "file", "bytes", "sha256", "artifact_bytes", "proot_works", "layer_count", "layers_done", "source_digest", "gpu_visible", "torch_version", "cuda_version", "server_script_present", "runtime_available", "error_class", "error_classes", "supervisor_pid", "server_pid", "ready_now", "elapsed_seconds", "cold_start_seconds", "already_running", "case", "repetition", "phase", "request_sha256", "wall_seconds", "confidence", "structure_present", "metrics", "affinities", "failure", "summary"}
SAFE_FIELDS.update({"startup_stages", "execution_mode"})


def deploy_clients():
    """Transfer only public test programs; never copy runtime credentials/state."""
    host = instance()
    port = str(host["ports"]["22/tcp"][0]["HostPort"])
    mkdir = "import pathlib;pathlib.Path('/root/nim19-test/client').mkdir(parents=True,exist_ok=True)"
    if remote(mkdir, timeout=45):
        return 1
    ssh = ["ssh", "-i", SSH_KEY, "-p", port, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR"]
    files = [str(Path(__file__).with_name(name)) for name in ["nim19_native_runtime.py", "nim19_benchmark.py"]]
    files.append("/Users/tvgdeena/nova-nanobody-standalone/data/msa_files/P40261.a3m")
    result = subprocess.run(["rsync", "-rlt", "--checksum", "--no-owner", "--no-group", "--no-perms", "-e", shlex.join(ssh), *files, "root@"+host["public_ipaddr"]+":/root/nim19-test/client/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    print(json.dumps({"event": "source", "file_count": len(files), "verified": result.returncode == 0, "exit_code": result.returncode}))
    return result.returncode


def native_client(operation):
    program = "nim19_benchmark.py" if operation == "benchmark" else "nim19_native_runtime.py"
    arguments = ["--output", "/root/nim19-test/smoke-results.json", "--target-msa", "/root/nim19-test/client/P40261.a3m"] if operation == "benchmark" else [operation]
    code = DOCKER + "\nresult=subprocess.run(" + repr(["python3", "/root/nim19-test/client/"+program, *arguments]) + ",stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800)\nsys.stdout.buffer.write(result.stdout)\nsys.exit(result.returncode)"
    return remote(code, timeout=1900)


def stop_test_instance(state="stopped"):
    """Exact ID/label/GPU guard, never stop another account instance."""
    host = instance()
    if state not in ("stopped", "running"):
        raise ValueError("invalid_state")
    if state == "stopped" and host.get("cur_state") == "stopped" and host.get("actual_status") in ["stopped", "exited"]:
        print(json.dumps({"event": "service", "container_status": "stopped"}))
        return 0
    credential = keychain("vast.ai")
    request = urllib.request.Request(
        "https://console.vast.ai/api/v0/instances/"+str(INSTANCE_ID)+"/",
        data=json.dumps({"state": state}).encode(), method="PUT",
        headers={"Authorization": "Bearer "+credential, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        status = response.status
        data = json.load(response)
    credential = None
    print(json.dumps({"event": "service", "http_status": status, "passed": data.get("success") is True, "container_status": state+"_requested"}))
    return 0 if data.get("success") is True else 1


def paired_status():
    code = DOCKER + r'''
from pathlib import Path
base=Path('/root/nim19-test')
for name in ['pair-setup.json','pair-runner.json','paired-progress.json']:
    path=base/name
    if path.is_file(): emit('source',file=name,summary=json.loads(path.read_text()))
path=base/'paired-nim.json'
if path.is_file():
    records=json.loads(path.read_text())['records']
    emit('source',file=path.name,summary={'completed':len(records),'passed':sum(x.get('passed',False) for x in records),'last_seconds':records[-1]['wall_seconds'] if records else None})
for name in ['paired-reference-0.json','paired-reference-1.json','paired-optimized-0.json']:
    path=base/name
    if path.is_file():
        data=json.loads(path.read_text())
        emit('source',file=name,summary={'count':len(data['records']),'wall_seconds':data['wall_seconds'],'inference_seconds':sum(x['wall_seconds'] for x in data['predict_step_timings'])})
'''
    return remote(code, timeout=60)


def paired_fetch():
    """Fetch individually named scientific artifacts, verifying remote hashes."""
    names=['paired-portfolio.json','paired-reference-0.json','paired-reference-1.json',
           'paired-optimized-0.json','paired-nim.json','pair-setup.json','pair-runner.json',
           'P40261-history.csv','NIM-contract-audit.json','asset-reverification.json']
    host=instance()
    port=str(host['ports']['22/tcp'][0]['HostPort'])
    ssh=['ssh','-i',SSH_KEY,'-p',port,'-o','BatchMode=yes','-o','ConnectTimeout=15',
         '-o','StrictHostKeyChecking=accept-new','-o','LogLevel=ERROR']
    code="import hashlib,json,tempfile;from pathlib import Path;b=Path('/root/nim19-test');s=Path(tempfile.mkdtemp(prefix='paired-fetch-',dir=b));(s/'.owned-by-paired-fetch').touch();items=[]\nfor n in "+repr(names)+":\n if (b/n).is_file():\n  data=(b/n).read_bytes();(s/n).write_bytes(data);items.append(dict(file=n,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))\nprint(json.dumps(dict(directory=s.name,files=items)))"
    result=subprocess.run([*ssh,'root@'+host['public_ipaddr'],'python3 -c '+shlex.quote(code)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
    if result.returncode:raise RuntimeError('manifest_failed')
    document=json.loads(result.stdout)
    snapshot=document['directory']
    if not re.fullmatch(r'paired-fetch-[a-z0-9_]{8}',snapshot):raise RuntimeError('unsafe_snapshot')
    manifest=document['files']
    destination=Path(__file__).resolve().parents[1]/'data/nim19_benchmark/20260917/paired'
    destination.mkdir(parents=True,exist_ok=True)
    for item in manifest:
        name=item['file']
        if name not in names or not re.fullmatch('[0-9a-f]{64}',item['sha256']) or not isinstance(item['bytes'],int):raise RuntimeError('unsafe_manifest')
        result=subprocess.run(['rsync','-rlt','--checksum','--no-owner','--no-group','--no-perms','-e',shlex.join(ssh),
                               'root@'+host['public_ipaddr']+':/root/nim19-test/'+snapshot+'/'+name,str(destination/name)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        passed=result.returncode==0 and (destination/name).stat().st_size==item['bytes'] and hashlib.sha256((destination/name).read_bytes()).hexdigest()==item['sha256']
        print(json.dumps(dict(event='source',**item,verified=passed)),flush=True)
        if not passed:return 1
    # Remove only this helper's temporary duplicate copies, never source data.
    cleanup="from pathlib import Path;p=Path('/root/nim19-test')/"+repr(snapshot)+";assert (p/'.owned-by-paired-fetch').is_file()\nfor n in "+repr(names)+":\n if (p/n).is_file():(p/n).unlink()\n(p/'.owned-by-paired-fetch').unlink();p.rmdir()"
    result=subprocess.run([*ssh,'root@'+host['public_ipaddr'],'python3 -c '+shlex.quote(cleanup)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    if result.returncode:raise RuntimeError('snapshot_cleanup_failed')
    return 0

DOCKER = r'''
import base64, http.client, json, os, re, socket, subprocess, sys, time
class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, timeout=3600):
        super().__init__('localhost', timeout=timeout)
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect('/var/run/docker.sock')
def docker(method, path, body=None, headers=None):
    conn = UnixHTTP()
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, '/v1.43' + path, body=data, headers=dict({'Content-Type': 'application/json'}, **(headers or {})))
    response = conn.getresponse()
    return conn, response
def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)
'''

PREFLIGHT = DOCKER + r'''
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    gpus = []
    for line in result.stdout.decode().splitlines():
        name, driver, total, free = [x.strip() for x in line.split(',')]
        if not re.fullmatch(r'[A-Za-z0-9 ._-]{1,80}', name) or not re.fullmatch(r'[0-9.]+', driver):
            raise RuntimeError()
        gpus.append(dict(name=name, driver=driver, memory_mb=int(total), free_mb=int(free)))
    with open('/proc/meminfo') as stream:
        ram = int(next(line.split()[1] for line in stream if line.startswith('MemTotal:'))) / 1024**2
    import shutil
    free = shutil.disk_usage('/').free / 1024**3
    conn, response = docker('GET', '/version')
    version = json.load(response).get('Version', '')
    conn.close()
    conn, response = docker('GET', '/info')
    runtime = 'nvidia' in json.load(response).get('Runtimes', {})
    conn.close()
    passed = len(gpus)==1 and gpus[0]['memory_mb'] >= 48000 and tuple(map(int,gpus[0]['driver'].split('.'))) >= (590,44,1) and os.cpu_count() >=12 and ram>=59 and free>=100 and runtime
    emit('preflight', passed=passed, gpus=gpus, cpu_count=os.cpu_count(), ram_gb=round(ram,2), disk_free_gb=round(free,2), docker_version=version if re.fullmatch(r'[0-9.]+',version) else None, gpu_runtime=runtime)
    sys.exit(0 if passed else 3)
except Exception:
    emit('operation_failed')
    sys.exit(1)
'''

PULL = DOCKER + r'''
try:
    from urllib.parse import urlencode
    data = json.load(sys.stdin)
    image = data['image']
    auth = base64.b64encode(json.dumps({'username':'$oauthtoken','password':data['credential'],'serveraddress':'nvcr.io'}).encode()).decode()
    data = None
    started = time.monotonic()
    conn, response = docker('POST', '/images/create?' + urlencode({'fromImage':image}), headers={'X-Registry-Auth':auth})
    auth = None
    emit('pull_started', http_status=response.status)
    if response.status != 200:
        emit('pull_failed', http_status=response.status)
        sys.exit(1)
    progress = {}; last = 0; failed = False
    for line in response:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get('error') or item.get('errorDetail'):
            failed = True
        detail = item.get('progressDetail', {})
        if isinstance(detail.get('current'), (int,float)):
            progress[item.get('id','')] = detail
        if time.monotonic() - last >= 30:
            emit('pull_progress', elapsed_s=round(time.monotonic()-started,1), downloaded_bytes=sum(x.get('current',0) for x in progress.values()), total_bytes=sum(x.get('total',0) for x in progress.values()))
            last = time.monotonic()
    conn.close()
    if failed:
        emit('pull_failed'); sys.exit(1)
    conn, response = docker('GET', '/images/'+image+'/json')
    metadata = json.load(response)
    verified = image in metadata.get('RepoDigests',[])
    identity = metadata.get('Id','')
    emit('pull_complete', digest_verified=verified, image_id=identity if re.fullmatch(r'sha256:[0-9a-f]{64}',identity) else None, elapsed_s=round(time.monotonic()-started,1))
    sys.exit(0 if verified else 1)
except Exception:
    emit('operation_failed'); sys.exit(1)
'''

ASSETS = DOCKER + r'''
try:
    import hashlib
    root = '/root/nim19-test/models'
    os.makedirs(root, exist_ok=True)
    specs = [
        ('boltz-2', 'boltz2_conf.ckpt', 2286561469, '090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1'),
        ('boltz-2', 'boltz2_aff.ckpt', 2062139170, 'dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e'),
        ('boltz-2', 'mols.tar', 1855662080, '39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7'),
        ('boltz-1', 'ccd.pkl', 345859128, '2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4'),
    ]
    # One writer, serial downloads, HTTP/1.1, resumable, exact bytes + SHA-256.
    import fcntl
    with open(root+'/.fetch.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for model, filename, expected, digest in specs:
            path = root+'/'+filename
            if not os.path.exists(path) or os.path.getsize(path)!=expected:
                result = subprocess.run(['curl','--silent','--show-error','--fail','--location','--http1.1','--continue-at','-','--retry','3','--max-time','1800','--output',path,'https://huggingface.co/boltz-community/'+model+'/resolve/main/'+filename],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1900)
                if result.returncode or os.path.getsize(path)!=expected:
                    emit('assets', file=filename, verified=False); sys.exit(1)
            hasher = hashlib.sha256()
            with open(path,'rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                    hasher.update(chunk)
            verified = hasher.hexdigest()==digest
            emit('assets', file=filename, bytes=os.path.getsize(path), sha256=hasher.hexdigest(), verified=verified)
            if not verified:
                sys.exit(1)
    emit('assets', file_count=len(specs), verified=True)
except Exception:
    emit('operation_failed'); sys.exit(1)
'''

START = DOCKER + r'''
try:
    data = json.load(sys.stdin)
    files = ['boltz2_conf.ckpt','boltz2_aff.ckpt','ccd.pkl','mols.tar']
    missing = [x for x in files if not os.path.isfile('/root/nim19-test/models/'+x)]
    if missing:
        emit('started', missing_files=missing); sys.exit(1)
    for path in ['/root/nim19-test/cache','/root/nim19-test/output']:
        os.makedirs(path,exist_ok=True)
        os.chmod(path,0o777)
    body = {
        'Image':data['image'],
        'Env':['MODEL_PATH=/opt/nim/custom-model','NIM_DISABLE_MODEL_DOWNLOAD=1','NIM_DEFAULT_RANDOM_SEED=68','NIM_TELEMETRY_MODE=0','NIM_LOG=WARNING','NIM_LOG_LEVEL=WARNING','NIM_OUTPUT_PATH=/opt/nim/output'],
        'HostConfig': {
            'ShmSize':16*1024**3,
            'Binds':['/root/nim19-test/models:/opt/nim/custom-model:ro','/root/nim19-test/cache:/opt/nim/.cache','/root/nim19-test/output:/opt/nim/output'],
            'DeviceRequests':[{'Driver':'nvidia','Count':1,'Capabilities':[['gpu']]}],
            'PortBindings':{'8000/tcp':[{'HostIp':'127.0.0.1','HostPort':'8000'}]},
        },
        'ExposedPorts':{'8000/tcp':{}},
    }
    conn,response = docker('POST','/containers/create?name=nova-nim19-test',body)
    status=response.status
    response.read(); conn.close()
    if status not in [201,409]:
        emit('started',http_status=status);sys.exit(1)
    conn,response = docker('POST','/containers/nova-nim19-test/start')
    status=response.status;response.read();conn.close()
    emit('started', http_status=status, seed_env=68)
    sys.exit(0 if status in [204,304] else 1)
except Exception:
    emit('operation_failed');sys.exit(1)
'''

SERVICE = DOCKER + r'''
try:
    import urllib.request
    conn,response = docker('GET','/containers/nova-nim19-test/json')
    data=json.load(response);conn.close()
    state=data.get('State',{})
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/v1/health/ready',timeout=5) as response:
            ready=response.status==200
    except Exception:
        ready=False
    emit('service',container_status=state.get('Status'),exit_code=state.get('ExitCode'),ready=ready)
    conn,response = docker('GET','/images/json')
    data=json.load(response);conn.close()
    emit('service',artifact_bytes=sum(x.get('Size',0) for x in data))
except Exception:
    emit('operation_failed');sys.exit(1)
'''

SCHEMA = DOCKER + r'''
try:
    import urllib.request
    with urllib.request.urlopen('http://127.0.0.1:8000/openapi.json',timeout=10) as response:
        data=json.load(response)
    operation=data['paths']['/biology/mit/boltz2/predict']['post']
    schema=operation['requestBody']['content']['application/json']['schema']
    if '$ref' in schema:
        schema=data['components']['schemas'][schema['$ref'].split('/')[-1]]
    fields=sorted(x for x in schema.get('properties',{}) if re.fullmatch(r'[A-Za-z0-9_]{1,100}',x))
    emit('schema', request_fields=fields, affinity_recycles_exposed='recycling_steps_affinity' in fields)
except Exception:
    emit('operation_failed');sys.exit(1)
'''

NATIVE_PREFLIGHT = DOCKER + r'''
try:
    import shutil
    result=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free','--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    gpus=[]
    for line in result.stdout.decode().splitlines():
        name,driver,total,free=[x.strip() for x in line.split(',')]
        if not re.fullmatch(r'[A-Za-z0-9 ._-]{1,80}',name) or not re.fullmatch(r'[0-9.]+',driver):raise RuntimeError()
        gpus.append(dict(name=name,driver=driver,memory_mb=int(total),free_mb=int(free)))
    with open('/proc/meminfo') as stream:
        ram=int(next(line.split()[1] for line in stream if line.startswith('MemTotal:')))/1024**2
    free=shutil.disk_usage('/root').free/1024**3
    passed=len(gpus)==1 and gpus[0]['memory_mb']>=48000 and tuple(map(int,gpus[0]['driver'].split('.'))) >= (590,44,1) and ram>=59 and free>=100
    emit('preflight',passed=passed,gpus=gpus,cpu_count=os.cpu_count(),ram_gb=round(ram,2),disk_free_gb=round(free,2))
    if not passed:sys.exit(3)
    env=dict(os.environ,DEBIAN_FRONTEND='noninteractive')
    for command in [['apt-get','update'],['apt-get','install','-y','umoci','proot']]:
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=240)
        if result.returncode:
            emit('native_runtime',runtime_available=False,exit_code=result.returncode);sys.exit(1)
    result=subprocess.run(['proot','-r','/','/bin/true'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
    emit('native_runtime',proot_works=result.returncode==0)
    sys.exit(0 if result.returncode==0 else 1)
except Exception:
    emit('operation_failed');sys.exit(1)
'''

NATIVE_EXTRACT = DOCKER + r'''
try:
    import hashlib, pathlib, shutil, urllib.request, urllib.parse, concurrent.futures
    data=json.load(sys.stdin)
    secret=data['credential'];data=None
    root=pathlib.Path('/root/nim19-test/oci')
    blobs=root/'blobs'/'sha256';blobs.mkdir(parents=True,exist_ok=True)
    source_digest='sha256:1d2b8cd39962a7841908e5fff6ea9cd76eb76c000ca19690f8582277376d88d7'
    class Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):
            if urllib.parse.urlparse(newurl).scheme!='https':raise RuntimeError()
            result=super().redirect_request(req,fp,code,msg,headers,newurl)
            if urllib.parse.urlparse(req.full_url).hostname!=urllib.parse.urlparse(newurl).hostname:
                result.remove_header('Authorization')
            return result
    opener=urllib.request.build_opener(Redirect())
    auth_url='https://nvcr.io/proxy_auth?'+urllib.parse.urlencode({'service':'nvcr.io','scope':'repository:nim/mit/boltz2:pull'})
    request=urllib.request.Request(auth_url,headers={'Authorization':'Basic '+base64.b64encode(('$oauthtoken:'+secret).encode()).decode()})
    with opener.open(request,timeout=40) as response:auth=json.load(response)
    token=auth.get('token') or auth.get('access_token');auth=None;secret=None
    if not token:raise RuntimeError()
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'}
    request=urllib.request.Request('https://nvcr.io/v2/nim/mit/boltz2/manifests/'+source_digest,headers=headers)
    with opener.open(request,timeout=40) as response:raw=response.read()
    if 'sha256:'+hashlib.sha256(raw).hexdigest()!=source_digest:raise RuntimeError()
    manifest=json.loads(raw);raw=None
    layers=manifest['layers']
    emit('native_extract',layer_count=len(layers),total_bytes=sum(x['size'] for x in layers),source_digest=source_digest)
    started=time.monotonic()
    def download(descriptor):
        digest=descriptor['digest']
        if not re.fullmatch(r'sha256:[0-9a-f]{64}',digest):raise RuntimeError()
        path=blobs/digest.split(':')[1]
        if path.exists() and path.stat().st_size==descriptor['size']:
            hasher=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024),b''):hasher.update(chunk)
            if 'sha256:'+hasher.hexdigest()==digest:return descriptor['size']
        # Unique immutable layer per writer. No shared append/download race.
        request=urllib.request.Request('https://nvcr.io/v2/nim/mit/boltz2/blobs/'+digest,headers={'Authorization':'Bearer '+token})
        partial=path.with_suffix('.partial')
        hasher=hashlib.sha256();count=0
        with opener.open(request,timeout=120) as response, partial.open('wb') as stream:
            while True:
                chunk=response.read(4*1024*1024)
                if not chunk:break
                stream.write(chunk);hasher.update(chunk);count+=len(chunk)
        if count!=descriptor['size'] or 'sha256:'+hasher.hexdigest()!=digest:raise RuntimeError()
        partial.replace(path)
        return count
    # OCI manifests can reference an identical blob more than once. Download
    # unique digests only, so two workers never own the same partial file.
    descriptors=list({x['digest']:x for x in [manifest['config']]+layers}.values())
    done=0;size=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(download,x) for x in descriptors]):
            size+=future.result();done+=1
            if done%10==0 or done==len(descriptors):emit('native_extract',layers_done=done,downloaded_bytes=size,elapsed_s=round(time.monotonic()-started,1))
    token=None
    manifest['mediaType']='application/vnd.oci.image.manifest.v1+json'
    manifest['config']['mediaType']='application/vnd.oci.image.config.v1+json'
    for layer in manifest['layers']:
        layer['mediaType']='application/vnd.oci.image.layer.v1.tar+gzip'
    encoded=json.dumps(manifest,separators=(',',':')).encode()
    converted=hashlib.sha256(encoded).hexdigest()
    (blobs/converted).write_bytes(encoded)
    (root/'oci-layout').write_text(json.dumps({'imageLayoutVersion':'1.0.0'}))
    (root/'index.json').write_text(json.dumps({'schemaVersion':2,'manifests':[{'mediaType':manifest['mediaType'],'digest':'sha256:'+converted,'size':len(encoded),'annotations':{'org.opencontainers.image.ref.name':'nim19'}}]}))
    config=json.loads((blobs/manifest['config']['digest'].split(':')[1]).read_text())
    # Persist only public image runtime metadata, never registry credentials.
    safeconfig={'Env':config['config'].get('Env',[]),'Cmd':config['config'].get('Cmd',[]),'WorkingDir':config['config'].get('WorkingDir','')}
    pathlib.Path('/root/nim19-test/image-runtime.json').write_text(json.dumps(safeconfig))
    bundle='/root/nim19-test/bundle'
    if not os.path.exists(bundle):
        result=subprocess.run(['umoci','unpack','--rootless','--image',str(root)+':nim19',bundle],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800)
        if result.returncode:
            emit('native_extract',verified=False,exit_code=result.returncode);sys.exit(1)
        pathlib.Path('/root/nim19-test/unpack-complete').write_text(source_digest)
    elif not pathlib.Path('/root/nim19-test/unpack-complete').is_file():
        # Never accept a partial bundle left by an interrupted unpack.
        emit('native_extract',verified=False,error_class='IncompleteBundle');sys.exit(1)
    marker=pathlib.Path(bundle)/'rootfs'/'opt'/'nim'/'start_server.sh'
    emit('native_extract',verified=marker.is_file(),server_script_present=marker.is_file(),elapsed_s=round(time.monotonic()-started,1),source_digest=source_digest)
    sys.exit(0 if marker.is_file() else 1)
except Exception as error:
    emit('operation_failed',error_class=type(error).__name__);sys.exit(1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["status", "preflight", "pull", "assets", "start", "service", "schema", "native-preflight", "native-extract", "deploy-clients", "native-probe", "native-start", "native-direct-probe", "native-direct-start", "native-status", "benchmark", "stop", "resume", "paired-status", "paired-fetch"])
    args = parser.parse_args()
    try:
        if args.operation == "status":
            host = instance()
            safe = {k:host.get(k) for k in ["id", "label", "actual_status", "cur_state", "gpu_name", "num_gpus", "public_ipaddr"]}
            safe["direct_ssh_port"] = (host.get("ports", {}).get("22/tcp") or [{}])[0].get("HostPort")
            print(json.dumps(safe))
            return 0
        if args.operation == "preflight":
            return remote(PREFLIGHT, timeout=90)
        if args.operation == "pull":
            # Fresh preflight is mandatory before the 10 GB image transfer.
            result = remote(PREFLIGHT, timeout=90)
            if result:
                return result
            return remote(PULL, {"image":IMAGE, "credential":keychain("NGC_API_KEY")})
        if args.operation == "assets":
            return remote(ASSETS, timeout=7400)
        if args.operation == "start":
            return remote(START, {"image":IMAGE}, timeout=90)
        if args.operation == "service":
            return remote(SERVICE, timeout=60)
        if args.operation == "schema":
            return remote(SCHEMA, timeout=60)
        if args.operation == "native-preflight":
            return remote(NATIVE_PREFLIGHT, timeout=600)
        if args.operation == "native-extract":
            return remote(NATIVE_EXTRACT, {"credential":keychain("NGC_API_KEY")}, timeout=5400)
        if args.operation == "deploy-clients":
            return deploy_clients()
        if args.operation in ["native-probe", "native-start", "native-direct-probe", "native-direct-start", "native-status", "benchmark"]:
            return native_client(args.operation.removeprefix("native-"))
        if args.operation == "stop":
            return stop_test_instance()
        if args.operation == "resume":
            return stop_test_instance("running")
        if args.operation == "paired-status":
            return paired_status()
        if args.operation == "paired-fetch":
            return paired_fetch()
    except Exception:
        print(json.dumps({"event":"operation_failed"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
