"""Exact archive transport reuse; never cache scientific observations or verdicts."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import threading

MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_ACKNOWLEDGED = set()
_LOCK = threading.Lock()


def prepare(config, host, request):
    """Keep durable request untouched; return its optional cache wire descriptor."""
    if (config.get('archive_cache_enabled', True) is not True or request.get('operation') != 'start'
            or 'archive.csv' not in request.get('files', {}) or not config.get('remote_root')):
        return None
    root = Path(config['remote_root'])
    if not root.is_absolute():
        raise ValueError('archive cache root must be absolute')
    raw = base64.b64decode(request['files']['archive.csv'], validate=True)
    if not 0 < len(raw) <= MAX_ARCHIVE_BYTES:
        raise ValueError('invalid cached archive size')
    digest = hashlib.sha256(raw).hexdigest()
    job = json.loads(base64.b64decode(request['files']['job.json'], validate=True))
    if job.get('archive_sha256') != digest:
        raise ValueError('archive cache job binding mismatch')
    runtime = config['hosts'][host]
    key = (runtime['address'], runtime['port'], runtime.get('host_key_alias'), str(root), digest)
    with _LOCK:
        reference_only = key in _ACKNOWLEDGED
    return {'key': key, 'reference_only': reference_only and not config.get('_archive_force_upload', False),
            'descriptor': {'schema': 'pr100-archive-cache-v1', 'root': str(root),
                           'sha256': digest, 'size': len(raw)}}


def wire_request(request, cache):
    if cache is None:
        return request
    files = dict(request['files'])
    if cache['reference_only']:
        files.pop('archive.csv')
    return {**request, 'files': files, 'archive_cache': cache['descriptor']}


def acknowledge(cache, reply):
    if cache is None or reply.get('archive_cache_verified_sha256') != cache['descriptor']['sha256']:
        return
    with _LOCK:
        if len(_ACKNOWLEDGED) >= 512:
            _ACKNOWLEDGED.clear()  # Losing a hint costs a full upload, never correctness.
        _ACKNOWLEDGED.add(cache['key'])


def invalidate(cache):
    with _LOCK:
        _ACKNOWLEDGED.discard(cache['key'])


# This stdlib-only function is embedded in the authenticated SSH receiver, so
# cache recovery does not depend on importing an unverified remote module.
RECEIVER = r'''
def hydrate_archive(r):
 import base64,hashlib,json,math,os,pathlib,re,stat,tempfile,time
 value=r.get('archive_cache')
 if value is None:return None
 if (not isinstance(value,dict) or set(value)!={'schema','root','sha256','size'}
  or value['schema']!='pr100-archive-cache-v1' or not isinstance(value['sha256'],str)
  or re.fullmatch('[0-9a-f]{64}',value['sha256']) is None
  or type(value['size'])is not int or not 0<value['size']<=32*1024*1024):raise ValueError('archive cache descriptor')
 root=pathlib.Path(value['root']);directory=pathlib.Path(r['directory'])
 forbidden={'.env','.netrc','.npmrc','.git','.ssh','.aws','.azure','credentials','cookies'}
 if (not root.is_absolute() or '..' in root.parts or any(p in forbidden or p.startswith('.env.') for p in root.parts)
  or not directory.is_relative_to(root) or '..' in directory.parts):raise ValueError('archive cache root')
 relative=directory.relative_to(root)
 if not relative.parts or relative.parts[0]!=str(r['spec']['epoch']):raise ValueError('archive cache epoch binding')
 job=json.loads(base64.b64decode(r['files']['job.json'],validate=True))
 if job.get('archive_sha256')!=value['sha256']:raise ValueError('archive cache job binding')
 deadline=min(float(r['spec']['deadline_unix']),float(r.get('transport_deadline_unix',r['spec']['deadline_unix'])))
 if not math.isfinite(deadline) or time.time()>=deadline:raise TimeoutError('archive cache deadline')
 cache=root/'archive-cache-v1';path=cache/(value['sha256']+'.csv')
 if cache.is_symlink():raise ValueError('archive cache directory symlink')
 raw=None
 if 'archive.csv' in r['files']:
  raw=base64.b64decode(r['files']['archive.csv'],validate=True)
 else:
  try:
   fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
   with os.fdopen(fd,'rb') as stream:
    info=os.fstat(stream.fileno())
    if stat.S_ISREG(info.st_mode) and info.st_size==value['size']:raw=stream.read(value['size']+1)
  except OSError:pass
 if raw is None or len(raw)!=value['size'] or hashlib.sha256(raw).hexdigest()!=value['sha256']:
  if 'archive.csv' in r['files']:raise ValueError('uploaded archive digest mismatch')
  return {'miss':value['sha256']}
 if time.time()>=deadline:raise TimeoutError('archive cache deadline')
 stored=True
 if 'archive.csv' in r['files']:
  name=None
  try:
   cache.mkdir(parents=True,exist_ok=True)
   if cache.is_symlink():raise ValueError('archive cache directory symlink')
   fd,name=tempfile.mkstemp(prefix='.archive-',dir=cache)
   with os.fdopen(fd,'wb') as stream:
    stream.write(raw);stream.flush();os.fsync(stream.fileno());os.fchmod(stream.fileno(),0o444)
   if time.time()>=deadline:raise TimeoutError('archive cache deadline')
   os.replace(name,path)
   fd=os.open(cache,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
   try:os.fsync(fd)
   finally:os.close(fd)
  except TimeoutError:raise
  except OSError:stored=False
  finally:
   if name is not None:
    try:os.unlink(name)
    except FileNotFoundError:pass
 r['files']['archive.csv']=base64.b64encode(raw).decode()
 return {'verified' if stored else 'materialized':value['sha256']}
'''
