"""Pi-owned, epoch-clocked PR100 controller: current rules, fresh measurements, monotone fleets.

Timing comes from the chain clock, never from a wall-clock constant. The usable window runs
from epoch start to the publication cutoff (``epoch_end_block - RESERVE_BLOCKS``, 12 s blocks)
and each phase cap is a share of the time that is actually left when the phase launches, so
caps can never oversubscribe the window. A partial fleet is published as soon as the first
scored margin-safe (tier A) candidates exist, and republished only when the new set is at
least as good: tier first, then margin (see docs/PR100_LIVE_CONTRACT.md).
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
import urllib.request

from nanobody_opt import pr100_filter, rules

BLOCK_SECONDS = 12.0
# Publication cutoff reserve (offset ~315 of 361); the Pi export path itself refuses to
# publish with 40 or fewer blocks left, so this stays strictly above that floor.
RESERVE_BLOCKS = 46
MINIMUM_RESERVE_BLOCKS = 40
# Start whenever more than this many blocks remain: a late partial fleet beats skipping the
# epoch, but the native phase needs its per-candidate budget plus the novelty pass (~600 s),
# so a start with fewer than ~190 blocks left publishes nothing.
START_MINIMUM_BLOCKS = 190
# Time held back from the phase budget for building, validating and publishing the fleet.
PUBLISH_RESERVE_SECONDS = 150.0
# Share of the remaining window a phase may consume; the last phase gets what is left.
PHASE_SHARES = {'sequence': 0.25, 'native': 0.5, 'score': 1.0}
PHASE_FLOORS = {'sequence': 60.0, 'native': 90.0, 'score': 90.0}
TAIL_BYTES = 2048
_JOURNAL_COLLECTOR = None
# Wall reserved at the end of the native phase for the archive novelty pass (234 s in 25956).
NOVELTY_RESERVE_SECONDS = 300.0
ACTIVATION_EPOCH = 25957
TIER_RANK = {'A': 0, 'B': 1}
HOST_KEYS = ('address', 'port', 'release', 'annotation_python', 'native_python',
             'scoring_python', 'python_path', 'path', 'references', 'scorer', 'scorer_settings')
_SECRET = re.compile(r'(?i)(authorization|bearer|token|secret|password|api[_-]?key|private[_-]?key)'
                     r'(\s*[:=]\s*|\s+)\S+')
_LONG_OPAQUE = re.compile(r'\b[A-Za-z0-9+/_-]{48,}\b')

TRANSPORT = r'''
import base64,hashlib,json,os,pathlib,re,stat,subprocess,sys,time
r=json.load(sys.stdin);d=pathlib.Path(r['directory'])
if r['operation']=='journal':
 out={'status':'unavailable'}
 try:
  fd=os.open(d/'native-observations.jsonl',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
  with os.fdopen(fd,'rb') as stream:
   info=os.fstat(stream.fileno())
   if not stat.S_ISREG(info.st_mode):out={'status':'not_regular'}
   elif info.st_size>4*1024*1024:out={'status':'oversized'}
   else:
    raw=stream.read(4*1024*1024+1)
    if len(raw)>4*1024*1024:out={'status':'oversized'}
    else:out={'status':'collected','sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()}
 except OSError:pass
 print(json.dumps(out));sys.exit(0)
d.mkdir(parents=True,exist_ok=True)
def atom(p,v):
 t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(v,allow_nan=False));os.replace(t,p)
atom(d/'clock.json',r['clock'])
if r['operation']=='start':
 release=pathlib.Path(r['spec']['cwd']);stamp=json.loads((release/'build_stamp.json').read_text())
 if stamp['git_sha']!=r['source_sha']:raise ValueError('source revision')
 for name,expected in stamp['files'].items():
  if hashlib.sha256((release/name).read_bytes()).hexdigest()!=expected:raise ValueError('source identity')
 identity=hashlib.sha256(json.dumps({k:v for k,v in r['spec'].items() if k not in('wall_seconds','deadline_unix')},sort_keys=True).encode()).hexdigest()
 if (d/'launch.json').exists():
  old=json.loads((d/'launch.json').read_text())
  if old['identity']!=identity:raise ValueError('different existing job')
 else:
  for name,raw in r['files'].items():
   if name not in ('input.json','archive.csv','job.json','gpu-job.json'):raise ValueError('file name')
   (d/name).write_bytes(base64.b64decode(raw,validate=True))
  atom(d/'spec.json',r['spec'])
  log=open(d/'transport-launch.log','ab')
  p=subprocess.Popen([r['python'],'-B',r['supervisor'],'--spec',str(d/'spec.json')],
   cwd=r['spec']['cwd'],env={**os.environ,**r['spec']['environment']},
   stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
  atom(d/'launch.json',{'identity':identity,'pid':p.pid})
out={'running':not(d/'supervisor-receipt.json').exists()}
if not out['running']:
 out['receipt']=json.loads((d/'supervisor-receipt.json').read_text());out['files']={}
 for name in ('result.json','worker-receipt.json','score/report.json','score/scorer_settings.json'):
  p=d/name
  if p.is_file():out['files'][name]=base64.b64encode(p.read_bytes()).decode()
 report=d/'score/report.json'
 if report.is_file():
  for group in json.loads(report.read_text()).get('groups',[]):
   f=str(group.get('file',''))
   if group.get('valid') is True and re.fullmatch(r'(?:wave-[0-9]+/)?gpu-[0-9]+/length[0-9]+/exact_all[.]json',f) and (d/'score'/f).is_file():
    out['files']['score/'+f]=base64.b64encode((d/'score'/f).read_bytes()).decode()
else:
 for name in ('supervisor-stderr.log','transport-launch.log'):
  p=d/name
  if p.is_file():
   with open(p,'rb') as h:
    h.seek(0,2);s=h.tell();h.seek(max(0,s-2048))
    out.setdefault('tails',{})[name]=h.read(2048).decode('utf-8','replace')
print(json.dumps(out))
'''


def atomic(path, value):
    pr100_filter.atomic(path, value)


def sanitize(text, limit=TAIL_BYTES):
    """Bounded, credential-free text that is safe to store in a receipt."""
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'replace')
    text = str(text)[-limit:]
    text = _SECRET.sub(lambda m: m.group(1) + '=<redacted>', text)
    return _LONG_OPAQUE.sub('<redacted>', text)


def clock_state(chain):
    block = int(chain.get_current_block())
    length = int(chain.tempo(68)) + 1
    return {'block': block, 'epoch': block // length, 'epoch_length': length,
            'updated_at_unix': time.time()}


def blocks_remaining(clock, epoch, *, now=None):
    """Blocks left in ``epoch``, discounting the age of the clock reading."""
    now = time.time() if now is None else now
    age = max(0.0, now - float(clock['updated_at_unix']))
    return ((epoch + 1) * int(clock['epoch_length']) - int(clock['block'])
            - age / BLOCK_SECONDS)


def admissible(clock, epoch, *, reserve=RESERVE_BLOCKS, now=None):
    return (clock['epoch'] == epoch and int(clock['epoch_length']) > 0
            and blocks_remaining(clock, epoch, now=now) > reserve)


def publication_cutoff(clock, epoch, *, reserve=RESERVE_BLOCKS, now=None):
    """Unix time of the publication cutoff for ``epoch`` (the end of the usable window)."""
    now = time.time() if now is None else now
    return now + (blocks_remaining(clock, epoch, now=now) - reserve) * BLOCK_SECONDS


def phase_cap(phase, stop_at, *, now=None, publish_reserve=PUBLISH_RESERVE_SECONDS):
    """Cap for ``phase`` as a share of the window that is still left."""
    now = time.time() if now is None else now
    usable = stop_at - now - publish_reserve
    cap = usable * PHASE_SHARES[phase]
    if cap < PHASE_FLOORS[phase]:
        raise TimeoutError(f'insufficient window for the {phase} phase')
    return cap


# ssh exits 255 for connection-level failures (closed gateway, banner timeout, refused). Every
# operation is idempotent ('start' is keyed by the job identity, 'poll' only reads), so a few
# bounded retries are safe and a one-minute gateway blip must not end an epoch.
TRANSPORT_ATTEMPTS = 5
TRANSPORT_BACKOFF_SECONDS = 3
POLL_FAILURE_LIMIT = 8


def remote(config, host, request, *, sleep=time.sleep):
    command = ['ssh', '-T', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes',
               '-oConnectTimeout=8', '-p', str(config['hosts'][host]['port']),
               config['hosts'][host]['address'], 'python3', '-c', shlex.quote(TRANSPORT)]
    # Retries share one epoch deadline, including every attempt and backoff.
    deadline = request.get('transport_deadline_unix')
    end = (time.monotonic() + max(0.0, float(deadline) - time.time())
           if deadline is not None else math.inf)

    def remaining():
        seconds = end - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('remote transport epoch deadline')
        return seconds

    def backoff(attempt):
        sleep(min(TRANSPORT_BACKOFF_SECONDS * attempt, remaining()))

    for attempt in range(1, TRANSPORT_ATTEMPTS + 1):
        try:
            result = subprocess.run(command, input=json.dumps(request).encode(),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=min(config.get('transport_timeout_seconds', 120), remaining()))
        except subprocess.TimeoutExpired as cause:
            if attempt == TRANSPORT_ATTEMPTS:
                error = RuntimeError('remote transport timed out')
                error.diagnostics = {'attempts': attempt}
                raise error from cause
            backoff(attempt)
            continue
        if result.returncode == 255 and attempt < TRANSPORT_ATTEMPTS:
            backoff(attempt)
            continue
        break
    if result.returncode:
        error = RuntimeError('remote transport failed')
        error.diagnostics = {'returncode': result.returncode,
                             'stderr_tail': sanitize(result.stderr)}
        raise error
    try:
        # A login banner or rc-file echo may precede the reply; the reply is the last line.
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return json.loads(lines[-1])
    except (ValueError, IndexError) as cause:
        error = RuntimeError('remote transport returned no result')
        error.diagnostics = {'stdout_tail': sanitize(result.stdout),
                             'stderr_tail': sanitize(result.stderr)}
        raise error from cause


def phase_request(config, challenge, phase, files, stop_at, clock, *, cap=None):
    host = 'slave' if phase == 'sequence' else 'master'
    runtime = config['hosts'][host]
    release = runtime['release']
    directory = f"{config['remote_root']}/{challenge['epoch']}/{phase}"
    python = runtime['annotation_python'] if phase == 'sequence' else runtime['native_python']
    environment = {'PYTHONPATH': release + ':' + runtime['python_path'],
                   'PATH': runtime['path'], 'PYTHONDONTWRITEBYTECODE': '1',
                   'HOME': '/root', 'USER': 'root', 'LANG': 'C.UTF-8',
                   'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    if runtime.get('ld_library_path'):
        environment['LD_LIBRARY_PATH'] = runtime['ld_library_path']
    job = {'challenge': challenge, 'phase': phase, 'cpu_slots': 4 if phase == 'sequence' else 8}
    # New worker knobs (proposals, workers, repeats, margin, allow_marginal, ...) pass
    # through unchanged; the controller never second-guesses them.
    job.update(config['worker'])
    if phase == 'sequence':
        job['references'] = runtime['references']
    if 'archive.csv' in files:
        # The worker runs novelty only when the archive snapshot it can verify is present,
        # so every phase that checks novelty is handed the archive and its binding.
        job.update(archive_sha256=hashlib.sha256(files['archive.csv']).hexdigest(),
                   archive_fetched_at=clock['updated_at_unix'])
    files = dict(files)
    if phase == 'score':
        python = runtime['scoring_python']
        environment['PYTHONPATH'] = release
        environment['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in config['gpu'])
        job.update(gpus=list(config['gpu']), release=release, scorer=runtime['scorer'],
                   scorer_settings=runtime['scorer_settings'], source_sha=config['source_sha'],
                   gpu_lock_directory=config['remote_root'] + '/gpu-locks',
                   output_directory=directory + '/score')
        job['cpu_slots'] = max(2, len(config['gpu']))
    cap = phase_cap(phase, stop_at) if cap is None else cap
    if phase == 'native' and 'phase_budget_seconds' not in config['worker']:
        # Without this the worker measures until the supervisor kills it, and a killed
        # native phase returns no candidates at all. Leave room for the novelty pass.
        reserve = float(config.get('novelty_reserve_seconds', NOVELTY_RESERVE_SECONDS))
        job['phase_budget_seconds'] = max(1.0, cap - reserve)
    if phase == 'score':
        # The worker bounds each GPU lease inside the supervisor's wall cap.
        job['wall_seconds'] = max(30.0, cap - 15.0)
        job['deadline_unix'] = stop_at
    files['job.json'] = json.dumps(job).encode()
    command = [python, '-B', '-m', 'nanobody_opt.pr100_live_worker', '--job', directory + '/job.json']
    spec = {'command': command, 'cwd': release, 'environment': environment,
            'wall_seconds': cap, 'deadline_unix': stop_at,
            'receipt_path': directory + '/supervisor-receipt.json', 'cpu_slots': job['cpu_slots'],
            'clock_path': directory + '/clock.json', 'epoch': challenge['epoch'],
            'epoch_length': challenge['epoch_length'], 'reserve_blocks': RESERVE_BLOCKS}
    return host, {'operation': 'start', 'directory': directory, 'python': python,
                  'transport_deadline_unix': stop_at + 15,
                  'source_sha': config['source_sha'],
                  'supervisor': release + '/scripts/pr100_live_supervisor.py', 'spec': spec,
                  'clock': clock, 'files': {k: base64.b64encode(v).decode() for k, v in files.items()}}


def run_phase(config, challenge, phase, files, stop_at, chain, local,
              *, transport=remote, sleep=time.sleep, diagnostics=None):
    def heartbeat(clock):
        try:
            atomic(Path(config['state_root']) / 'heartbeat.json',
                   {'status': 'running', 'phase': phase, 'work_epoch': challenge['epoch'], **clock})
        except OSError:
            if diagnostics is not None:
                diagnostics['heartbeat_write_errors'] = diagnostics.get('heartbeat_write_errors', 0) + 1

    clock = clock_state(chain)
    if not admissible(clock, challenge['epoch']) or time.time() >= stop_at:
        raise TimeoutError('epoch cutoff before launch')
    host, request = phase_request(config, challenge, phase, files, stop_at, clock)
    heartbeat(clock)
    result = transport(config, host, request)
    failures = 0
    while result['running']:
        sleep(5)
        if time.time() > stop_at + 15:
            raise TimeoutError('remote receipt deadline')
        try:
            clock = clock_state(chain)
            heartbeat(clock)
            request = {'operation': 'poll', 'directory': request['directory'], 'clock': clock,
                       'transport_deadline_unix': stop_at + 15}
            result = transport(config, host, request)
            failures = 0
        except TimeoutError:
            raise  # An epoch deadline is final, unlike a missed poll.
        except (RuntimeError, OSError) as error:
            # The remote job keeps running under its own supervisor; a failed poll is only a
            # missed look at it. Give up on the phase only after sustained failure.
            failures += 1
            if diagnostics is not None:
                diagnostics.setdefault('poll_failures', []).append(type(error).__name__)
            if failures > POLL_FAILURE_LIMIT:
                raise
            result = {'running': True}
    receipt = result['receipt']
    atomic(local / (phase + '-supervisor.json'), receipt)
    if diagnostics is not None:
        diagnostics[phase] = {'reason': receipt.get('reason'), 'exit_code': receipt.get('exit_code'),
                              'killed': receipt.get('killed'),
                              'elapsed_seconds': receipt.get('elapsed_seconds'),
                              'stderr_tail': sanitize(receipt.get('stderr_tail') or ''),
                              'tails': {k: sanitize(v) for k, v in (result.get('tails') or {}).items()}}
    decoded = {k: base64.b64decode(v, validate=True) for k, v in result.get('files', {}).items()}
    for name, raw in decoded.items():
        output = local / phase / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
    if receipt.get('reason') not in (None, 'completed') and 'result.json' not in decoded:
        error = RuntimeError(f'{phase} phase produced no result')
        error.diagnostics = {'phase': phase, 'reason': receipt.get('reason'),
                             'exit_code': receipt.get('exit_code'),
                             'stderr_tail': sanitize(receipt.get('stderr_tail') or '')}
        raise error
    # Only already completed per-candidate records can survive a later timeout.
    return decoded


def fetch_archive(target):
    if target != rules.TARGET:
        raise ValueError('unexpected target')
    url = f'https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main/{target}_nanobodies.csv'
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024 or not data:
        raise ValueError('invalid archive size')
    return data


def merge_scores(files, target):
    report = json.loads(files.get('score/report.json', b'{}'))
    merged = {'target': target, 'per_nanobody_components': {'0': {}}, 'structures': {}, 'source_batches': []}
    if 'score/scorer_settings.json' not in files or not report.get('registered_scorer_sha256'):
        raise ValueError('score phase returned no scored group')
    for group in report.get('groups', []):
        if group.get('valid') is not True:
            continue
        raw = files.get('score/' + str(group.get('file')))
        if raw is None:
            continue  # partial result: the group file never arrived
        value = json.loads(raw)
        if value.get('target') != target:
            raise ValueError('scorer target changed')
        rows = value['per_nanobody_components']['0']
        if set(rows) & set(merged['per_nanobody_components']['0']):
            raise ValueError('duplicate scorer group')
        merged['per_nanobody_components']['0'].update(rows)
        merged['structures'].update(value['structures'])
        merged['source_batches'].append({'sha256': hashlib.sha256(raw).hexdigest(), 'count': len(rows)})
    merged['n'] = len(merged['per_nanobody_components']['0'])
    merged['wall_seconds'] = report.get('elapsed_seconds')
    return merged, json.loads(files['score/scorer_settings.json']), report['registered_scorer_sha256']


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def candidate_tiers(candidates):
    """Map sequence -> the worker's tier and mean PSH, for the quality comparison."""
    return {row['sequence']: {'tier': row.get('tier'),
                              'psh_mean': float(row['psh_mean']) if _finite(row.get('psh_mean')) else None,
                              'repeats': row.get('repeats'), 'margin_safe': row.get('margin_safe')}
            for row in candidates}


def order_candidates(candidates, *, allow_marginal=True):
    """Worker order (tier A by ascending mean PSH, then tier B) with a local fallback."""
    try:
        from nanobody_opt.pr100_live_selection import order_candidates as ordered
    except ImportError:
        def key(row):
            mean = row.get('psh_mean')
            return (TIER_RANK.get(row.get('tier'), len(TIER_RANK)),
                    mean if _finite(mean) else math.inf,
                    str(row.get('sequence_hash') or ''))
        rows = sorted((r for r in candidates if r.get('tier') in TIER_RANK), key=key)
        return rows if allow_marginal else [r for r in rows if r.get('tier') == 'A']
    return ordered(candidates, allow_marginal=allow_marginal)


def fleet_quality(fleet, tiers):
    """Quality of a fleet: tier A count, size, then each member's tier and margin."""
    members = []
    for sequence in fleet.get('sequence_pool', []):
        info = tiers.get(sequence) or {}
        members.append({'sequence_hash': rules.sequence_hash(sequence),
                        'tier_rank': TIER_RANK.get(info.get('tier'), len(TIER_RANK)),
                        'psh_mean': info.get('psh_mean')})
    members.sort(key=lambda m: (m['tier_rank'], math.inf if m['psh_mean'] is None else m['psh_mean']))
    return {'tier_a': sum(1 for m in members if m['tier_rank'] == TIER_RANK['A']),
            'count': len(members), 'members': members}


def quality_key(quality):
    """Higher is strictly better, so a lexicographic comparison is the monotone rule."""
    def member(entry):
        mean = entry['psh_mean']
        return (-entry['tier_rank'], -(math.inf if mean is None else mean))
    return (quality['tier_a'], quality['count'],
            tuple(member(entry) for entry in quality['members']))


def monotone_decision(previous_quality, quality):
    """'replace', 'idempotent' or 'rejected' under the monotone-quality rule."""
    if previous_quality is None:
        return 'replace'
    new, old = quality_key(quality), quality_key(previous_quality)
    if new > old:
        return 'replace'
    return 'idempotent' if new == old else 'rejected'


def quality_path(local):
    return Path(local) / 'published-quality.json'


def published_quality(local, epoch):
    try:
        record = json.loads(quality_path(local).read_text())
    except (OSError, ValueError):
        return None
    return record.get('quality') if record.get('epoch') == epoch else None


def publication_sequence(ordered, *, allow_marginal=True):
    """Margin-safe candidates first, then the monotone upgrade that adds tier B."""
    safe = [row for row in ordered if row.get('tier') == 'A']
    attempts = []
    if safe:
        attempts.append(safe)
    if allow_marginal and len(ordered) > len(safe):
        attempts.append(list(ordered))
    return attempts or [list(ordered)]


def run_epoch(config, chain, clock, *, dry_run=False):
    epoch, length = clock['epoch'], clock['epoch_length']
    local = Path(config['state_root']) / str(epoch)
    local.mkdir(parents=True, exist_ok=True)
    started = time.time()
    reserve = config.get('reserve_blocks', RESERVE_BLOCKS)
    stop_at = publication_cutoff(clock, epoch, reserve=reserve, now=started)
    phase = 'challenge'
    diagnostics = {}
    receipt = {'epoch': epoch, 'started_at_unix': started, 'source_sha': config['source_sha'],
               'mac_runtime_dependency': False, 'signing_performed': False, 'dry_run': bool(dry_run),
               'publication_cutoff_unix': stop_at, 'reserve_blocks': reserve,
               'usable_window_seconds': max(0.0, stop_at - started), 'publications': []}
    try:
        block_hash = chain.determine_block_hash(epoch * length)
        challenge = rules.challenge(rules.pinned_config(epoch), epoch, block_hash, length)
        receipt['challenge_cell'] = {k: challenge[k] for k in ('allowed_v', 'allowed_cdr3_length')
                                     if k in challenge}
        phase = 'archive'
        archive = fetch_archive(challenge['target'])
        phase = 'sequence'
        files = run_phase(config, challenge, phase, {'archive.csv': archive}, stop_at, chain, local,
                          diagnostics=diagnostics)
        validity = json.loads(files['result.json'])
        if not validity['candidates']:
            raise ValueError('no novel annotated candidates')
        phase = 'native'
        files = run_phase(config, challenge, phase,
                          {'input.json': json.dumps(validity).encode(), 'archive.csv': archive},
                          stop_at, chain, local, diagnostics=diagnostics)
        validity = json.loads(files['result.json'])
        if validity.get('status') != 'complete':
            raise ValueError('native phase did not complete')
        # No candidate cap: whatever the worker returns is scored.
        allow_marginal = bool(config['worker'].get('allow_marginal', True))
        ordered = order_candidates(validity['candidates'], allow_marginal=allow_marginal)
        if not ordered:
            raise ValueError('no native passing candidate')
        # The native stage also runs novelty (only on its passers). An unchecked novelty
        # result is never publishable.
        if validity.get('novelty_checked') is not True:
            raise ValueError('native stage reported no novelty check')
        if len({r['sequence'] for r in ordered}) != len(ordered):
            raise ValueError('duplicate sequence in native result')
        validity['candidates'] = ordered
        phase = 'score'
        files = run_phase(config, challenge, phase,
                          {'input.json': json.dumps({'sequences': [r['sequence'] for r in ordered]}).encode()},
                          stop_at, chain, local, diagnostics=diagnostics)
        scores, settings, scorer_hash = merge_scores(files, challenge['target'])
        settings_hash = hashlib.sha256(files['score/scorer_settings.json']).hexdigest()
        phase = 'publish'
        tiers = candidate_tiers(ordered)
        receipt['scored_candidates'] = scores['n']
        for subset in publication_sequence(ordered, allow_marginal=allow_marginal):
            # Only scored candidates can enter a fleet; an unscored margin-safe set is skipped.
            scored = [r for r in subset if r['sequence'] in scores['per_nanobody_components']['0']]
            if not scored:
                continue
            attempt = dict(validity, candidates=scored)
            try:
                outcome = publish(config, chain, challenge, attempt, scores, settings, scorer_hash,
                                  settings_hash, local, tiers=tiers, dry_run=dry_run)
            except Exception as error:
                if not any(p.get('status') in ('published', 'dry_run') for p in receipt['publications']):
                    raise
                # An earlier attempt already published; a refused upgrade must not erase it.
                receipt['publications'].append({'status': 'upgrade_refused',
                                                'error_type': type(error).__name__,
                                                'error': sanitize(str(error))})
                continue
            receipt['publications'].append(outcome)
            receipt.update({k: v for k, v in outcome.items() if k != 'decision'})
        if not receipt['publications']:
            raise ValueError('no scored candidate to publish')
    except Exception as error:
        receipt.update(status='failed', stage=phase, error_type=type(error).__name__,
                       error=sanitize(str(error)))
        extra = getattr(error, 'diagnostics', None)
        if isinstance(extra, dict):
            receipt['failure_diagnostics'] = {
                k: sanitize(v) if isinstance(v, (str, bytes)) else v for k, v in extra.items()}
    receipt['phase_diagnostics'] = diagnostics
    receipt['wall_seconds'] = time.time() - started
    atomic(local / 'receipt.json', receipt)
    # The scientific outcome is already durable. Optional diagnostics get their
    # own short deadline and cannot delay scoring, publishing or the next epoch.
    if 'native' in diagnostics:
        start_native_journal_collection(config, epoch, local, stop_at)
    return receipt


def start_native_journal_collection(config, epoch, local, stop_at):
    """At most one optional daemon; even stalled disk I/O cannot hold the controller."""
    global _JOURNAL_COLLECTOR
    if _JOURNAL_COLLECTOR is not None and _JOURNAL_COLLECTOR.is_alive():
        return False
    _JOURNAL_COLLECTOR = threading.Thread(target=collect_native_journal,
        args=(config, epoch, local, stop_at), daemon=True, name='pr100-journal-collector')
    _JOURNAL_COLLECTOR.start()
    return True


def collect_native_journal(config, epoch, local, stop_at, *, transport=remote):
    deadline = min(stop_at, time.time() + 3.0)
    if deadline <= time.time():
        return {'status': 'deadline_elapsed'}
    outcome = {'status': 'unavailable'}
    try:
        reply = transport({**config, 'transport_timeout_seconds': 3.0}, 'master',
            {'operation': 'journal', 'directory': f"{config['remote_root']}/{epoch}/native",
             'transport_deadline_unix': deadline})
        status = reply.get('status')
        if status in ('unavailable', 'oversized', 'not_regular'):
            outcome = {'status': status}
        elif status == 'collected':
            encoded = reply.get('data', '')
            if not isinstance(encoded, str) or len(encoded) > 6 * 1024 * 1024:
                raise ValueError('journal size')
            raw = base64.b64decode(encoded, validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) > 4 * 1024 * 1024 or digest != reply.get('sha256'):
                raise ValueError('journal integrity')
            path = local / 'native' / 'native-observations.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            outcome = {'status': 'collected', 'bytes': len(raw), 'sha256': digest}
    except Exception:
        outcome = {'status': 'collection_failed'}
    try:
        atomic(local / 'journal-receipt.json', outcome)
    except OSError:
        pass
    return outcome


def publish(config, chain, challenge, validity, scores, settings, scorer_hash, settings_hash, local,
            *, tiers=None, dry_run=False):
    from nanobody_opt.pr100_live_trial_export import build
    epoch = challenge['epoch']
    reserve = config.get('reserve_blocks', RESERVE_BLOCKS)
    clock = clock_state(chain)
    if not admissible(clock, epoch, reserve=reserve):
        raise TimeoutError('publication cutoff')
    if chain.determine_block_hash(epoch * challenge['epoch_length']) != challenge['epoch_start_block_hash']:
        raise ValueError('epoch start changed')
    fresh_archive = fetch_archive(challenge['target'])
    if hashlib.sha256(fresh_archive).hexdigest() != validity['archive_sha256']:
        raise ValueError('archive changed after exact novelty check')
    fleet = build(validity, scores, settings, scorer_sha256=scorer_hash,
                  settings_file_sha256=settings_hash, now=time.time())
    verified = rules.verified_fleet(fleet, epoch, challenge['epoch_start_block_hash'],
                                    challenge['epoch_length'])
    if verified['sequence_pool'] != fleet['sequence_pool'] or not fleet['sequence_pool']:
        raise ValueError('actual Pi validation rejected candidate')
    tiers = candidate_tiers(validity['candidates']) if tiers is None else tiers
    quality = fleet_quality(fleet, tiers)
    destination = Path(config['publication_path'])
    decision = 'replace'
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous.get('epoch') == epoch:
            earlier = published_quality(local, epoch)
            if earlier is None:
                # Nothing recorded the earlier fleet's quality (a legacy producer write, for
                # instance), so fall back to the never-discard subset rule.
                if not set(previous.get('sequence_pool', [])) <= set(fleet['sequence_pool']):
                    raise ValueError('cannot discard an existing valid partial fleet')
            else:
                decision = monotone_decision(earlier, quality)
                if decision == 'rejected':
                    raise ValueError('cannot degrade an existing valid partial fleet')
    atomic(local / 'prepared-fleet.json', fleet)
    outcome = {'count': len(fleet['sequence_pool']), 'decision': decision,
               'sequence_hashes': [rules.sequence_hash(s) for s in fleet['sequence_pool']],
               'tier_a_count': quality['tier_a'], 'actual_pi_validation': True,
               'publication_stage': 'live_trial'}
    if dry_run:
        atomic(local / 'dry-run-fleet.json', fleet)
        outcome.update(status='dry_run', published=False)
        return outcome
    if not admissible(clock_state(chain), epoch, reserve=reserve):
        raise TimeoutError('publication cutoff after validation')
    atomic(destination, fleet)
    atomic(quality_path(local), {'epoch': epoch, 'quality': quality, 'updated_at_unix': time.time()})
    outcome.update(status='published', published=True,
                   published_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    return outcome


def validate_config(config):
    missing = [k for k in ('start_epoch', 'hosts', 'remote_root', 'state_root', 'publication_path',
                           'source_sha', 'worker', 'gpu') if k not in config]
    if missing:
        raise ValueError('config is missing required keys: ' + ', '.join(missing))
    if isinstance(config['start_epoch'], bool) or not isinstance(config['start_epoch'], int) \
            or config['start_epoch'] < ACTIVATION_EPOCH:
        raise ValueError(f'start_epoch must be an integer >= {ACTIVATION_EPOCH}')
    gpus = config['gpu']
    if (not isinstance(gpus, list) or not gpus
            or any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in gpus)):
        raise ValueError('gpu must be a non-empty list of non-negative gpu ids')
    if len(set(gpus)) != len(gpus):
        raise ValueError('gpu list contains a duplicate id')
    for key in ('state_root', 'publication_path'):
        if not Path(config[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute path')
    if not str(config['remote_root']).startswith('/'):
        raise ValueError('remote_root must be an absolute remote path')
    if not isinstance(config['source_sha'], str) or not re.fullmatch(r'[0-9a-f]{40}', config['source_sha']):
        raise ValueError('source_sha must be a 40-character git revision')
    if not isinstance(config['worker'], dict) or 'native_tools' not in config['worker']:
        raise ValueError('worker must be an object including native_tools')
    hosts = config['hosts']
    if not isinstance(hosts, dict) or not {'master', 'slave'} <= set(hosts):
        raise ValueError('hosts must define both master and slave')
    for name in ('master', 'slave'):
        runtime = hosts[name]
        if not isinstance(runtime, dict):
            raise ValueError(f'hosts.{name} must be an object')
        absent = [k for k in HOST_KEYS if k not in runtime]
        if absent:
            raise ValueError(f'hosts.{name} is missing: ' + ', '.join(absent))
        port = runtime['port']
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f'hosts.{name}.port must be a tcp port')
        if not str(runtime['release']).startswith('/'):
            raise ValueError(f'hosts.{name}.release must be an absolute path')
    reserve = config.get('reserve_blocks', RESERVE_BLOCKS)
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve <= MINIMUM_RESERVE_BLOCKS:
        raise ValueError(f'reserve_blocks must be an integer greater than {MINIMUM_RESERVE_BLOCKS}')
    verify_build_stamp(config)
    return config


def verify_build_stamp(config, root=None):
    """The controller may only run from a release whose stamp matches ``source_sha``."""
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    stamp = json.loads((root / 'build_stamp.json').read_text())
    if stamp['git_sha'] != config['source_sha']:
        raise ValueError('controller source revision mismatch')
    for name, digest in stamp['files'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('controller source drift')
    return stamp


def main(argv=None):
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='run every phase but never write the Pi publication destination')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate_config(config)
    root = Path(config['state_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import bittensor as bt
    chain = bt.Subtensor(network='finney')
    while True:
        try:
            clock = clock_state(chain)
            epoch = clock['epoch']
            if epoch >= config['start_epoch'] and not (root / str(epoch) / 'receipt.json').exists():
                if admissible(clock, epoch, reserve=START_MINIMUM_BLOCKS):
                    run_epoch(config, chain, clock, dry_run=args.dry_run)
                    clock = clock_state(chain)
                else:
                    directory = root / str(epoch)
                    directory.mkdir(exist_ok=True)
                    atomic(directory / 'receipt.json',
                           {'status': 'missed_start_window', 'epoch': epoch,
                            'blocks_remaining': blocks_remaining(clock, epoch)})
            atomic(root / 'heartbeat.json', {'status': 'watching', **clock})
        except Exception as error:
            atomic(root / 'heartbeat.json', {'status': 'failed', 'error_type': type(error).__name__,
                                             'error': sanitize(str(error)),
                                             'updated_at_unix': time.time()})
        if args.once:
            return 0
        time.sleep(10)


if __name__ == '__main__':
    raise SystemExit(main())
