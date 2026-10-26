"""Pi-owned, bounded baseline: current rules, fresh measurements, partial fleets."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from nanobody_opt import pr100_filter, rules

RESERVE_BLOCKS = 40
PHASE_CAPS = {'sequence': 420, 'native': 900, 'score': 600}
TRANSPORT = r'''
import base64,hashlib,json,os,pathlib,subprocess,sys,time
r=json.load(sys.stdin);d=pathlib.Path(r['directory']);d.mkdir(parents=True,exist_ok=True)
def atom(p,v):
 t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(v,allow_nan=False));os.replace(t,p)
atom(d/'clock.json',r['clock'])
if r['operation']=='start':
 release=pathlib.Path(r['spec']['cwd']);stamp=json.loads((release/'build_stamp.json').read_text())
 if stamp['git_sha']!=r['source_sha']:raise ValueError('source revision')
 for name,expected in stamp['files'].items():
  if hashlib.sha256((release/name).read_bytes()).hexdigest()!=expected:raise ValueError('source identity')
 identity=hashlib.sha256(json.dumps(r['spec'],sort_keys=True).encode()).hexdigest()
 if (d/'launch.json').exists():
  old=json.loads((d/'launch.json').read_text())
  if old['identity']!=identity:raise ValueError('different existing job')
 else:
  for name,raw in r['files'].items():
   if name not in ('input.json','archive.csv','job.json','gpu-job.json'):raise ValueError('file name')
   (d/name).write_bytes(base64.b64decode(raw,validate=True))
  atom(d/'spec.json',r['spec'])
  p=subprocess.Popen([r['python'],'-B',r['supervisor'],'--spec',str(d/'spec.json')],
   cwd=r['spec']['cwd'],env={**os.environ,**r['spec']['environment']},
   stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
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
   if group.get('valid') is True:
    name='score/length'+str(int(group['length']))+'/exact_all.json'
    out['files'][name]=base64.b64encode((d/name).read_bytes()).decode()
print(json.dumps(out))
'''

def atomic(path, value):
    pr100_filter.atomic(path, value)

def clock_state(chain):
    block = int(chain.get_current_block())
    length = int(chain.tempo(68)) + 1
    return {'block': block, 'epoch': block // length, 'epoch_length': length,
            'updated_at_unix': time.time()}

def admissible(clock, epoch, *, reserve=RESERVE_BLOCKS):
    return (clock['epoch'] == epoch and clock['epoch_length'] > 0 and
            (epoch + 1) * clock['epoch_length'] - clock['block'] > reserve)

def remote(config, host, request):
    command = ['ssh', '-T', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes',
               '-oConnectTimeout=8', '-p', str(config['hosts'][host]['port']),
               config['hosts'][host]['address'], 'python3', '-c',
               __import__('shlex').quote(TRANSPORT)]
    result = subprocess.run(command, input=json.dumps(request).encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        raise RuntimeError('remote transport failed')
    return json.loads(result.stdout)


def phase_request(config, challenge, phase, files, stop_at, clock):
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
    job.update(config['worker'])
    if phase == 'sequence':
        job.update(archive_sha256=hashlib.sha256(files['archive.csv']).hexdigest(),
                   archive_fetched_at=clock['updated_at_unix'], references=runtime['references'])
    files = dict(files)
    files['job.json'] = json.dumps(job).encode()
    command = [python, '-B', '-m', 'nanobody_opt.pr100_live_worker', '--job', directory + '/job.json']
    if phase == 'score':
        python = runtime['scoring_python']
        environment['PYTHONPATH'] = release
        environment['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])
        probe = [python, '-B', release + '/scripts/pr100_gpu_probe.py', '--release', release,
                 '--expected-sha', config['source_sha'], '--input', directory + '/input.json',
                 '--output', directory + '/score', '--settings', runtime['scorer_settings'],
                 '--gpu', str(config['gpu']), '--passes', '1', '--mode', 'fresh', '--timeout', '580']
        gpu = {'gpu': config['gpu'], 'ttl_seconds': 590, 'minimum_free_mib': 8192,
               'lock_directory': config['remote_root'] + '/gpu-locks',
               'receipt': directory + '/gpu-receipt.json', 'scorer': runtime['scorer'],
               'command': probe, 'cwd': release,
               'env': {k: v for k, v in environment.items() if k != 'CUDA_VISIBLE_DEVICES'}}
        files['gpu-job.json'] = json.dumps(gpu).encode()
        command = [python, '-B', '-m', 'nanobody_opt.pr100_live_gpu', '--job', directory + '/gpu-job.json']
    spec = {'command': command, 'cwd': release, 'environment': environment,
            'wall_seconds': PHASE_CAPS[phase], 'deadline_unix': stop_at,
            'receipt_path': directory + '/supervisor-receipt.json', 'cpu_slots': job['cpu_slots'],
            'clock_path': directory + '/clock.json', 'epoch': challenge['epoch'],
            'epoch_length': challenge['epoch_length'], 'reserve_blocks': RESERVE_BLOCKS}
    return host, {'operation': 'start', 'directory': directory, 'python': python, 'source_sha': config['source_sha'],
                  'supervisor': release + '/scripts/pr100_live_supervisor.py', 'spec': spec,
                  'clock': clock, 'files': {k: base64.b64encode(v).decode() for k, v in files.items()}}


def run_phase(config, challenge, phase, files, stop_at, chain, local,
              *, transport=remote, sleep=time.sleep):
    clock = clock_state(chain)
    if not admissible(clock, challenge['epoch']) or time.time() >= stop_at:
        raise TimeoutError('epoch cutoff before launch')
    host, request = phase_request(config, challenge, phase, files, stop_at, clock)
    result = transport(config, host, request)
    while result['running']:
        sleep(5)
        request = {'operation': 'poll', 'directory': request['directory'], 'clock': clock_state(chain)}
        result = transport(config, host, request)
        if time.time() > stop_at + 15:
            raise TimeoutError('remote receipt deadline')
    atomic(local / (phase + '-supervisor.json'), result['receipt'])
    decoded = {k: base64.b64decode(v, validate=True) for k, v in result.get('files', {}).items()}
    for name, raw in decoded.items():
        output = local / phase / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
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
    for group in report.get('groups', []):
        if group.get('valid') is not True:
            continue
        raw = files[f"score/length{group['length']}/exact_all.json"]
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


def run_epoch(config, chain, clock):
    epoch, length = clock['epoch'], clock['epoch_length']
    local = Path(config['state_root']) / str(epoch)
    local.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stop_at = started + 1800
    phase = 'challenge'
    receipt = {'epoch': epoch, 'started_at_unix': started, 'source_sha': config['source_sha'],
               'mac_runtime_dependency': False, 'signing_performed': False}
    try:
        block_hash = chain.determine_block_hash(epoch * length)
        challenge = rules.challenge(rules.pinned_config(epoch), epoch, block_hash, length)
        phase = 'archive'
        archive = fetch_archive(challenge['target'])
        phase = 'sequence'
        files = run_phase(config, challenge, phase, {'archive.csv': archive}, stop_at, chain, local)
        validity = json.loads(files['result.json'])
        if not validity['candidates']:
            raise ValueError('no novel annotated candidates')
        phase = 'native'
        files = run_phase(config, challenge, phase, {'input.json': json.dumps(validity).encode()}, stop_at, chain, local)
        validity = json.loads(files['result.json'])
        validity['candidates'] = validity['candidates'][:8]
        if not validity['candidates']:
            raise ValueError('no native passing candidate')
        phase = 'score'
        files = run_phase(config, challenge, phase,
                          {'input.json': json.dumps({'sequences': [r['sequence'] for r in validity['candidates']]}).encode()},
                          stop_at, chain, local)
        scores, settings, scorer_hash = merge_scores(files, challenge['target'])
        phase = 'publish'
        receipt.update(publish(config, chain, challenge, validity, scores, settings, scorer_hash,
                               hashlib.sha256(files['score/scorer_settings.json']).hexdigest(), local))
    except Exception as error:
        receipt.update(status='failed', stage=phase, error_type=type(error).__name__)
    receipt['wall_seconds'] = time.time() - started
    atomic(local / 'receipt.json', receipt)
    return receipt


def publish(config, chain, challenge, validity, scores, settings, scorer_hash, settings_hash, local):
    from nanobody_opt.pr100_live_trial_export import build
    epoch = challenge['epoch']
    clock = clock_state(chain)
    if not admissible(clock, epoch):
        raise TimeoutError('publication cutoff')
    if chain.determine_block_hash(epoch * challenge['epoch_length']) != challenge['epoch_start_block_hash']:
        raise ValueError('epoch start changed')
    fresh_archive = fetch_archive(challenge['target'])
    if hashlib.sha256(fresh_archive).hexdigest() != validity['archive_sha256']:
        raise ValueError('archive changed after exact novelty check')
    fleet = build(validity, scores, settings, scorer_sha256=scorer_hash,
                  settings_file_sha256=settings_hash, now=time.time())
    verified = rules.verified_fleet(fleet, epoch, challenge['epoch_start_block_hash'], challenge['epoch_length'])
    if verified['sequence_pool'] != fleet['sequence_pool'] or not fleet['sequence_pool']:
        raise ValueError('actual Pi validation rejected candidate')
    destination = Path(config['publication_path'])
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous.get('epoch') == epoch and not set(previous.get('sequence_pool', [])) <= set(fleet['sequence_pool']):
            raise ValueError('cannot discard an existing valid partial fleet')
    atomic(local / 'prepared-fleet.json', fleet)
    if not admissible(clock_state(chain), epoch):
        raise TimeoutError('publication cutoff after validation')
    atomic(destination, fleet)
    return {'status': 'published', 'count': len(fleet['sequence_pool']),
            'sequence_hashes': [rules.sequence_hash(s) for s in fleet['sequence_pool']],
            'published_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'actual_pi_validation': True, 'publication_stage': 'live_trial'}


def validate_config(config):
    if config['start_epoch'] < 25957 or config['gpu'] != 1:
        raise ValueError('unregistered activation or GPU')
    if not Path(config['state_root']).is_absolute() or not Path(config['publication_path']).is_absolute():
        raise ValueError('absolute Pi paths required')
    stamp = json.loads((Path(__file__).resolve().parents[1] / 'build_stamp.json').read_text())
    if stamp['git_sha'] != config['source_sha']:
        raise ValueError('controller source revision mismatch')
    root = Path(__file__).resolve().parents[1]
    for name, digest in stamp['files'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('controller source drift')


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
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
                if admissible(clock, epoch, reserve=180):
                    run_epoch(config, chain, clock)
                else:
                    directory = root / str(epoch)
                    directory.mkdir(exist_ok=True)
                    atomic(directory / 'receipt.json', {'status': 'missed_start_window', 'epoch': epoch})
            atomic(root / 'heartbeat.json', {'status': 'watching', **clock})
        except Exception as error:
            atomic(root / 'heartbeat.json', {'status': 'failed', 'error_type': type(error).__name__,
                                             'updated_at_unix': time.time()})
        if args.once:
            return 0
        time.sleep(10)


if __name__ == '__main__':
    raise SystemExit(main())
