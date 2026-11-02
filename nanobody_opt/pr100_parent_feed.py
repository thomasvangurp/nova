"""Pi-owned incremental parent refresh and bounded annotation-only qualification.

All remote output is a fixed public-data envelope; private child logs never
leave the host. No credentials are read, copied, or put into subprocess argv.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from nanobody_opt import pr100_parent_bank as bank, rules


def fetch_epoch(epoch, target, before_epoch):
    url = f'https://compound-api.metanova-labs.ai/api/competitions/leaderboard/{epoch}/nanobodies?limit=10000'
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'nova-parent-feed/1'})
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
        bank.require(len(raw) <= 8 * 1024 * 1024, 'public_response_size')
        value = bank.public_epoch(json.loads(raw), target, epoch, before_epoch)
        return {'epoch': epoch, 'status': 'ok', 'board': value}
    except urllib.error.HTTPError as error:
        return {'epoch': epoch, 'status': 'unavailable', 'http_status': error.code}
    except Exception as error:
        return {'epoch': epoch, 'status': 'unavailable', 'error_type': type(error).__name__}


def public_refresh(directory, ledger, before_epoch, target, *, limit=8, now=None,
                   fetch=fetch_epoch):
    now = time.time() if now is None else now
    epochs = bank.due_epochs(ledger, before_epoch, now, limit)
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(epochs)))) as pool:
        fetched = list(pool.map(lambda e: fetch(e, target, before_epoch), epochs))
    for response in fetched:
        epoch = response['epoch']; previous = ledger.get(str(epoch), {})
        entry = {'last_checked_unix': now, 'status': response['status']}
        if response['status'] == 'ok':
            content = bank.raw(response['board']); digest = bank.sha(content)
            bank.immutable(directory / 'public' / (digest + '.json'), content)
            entry.update(sha256=digest, row_count=len(response['board']['rows']), failures=0,
                         retry_after_unix=now + (300 if epoch >= before_epoch - 3 else 3600))
        else:
            failures = previous.get('failures', 0) + 1
            entry.update({k: previous[k] for k in ('sha256', 'row_count') if k in previous})
            entry.update(failures=failures,
                         retry_after_unix=now + min(3600, 300 * 2 ** min(failures - 1, 4)))
            entry.update({k: response[k] for k in ('http_status', 'error_type') if k in response})
        ledger[str(epoch)] = entry
        bank.atomic(directory / 'public-ledger.json', ledger)
    rows = []
    sources = {}
    for text, item in sorted(ledger.items(), key=lambda v: int(v[0])):
        epoch = int(text)
        if epoch >= before_epoch or not item.get('sha256'):
            continue
        content = bank.read(directory / 'public' / (item['sha256'] + '.json'))
        bank.require(bank.sha(content) == item['sha256'], 'cached_public_board_changed')
        board = json.loads(content)
        bank.require(board['target'] == target and board['epoch'] == epoch, 'cached_public_board_identity')
        rows.extend(board['rows']); sources[text] = item['sha256']
    return {'schema': 'pr100-public-parent-bundle-v1', 'target': target,
            'before_epoch': before_epoch, 'sources': sources, 'rows': rows}, len(epochs)


def verify_source(source_sha):
    stamp = json.loads(bank.read(ROOT / 'build_stamp.json'))
    bank.require(stamp.get('git_sha') == source_sha and stamp.get('validation', {}).get('passed') is True,
                 'parent_source_not_certified')
    for name, expected in stamp['files'].items():
        bank.require(bank.sha((ROOT / name).read_bytes()) == expected, 'parent_source_drift')


def initialize_annotation(epoch):
    global REQUIREMENTS
    from nanobody_opt import pr100_filter
    pr100_filter.activate_filter(); pr100_filter.validate_annotation_runtime()
    REQUIREMENTS = rules.pinned_config(epoch)['nanobody_requirements']


def annotate_one(row):
    from nanobody_opt import pr100_filter, gates
    try:
        annotation = {**pr100_filter._annotate_uncached(row['sequence']),
                      'annotation_identity': pr100_filter.annotation_identity()}
        reason = rules.annotation_reason(annotation, REQUIREMENTS, sequence=row['sequence'])
        if not reason and gates.check_sequence(row['sequence'], REQUIREMENTS):
            reason = 'basic_sequence'
        return {'sequence_hash': row['sequence_hash'], 'annotation': annotation, 'reason': reason or 'passed'}
    except Exception as error:
        return {'sequence_hash': row['sequence_hash'], 'reason': 'annotation_error',
                'error_type': type(error).__name__}


def qualify_rows(job, original, public, annotate):
    """Injectable pure qualification boundary; any transient error aborts promotion."""
    from tools.de_novo_bank import allowed_v_genes, validate
    rows = bank.merge_rows(original['rows'], public['rows'], target=job['target'],
                           before_epoch=job['before_epoch'])
    old = {r['sequence_hash']: r for r in original['rows']}
    findings = list(annotate(rows))
    bank.require(len(findings) == len(rows) and {f['sequence_hash'] for f in findings}
                 == {r['sequence_hash'] for r in rows}, 'annotation_incomplete')
    by_hash = {f['sequence_hash']: f for f in findings}
    bank.require(all(f['reason'] != 'annotation_error' for f in findings), 'annotation_transient_failure')
    kept = []
    for row in rows:
        finding = by_hash[row['sequence_hash']]; annotation = finding.get('annotation', {})
        length = annotation.get('cdr3_length')
        cells = ([f'{gene}:{length}' for gene in allowed_v_genes(annotation)]
                 if type(length) is int and 5 <= length <= 23 else [])
        passed = finding['reason'] == 'passed' and cells
        if row['sequence_hash'] in old:
            bank.require(passed and sorted(cells) == sorted(old[row['sequence_hash']]['cells']),
                         'previous_parent_annotation_changed')
        elif not passed:
            continue
        row['cells'] = cells; row['pr100_annotation'] = annotation; kept.append(row)
    validate(kept)
    bank.require(kept and set(old) <= {r['sequence_hash'] for r in kept}, 'old_parent_lost')
    # Join the durable native developability profile (raw PSH) for any parent
    # already measured on the master box, so the selector reads the real
    # profile instead of the leaderboard developability_rank proxy. Opt-in:
    # absent means disabled. When present the store must work -- a missing or
    # empty store raises, so a broken join can never publish as success.
    # Read-only; never touches quality_evidence.
    developability_db = job.get('developability_db')
    if developability_db:
        from nanobody_opt import pr100_developability_store
        pr100_developability_store.attach(kept, developability_db, target=job['target'])
    identity = {k: job[k] for k in ('source_sha', 'target', 'before_epoch', 'runtime_sha256', 'annotation_identity')}
    value = {'schema': bank.SCHEMA, **identity, 'original_bank_sha256': job['original_bank_sha256'],
             'public_records_sha256': job['public_records_sha256'],
             'native_qualified': False, 'novelty_qualified': False, 'rows': kept}
    qualification = {'schema': 'pr100-parent-bank-qualification-v1', **identity,
        'bank_sha256': bank.sha(bank.raw(value)), 'original_bank_sha256': job['original_bank_sha256'],
        'public_records_sha256': job['public_records_sha256'], 'tested_rows': len(rows),
        'annotated_rows': len(kept), 'existing_rows_preserved': len(old),
        'annotation_reasons': dict(Counter(f['reason'] for f in findings)),
        'coverage': dict(Counter(c for r in kept for c in r['cells'])),
        'native_qualified': False, 'novelty_qualified': False, 'publication_authorized': False}
    return {'bank': value, 'qualification': qualification}


def annotate_job(directory):
    from nanobody_opt import pr100_filter, pr100_novelty_runtime
    job = json.loads(bank.read(directory / 'job.json')); verify_source(job['source_sha'])
    initialize_annotation(job['before_epoch'])
    bank.require(pr100_filter.annotation_identity() == job['annotation_identity'], 'annotation_identity_drift')
    bank.require(rules.digest(pr100_novelty_runtime.runtime_identity()) == job['runtime_sha256'],
                 'annotation_runtime_drift')
    original = bank.read(job['original_bank_path']); public = bank.read(directory / 'public.json')
    bank.require(bank.sha(original) == job['original_bank_sha256']
                 and bank.sha(public) == job['public_records_sha256'], 'parent_inputs_changed')
    public = json.loads(public)
    bank.require(public['target'] == job['target'] and public['before_epoch'] == job['before_epoch'],
                 'public_bundle_identity')
    with ProcessPoolExecutor(max_workers=job['workers'],
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize_annotation,
            initargs=(job['before_epoch'],)) as pool:
        result = qualify_rows(job, json.loads(original), public,
                              lambda rows: pool.map(annotate_one, rows, chunksize=4))
    bank.require(rules.digest(pr100_novelty_runtime.runtime_identity()) == job['runtime_sha256'],
                 'annotation_runtime_drift')
    verify_source(job['source_sha'])
    output = Path(job['remote_directory']) / 'banks' / ('bank-' + result['qualification']['bank_sha256'] + '.json')
    bank.immutable(output, bank.raw(result['bank']))
    bank.atomic(directory / 'annotation.json', result)


def supervise(directory):
    """One bounded subreaper owner; process/log data never becomes public output."""
    from scripts import pr100_native_probe as containment
    job = json.loads(bank.read(directory / 'job.json'))
    containment.dedicated_controller()
    owner_lock = open(Path(job['remote_directory']) / 'annotation-owner.lock', 'a+b')
    child = None; started = time.monotonic(); result = {'status': 'failed'}
    def stop(*_):
        raise InterruptedError('parent_annotation_interrupted')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGHUP, stop)
    try:
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.atomic(directory / 'owner.json', {'pid': os.getpid(), 'started_at_unix': time.time()})
        available = set(os.sched_getaffinity(0))
        bank.require(set(job['cpu_affinity']) <= available, 'parent_cpu_affinity_unavailable')
        os.sched_setaffinity(0, job['cpu_affinity'])
        runtime = job['annotation_runtime']
        environment = {'PATH': runtime['path'], 'PYTHONPATH': str(ROOT) + ':' + runtime['python_path'],
            'HOME': '/root', 'USER': 'root', 'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1',
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': ''}
        with (directory / 'private-annotation.log').open('ab') as log:
            child = subprocess.Popen([runtime['python'], '-B', str(Path(__file__).resolve()),
                'annotate', '--directory', str(directory)], cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            bank.require(child.wait(timeout=job['wall_seconds']) == 0, 'annotation_child_failed')
        result = json.loads(bank.read(directory / 'annotation.json')); result['status'] = 'complete'
    except BaseException as error:
        result = {'status': 'failed', 'error_type': type(error).__name__}
    finally:
        cleanup = containment.stop_descendants([] if child is None else [child], timeout=8)
        result.update(cleanup=cleanup, elapsed_seconds=time.monotonic() - started)
        if cleanup['remaining_descendants'] or not cleanup['fixed_point_reached']:
            result['status'] = 'containment_failure'
        if result['status'] == 'complete':
            result['qualification'].update(status='complete', cleanup=cleanup,
                elapsed_seconds=result['elapsed_seconds'], cpu_affinity=job['cpu_affinity'],
                completed_at_unix=time.time())
        bank.atomic(directory / 'result.json', result)
        owner_lock.close()


def remote_request(request):
    """Invoked on slave from stdin; immutable intent makes retries nonduplicating."""
    job = request['job']; directory = Path(request['directory'])
    bank.require(directory.parent == Path(job['remote_directory']) / 'jobs'
                 and directory.name == request['request_sha256'], 'invalid_parent_job_path')
    bank.require(bank.sha(bank.raw(job)) == request['request_sha256'], 'parent_request_changed')
    bank.require(str(ROOT) == job['release'], 'parent_release_changed')
    if request['operation'] == 'start':
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                pass  # Another retry owns this immutable request; never start a second child.
            else:
                child = None
                try:
                    bank.immutable(directory / 'job.json', bank.raw(job))
                    verify_source(job['source_sha'])
                    public = base64.b64decode(request['public'], validate=True)
                    bank.require(len(public) <= bank.MAX_BYTES and bank.sha(public) == job['public_records_sha256'],
                                 'public_transport_changed')
                    bank.immutable(directory / 'public.json', public)
                    with (directory / 'private-supervisor.log').open('ab') as log:
                        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                            'supervise', '--directory', str(directory)], cwd=ROOT,
                            env={'PATH': '/usr/bin:/bin', 'HOME': '/root', 'PYTHONDONTWRITEBYTECODE': '1'},
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                    bank.atomic(directory / 'launch.json', {'pid': child.pid, 'request_sha256': request['request_sha256']})
                except Exception as error:
                    if child is None:  # Source/input failure before the only scientific process launch.
                        bank.atomic(directory / 'result.json', {'status': 'failed',
                            'error_type': type(error).__name__, 'pre_spawn': True,
                            'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}})
    bank.require(request['operation'] in ('start', 'poll'), 'invalid_parent_operation')
    if not directory.exists():
        return {'running': False, 'not_started': True, 'request_sha256': request['request_sha256']}
    if (directory / 'result.json').is_file():
        return {'running': False, 'request_sha256': request['request_sha256'],
                'result': json.loads(bank.read(directory / 'result.json'))}
    return {'running': True, 'request_sha256': request['request_sha256'],
            'launch_uncertain': not (directory / 'launch.json').is_file()}


def transport(host, release, request):
    command = ['ssh', '-T', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes', '-oConnectTimeout=8',
               '-p', str(host['port'])]
    if host.get('host_key_alias'):
        command += ['-oHostKeyAlias=' + host['host_key_alias']]
    command += [host['address'], shlex.join(['/usr/bin/python3', '-B',
                    release + '/nanobody_opt/pr100_parent_feed.py', 'remote'])]
    response = subprocess.run(command, input=bank.raw(request), capture_output=True, timeout=40)
    bank.require(response.returncode == 0, 'parent_transport_failed')
    bank.require(len(response.stdout) <= bank.MAX_BYTES, 'parent_reply_size')
    # Never return or persist stderr, login banners, raw error messages, or logs.
    result = json.loads(response.stdout)
    bank.require(result.get('request_sha256') == request['request_sha256'], 'parent_reply_identity')
    return result


def promote(directory, pending, result, config, challenge):
    """CAS pointer update only after complete source/runtime-bound clean qualification."""
    bank.require(result.get('status') == 'complete', 'parent_qualification_failed')
    receipt = result['qualification']; value = result['bank']; job = pending['request']['job']
    expected = bank.expected_identity(config, challenge['target'])
    bank.require(all(job[k] == v for k, v in expected.items()), 'parent_configuration_changed')
    bank.require(receipt.get('cleanup', {}).get('remaining_descendants') == 0
        and receipt.get('cleanup', {}).get('fixed_point_reached') is True,
        'parent_cleanup_unverified')
    for key in (*expected, 'before_epoch', 'original_bank_sha256', 'public_records_sha256'):
        bank.require(receipt.get(key) == job[key] and value.get(key) == job[key], 'parent_qualification_binding')
    bank.require(receipt.get('status') == 'complete' and receipt['bank_sha256'] == bank.sha(bank.raw(value))
        and receipt['annotated_rows'] == len(value['rows']) > 0
        and receipt['existing_rows_preserved'] > 0, 'parent_qualification_incomplete')
    current = directory / 'current.json'
    current_sha = bank.sha(bank.read(current)) if current.exists() else None
    bank.require(current_sha == pending.get('previous_pointer_sha256'), 'parent_pointer_race')
    bank.immutable(directory / ('bank-' + receipt['bank_sha256'] + '.json'), bank.raw(value))
    receipt_sha = bank.sha(bank.raw(receipt))
    bank.immutable(directory / ('qualification-' + receipt_sha + '.json'), bank.raw(receipt))
    pointer = {'schema': bank.POINTER, **expected, 'before_epoch': job['before_epoch'],
        'bank_sha256': receipt['bank_sha256'], 'qualification_sha256': receipt_sha,
        'qualified_at_unix': receipt['completed_at_unix'],
        'remote_bank_path': job['remote_directory'] + '/banks/bank-' + receipt['bank_sha256'] + '.json'}
    bank.validate_pointer(pointer, config, challenge, fresh=False)
    bank.atomic(current, pointer)
    return pointer


def tick(config, *, now=None, fetch=fetch_epoch, call=transport, config_provider=None):
    now = time.time() if now is None else now
    cfg = bank.settings(config)
    if cfg is None:
        return {'status': 'disabled'}
    root = Path(cfg['state_directory']); root.mkdir(parents=True, exist_ok=True)
    pending_path = root / 'pending.json'
    if pending_path.exists():
        pending = json.loads(bank.read(pending_path))
        job = pending['request']['job']; directory = root / job['target']
        reply = call(pending['host'], pending['release'], {**pending['request'], 'operation': 'poll'})
        if reply.get('not_started') is True:
            # Reuse the SAME immutable request. Concurrent delayed SSH starts
            # race on mkdir and exactly one can own the annotation launch.
            call(pending['host'], pending['release'], pending['request'])
            return {'status': 'qualification_start_retried', 'request_sha256': pending['request']['request_sha256']}
        if reply.get('running'):
            unhealthy = now - job['created_at_unix'] > job['wall_seconds'] + 90
            return {'status': 'qualification_unhealthy' if unhealthy else 'qualification_running',
                    'owner_age_seconds': max(0, now - job['created_at_unix']),
                    'reason': 'owner_completion_unverified' if unhealthy else None,
                    'request_sha256': pending['request']['request_sha256'],
                    'launch_uncertain': bool(reply.get('launch_uncertain')), 'epoch': job['before_epoch']}
        result = reply['result']
        bank.immutable(directory / ('attempt-' + pending['request']['request_sha256'] + '.json'), bank.raw(result))
        # A dirty owner is never cleared by elapsed time or a new source release.
        bank.require(result.get('cleanup', {}).get('remaining_descendants') == 0
            and result.get('cleanup', {}).get('fixed_point_reached') is True, 'parent_owner_uncertain')
        try:
            config = config_provider() if config_provider is not None else config
            heartbeat = json.loads(bank.read(Path(config['state_root']) / 'heartbeat.json'))
            bank.require(type(heartbeat.get('epoch')) is int
                and type(heartbeat.get('updated_at_unix')) in (int, float)
                and -30 <= now - heartbeat['updated_at_unix'] <= 180, 'parent_clock_stale')
            epoch = heartbeat['epoch']
            challenge = {'epoch': epoch, 'target': rules.target_for_epoch(epoch)}
            pointer = promote(directory, pending, result, config, challenge)
            public = json.loads(base64.b64decode(pending['request']['public'], validate=True))
            identity = rules.digest({'public_sources': public['sources'],
                'original_bank_sha256': pointer['bank_sha256'], **bank.expected_identity(config, job['target'])})
            bank.atomic(directory / 'last-input.json', {'identity': identity})
            status = {'status': 'promoted', 'bank_sha256': pointer['bank_sha256'],
                      'rows': result['qualification']['annotated_rows'], 'epoch': epoch}
        except (ValueError, KeyError, TypeError) as error:
            status = {'status': 'retained_previous', 'error_type': type(error).__name__,
                      'reason': str(error) if isinstance(error, ValueError) else 'invalid_qualification',
                      'epoch': job['before_epoch']}
        pending_path.unlink()
        return status
    heartbeat = json.loads(bank.read(Path(config['state_root']) / 'heartbeat.json'))
    bank.require(type(heartbeat.get('epoch')) is int and type(heartbeat.get('updated_at_unix')) in (int, float)
        and -30 <= now - heartbeat['updated_at_unix'] <= 180, 'parent_clock_stale')
    epoch = heartbeat['epoch']; target = rules.target_for_epoch(epoch)
    challenge = {'epoch': epoch, 'target': target}
    directory = root / target; directory.mkdir(parents=True, exist_ok=True)
    ledger_path = directory / 'public-ledger.json'
    ledger = json.loads(bank.read(ledger_path)) if ledger_path.exists() else {}
    public, count = public_refresh(directory, ledger, epoch, target,
                                  limit=cfg['max_epoch_requests'], now=now, fetch=fetch)
    original_path, original_sha = bank.reannotation_base(config, challenge)
    bank.require(bank.hex64(original_sha), 'configured_bank_requires_hash')
    digest = bank.sha(bank.raw(public))
    input_identity = rules.digest({'public_sources': public['sources'], 'original_bank_sha256': original_sha,
                                  **bank.expected_identity(config, target)})
    complete_path = directory / 'last-input.json'
    if complete_path.exists() and json.loads(bank.read(complete_path)).get('identity') == input_identity:
        return {'status': 'unchanged', 'epochs_fetched': count, 'epoch': epoch}
    host = config['hosts']['slave']; runtime = host['novelty_runtime']
    job = {**bank.expected_identity(config, target), 'before_epoch': epoch, 'created_at_unix': now,
        'release': host['release'],
        'original_bank_path': original_path, 'original_bank_sha256': original_sha,
        'public_records_sha256': digest, 'annotation_runtime': runtime,
        'remote_directory': cfg['remote_directory'], 'cpu_affinity': cfg['cpu_affinity'],
        'workers': cfg['workers'], 'wall_seconds': cfg['wall_seconds']}
    if cfg.get('developability_db'):
        job['developability_db'] = cfg['developability_db']
    request_sha = bank.sha(bank.raw(job))
    request = {'operation': 'start', 'request_sha256': request_sha, 'job': job,
               'directory': cfg['remote_directory'] + '/jobs/' + request_sha,
               'public': base64.b64encode(bank.raw(public)).decode()}
    pointer_path = directory / 'current.json'
    pending = {'request': request, 'release': host['release'], 'input_identity': input_identity,
        'host': {k: host[k] for k in ('address', 'port', 'host_key_alias') if k in host},
        'previous_pointer_sha256': bank.sha(bank.read(pointer_path)) if pointer_path.exists() else None}
    bank.atomic(pending_path, pending)  # Ownership persisted before uncertain transport.
    call(pending['host'], pending['release'], request)
    return {'status': 'qualification_dispatched', 'epoch': epoch, 'epochs_fetched': count,
            'request_sha256': request_sha, 'public_rows': len(public['rows'])}


def health_summary(config):
    """Small operational summary, never raw bank/profiles or private process logs."""
    cfg = bank.settings(config)
    try:
        clock = json.loads(bank.read(Path(config['state_root']) / 'heartbeat.json'))
        challenge = {'epoch': clock['epoch'], 'target': rules.target_for_epoch(clock['epoch'])}
        directory = Path(cfg['state_directory']) / challenge['target']
        pointer = bank.validate_pointer(json.loads(bank.read(directory / 'current.json')),
                                        config, challenge, fresh=False)
        receipt = json.loads(bank.read(directory / ('qualification-' + pointer['qualification_sha256'] + '.json')))
        return {'last_success_unix': pointer['qualified_at_unix'],
                'last_success_age_seconds': max(0, time.time() - pointer['qualified_at_unix']),
                'bank_before_epoch': pointer['before_epoch'],
                'bank_age_epochs': challenge['epoch'] - pointer['before_epoch'],
                'qualified_rows': receipt['annotated_rows'], 'cell_coverage': receipt['coverage'],
                'bank_sha256': pointer['bank_sha256']}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {'qualified_pointer_status': 'unavailable', 'pointer_error_type': type(error).__name__}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('tick', 'remote', 'supervise', 'annotate'))
    parser.add_argument('--config', type=Path); parser.add_argument('--directory', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == 'remote':
            content = sys.stdin.buffer.read(bank.MAX_BYTES + 1)
            bank.require(len(content) <= bank.MAX_BYTES, 'parent_request_size')
            result = remote_request(json.loads(content))
        elif args.mode == 'supervise':
            supervise(args.directory); return 0
        elif args.mode == 'annotate':
            annotate_job(args.directory); return 0
        else:
            config = json.loads(bank.read(args.config)); verify_source(config['source_sha'])
            cfg = bank.settings(config)
            if cfg is None:
                result = {'status': 'disabled'}
            else:
                root = Path(cfg['state_directory']); root.mkdir(parents=True, exist_ok=True)
                with (root / 'tick.lock').open('a+b') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        result = {'status': 'tick_already_running'}
                    else:
                        result = tick(config, config_provider=lambda: json.loads(bank.read(args.config)))
                        result.update(health_summary(config))
                        bank.atomic(root / 'status.json', {'updated_at_unix': time.time(), **result})
        print(json.dumps(result, allow_nan=False)); return 0
    except Exception as error:
        result = {'status': 'failed', 'error_type': type(error).__name__}
        if args.mode == 'tick' and 'cfg' in locals() and cfg:
            try:
                bank.atomic(Path(cfg['state_directory']) / 'status.json',
                            {'updated_at_unix': time.time(), **result})
            except OSError:
                pass
        print(json.dumps(result)); return 1


if __name__ == '__main__':
    raise SystemExit(main())
