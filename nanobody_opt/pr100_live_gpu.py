"""Bounded cooperative GPU ownership for the Pi-driven PR100 baseline.

Legacy scorer-daemon GPU locks are process-local: this lease is NOT exclusive
against that daemon. Unknown/active residents refuse admission; new contention
stops only our owned process tree. Existing dispatchers must opt into the same
file locks, or the controller must exclude this GPU from their dispatch lists.
No resident worker is killed or restarted here.
"""
from __future__ import annotations

import argparse
import array
from contextlib import ExitStack, contextmanager
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import termios
import time

ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = '/root/nanobody_opt/scorer_daemon.py'
ENV_KEYS = {'PATH', 'HOME', 'USER', 'LANG', 'LD_LIBRARY_PATH', 'PYTHONPATH',
            'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'NANOBODY_FAST_SCORER_CONFIG', 'NANOBODY_SCORER_DAEMON',
            'NANOBODY_SCORER_SOCKET', 'PYTHONDONTWRITEBYTECODE'}


class Unavailable(RuntimeError):
    pass


class LeaseExpired(TimeoutError):
    pass


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def lease_locks(gpu, directory, dispatcher_locks=()):
    """Kernel-owned leases cannot survive the owning process's death."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    own = directory / ('gpu-' + str(gpu) + '.lock')
    paths = [own, *map(Path, dispatcher_locks)]
    if len({str(p.resolve()) for p in paths}) != len(paths):
        raise Unavailable('duplicate_coordination_lock')
    with ExitStack() as stack:
        for path in sorted(paths, key=lambda p: str(p.resolve())):
            if path != own and (not path.is_file() or path.suffix != '.lock'):
                raise Unavailable('dispatcher_lock_missing')
            flags = os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
            if path == own:
                flags |= os.O_CREAT
            fd = os.open(path, flags, 0o600)
            stack.callback(os.close, fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Unavailable('coordination_lock_busy') from None
        yield


def _query(fields, kind):
    try:
        p = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + fields,
                            '--format=csv,noheader,nounits'], capture_output=True,
                           text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        raise Unavailable('gpu_telemetry_failed') from None
    if p.returncode:
        raise Unavailable('gpu_telemetry_failed')
    return [[s.strip() for s in line.split(',')] for line in p.stdout.splitlines() if line.strip()]


def _resident(pid, scorer):
    """Read only safe kernel facts; command arguments never leave this function."""
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        args = (path / 'cmdline').read_bytes().split(b'\0')
        known = (os.fsencode(WORKER_SCRIPT) in args and b'--shard-worker' in args
                 and b'--scorer' in args)
        if known:
            pos = args.index(b'--scorer')
            known = pos + 1 < len(args) and args[pos + 1] == os.fsencode(scorer)
        idle = known and (path / 'wchan').read_text().strip() == 'pipe_read'
        pending = None
        if idle:
            fd = os.open(path / 'fd/0', os.O_RDONLY | os.O_NONBLOCK)
            try:
                size = array.array('i', [0]); fcntl.ioctl(fd, termios.FIONREAD, size, True)
                pending = size[0]
            finally:
                os.close(fd)
        return {'pid': pid, 'start_ticks': int(stat[19]),
                'cpu_ticks': int(stat[11]) + int(stat[12]),
                'known_idle': bool(idle and pending == 0)}
    except (OSError, ValueError, IndexError):
        return {'pid': pid, 'start_ticks': None, 'cpu_ticks': None, 'known_idle': False}


def gpu_snapshot(gpu, scorer):
    rows = _query('index,uuid,memory.total,memory.free,utilization.gpu', 'gpu')
    found = [r for r in rows if r[0] == str(gpu)]
    if len(found) != 1 or len(found[0]) != 5:
        raise Unavailable('gpu_missing')
    row = found[0]
    try:
        result = {'gpu': int(row[0]), 'uuid': row[1], 'total_mib': int(row[2]),
                  'free_mib': int(row[3]), 'utilization': int(row[4]), 'processes': []}
        for r in _query('gpu_uuid,pid', 'compute-apps'):
            if len(r) != 2:
                raise ValueError('telemetry shape')
            if r[0] == result['uuid']:
                state = _resident(int(r[1]), scorer)
                # The GPU listing can outlive an exited process. A still-live
                # unknown process remains a blocker; never excuse read errors.
                if state['start_ticks'] is None and not (Path('/proc') / r[1]).exists():
                    continue
                result['processes'].append(state)
    except (TypeError, ValueError):
        raise Unavailable('gpu_telemetry_invalid') from None
    return result


def admit(first, second, minimum_free_mib):
    for sample in (first, second):
        if sample['free_mib'] < minimum_free_mib:
            raise Unavailable('insufficient_gpu_headroom')
        if sample['utilization'] != 0 or any(not p['known_idle'] for p in sample['processes']):
            raise Unavailable('gpu_not_idle')
    if (first['gpu'], first['uuid']) != (second['gpu'], second['uuid']):
        raise Unavailable('gpu_identity_changed')
    identity = lambda s: sorted((p['pid'], p['start_ticks'], p['cpu_ticks']) for p in s['processes'])
    if identity(first) != identity(second):
        raise Unavailable('resident_changed_during_admission')


def uncontended(current, admitted, owned_pids, minimum_free_mib):
    if (current['gpu'], current['uuid']) != (admitted['gpu'], admitted['uuid']):
        return False
    if current['free_mib'] < minimum_free_mib:
        return False
    residents = {(p['pid'], p['start_ticks']) for p in admitted['processes']}
    for p in current['processes']:
        if p['pid'] in owned_pids:
            continue
        if (p['pid'], p['start_ticks']) not in residents or not p['known_idle']:
            return False
    return True


def _controller():
    sys.path.insert(0, str(ROOT))
    path = ROOT / 'scripts/pr100_native_probe.py'
    spec = importlib.util.spec_from_file_location('_pr100_gpu_owned_tree', path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_reserved(job, *, snapshot=gpu_snapshot, control=None, launch=subprocess.Popen,
                 clock=time.monotonic, sleep=time.sleep):
    """Run in a dedicated session/process. TTL includes admission and cleanup.

    Caller supplies a public command/environment, never credential values. This
    helper prints no subprocess streams or arguments. Receipt status says only
    process completion: the worker must validate its scientific result itself.
    """
    gpu, ttl = job['gpu'], job['ttl_seconds']
    minimum = job.get('minimum_free_mib', 8192)
    if (isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
            or isinstance(ttl, bool) or not isinstance(ttl, (int, float))
            or not math.isfinite(ttl) or not 10 <= ttl <= 1800
            or not isinstance(minimum, int) or not 1024 <= minimum <= 200000):
        raise ValueError('invalid GPU reservation bounds')
    if not isinstance(job['command'], list) or not job['command'] or any(
            not isinstance(v, str) or not v or '\0' in v for v in job['command']):
        raise ValueError('invalid public command')
    supplied_env = job.get('env', {})
    if set(supplied_env) - ENV_KEYS or any(not isinstance(v, str) for v in supplied_env.values()):
        raise ValueError('unregistered environment variable')
    control = control or _controller()
    control.dedicated_controller()
    began = clock(); deadline = began + ttl
    result = {'schema': 'pr100-cooperative-gpu-lease-v1', 'gpu': gpu,
              'owner_pid': os.getpid(), 'ttl_seconds': ttl, 'exclusive_gpu': False,
              'started_monotonic': began, 'deadline_monotonic': deadline,
              'created_unix': time.time(), 'expires_unix': time.time() + ttl,
              'coordination': 'file-lock participants only; legacy daemon lock is process-local',
              'dispatcher_locks': list(job.get('dispatcher_locks', [])), 'status': 'preparing'}
    children = []; prior = {}
    def expired(signum, frame):
        raise LeaseExpired('GPU reservation expired')
    def interrupted(signum, frame):
        raise InterruptedError('owned GPU reservation interrupted')
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            prior[sig] = signal.signal(sig, interrupted)
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise Unavailable('existing_process_deadline')
        prior[signal.SIGALRM] = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, ttl - 4)
        with lease_locks(gpu, job['lock_directory'], job.get('dispatcher_locks', [])):
            try:
                first = snapshot(gpu, job['scorer'])
                sleep(2)
                second = snapshot(gpu, job['scorer'])
                admit(first, second, minimum)
                result.update(status='reserved', admission=second)
                atomic(job['receipt'], result)
                env = dict(supplied_env, CUDA_VISIBLE_DEVICES=str(gpu))
                if clock() >= deadline - 5:
                    raise Unavailable('reservation_expired_before_launch')
                child = launch(job['command'], env=env, cwd=job['cwd'],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
                children.append(child); result.update(status='running', child_pid=child.pid)
                atomic(job['receipt'], result)
                while child.poll() is None:
                    if clock() >= deadline - 5:
                        result['status'] = 'expired'; break
                    if not uncontended(snapshot(gpu, job['scorer']), second,
                                       control.descendants(), 512):
                        result['status'] = 'contention_abort'; break
                    sleep(min(2, max(.01, deadline - 5 - clock())))
                else:
                    result['status'] = 'process_finished'
                result['child_exit_code'] = child.poll()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                # Keep the lock until the entire owned process tree is gone.
                if children:
                    result['cleanup'] = control.stop_descendants(children, timeout=4)
                    children.clear()
    except LeaseExpired:
        result['status'] = 'expired'
    except Unavailable as error:
        result.update(status='unavailable', reason=str(error))
    except BaseException as error:
        result.update(status='failed', error_class=type(error).__name__)
    finally:
        if signal.SIGALRM in prior:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if children:
            result['cleanup'] = control.stop_descendants(children, timeout=4)
        for sig, handler in prior.items():
            signal.signal(sig, handler)
        cleanup = result.get('cleanup')
        if cleanup and (cleanup['remaining_descendants'] or not cleanup['fixed_point_reached']):
            result['status'] = 'containment_failure'
        result['released'] = True
        result['elapsed_seconds'] = clock() - began
        atomic(job['receipt'], result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    if os.getsid(0) != os.getpid():
        os.setsid()
    try:
        result = run_reserved(json.loads(args.job.read_text()))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'invalid_job', 'error_class': type(error).__name__}))
        return 2
    print(json.dumps({k: result.get(k) for k in ('status', 'reason', 'child_exit_code', 'released')}))
    return 0 if result['status'] == 'process_finished' and result['child_exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
