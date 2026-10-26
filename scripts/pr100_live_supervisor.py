#!/usr/bin/env python3
"""Bounded supervisor for one PR100 live controller phase on a GPU/CPU host.

Launched detached by the controller's remote transport with ``--spec <dir>/spec.json``.
Runs the phase command in its own process group, enforces the wall-clock cap, the
absolute deadline and the epoch block reserve (re-read from ``clock_path``), kills the
whole group (TERM, then KILL) on any cutoff, and always writes ``receipt_path``
atomically.  Child stdout/stderr are kept in the job directory for diagnosis.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

BLOCK_SECONDS = 12.0
TAIL_BYTES = 4096
GRACE_SECONDS = 10.0
POLL_SECONDS = 1.0
MAXIMUM_CLOCK_AGE = 180.0
MAXIMUM_CLOCK_SKEW = 30.0
# A running child is never killed for a clock that merely stopped updating (a brief Pi-to-host
# ssh outage looks exactly like that). deadline_unix and wall_seconds stay the hard bounds; only a
# clock that has been unusable for this long is treated as a controller that is gone.
CLOCK_GRACE_SECONDS = 600.0
REQUIRED = ('command', 'cwd', 'environment', 'wall_seconds', 'deadline_unix', 'receipt_path',
            'cpu_slots', 'clock_path', 'epoch', 'epoch_length', 'reserve_blocks')
# Strip anything that looks like a credential before it can leave the host.
_SECRET = re.compile(r'(?i)(authorization|bearer|token|secret|password|api[_-]?key|private[_-]?key)'
                     r'(\s*[:=]\s*|\s+)\S+')
_LONG_HEX = re.compile(r'\b[A-Za-z0-9+/_-]{48,}\b')


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, sort_keys=True))
    os.replace(temporary, path)


def sanitized_tail(path, limit=TAIL_BYTES):
    try:
        with open(path, 'rb') as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            raw = handle.read(limit)
    except OSError:
        return ''
    text = raw.decode('utf-8', 'replace')
    text = _SECRET.sub(lambda m: m.group(1) + '=<redacted>', text)
    return _LONG_HEX.sub('<redacted>', text)


def blocks_remaining(clock_path, epoch, now=None):
    """Blocks left in ``epoch`` per the clock file, or None if it is unusable/stale."""
    now = time.time() if now is None else now
    try:
        state = json.loads(Path(clock_path).read_text())
        length, block = int(state['epoch_length']), int(state['block'])
        age = now - float(state['updated_at_unix'])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    # The clock is written with the Pi's wall time; tolerate small host clock skew in
    # either direction instead of killing the phase the moment the remote runs behind.
    if length <= 0 or not -MAXIMUM_CLOCK_SKEW <= age <= MAXIMUM_CLOCK_AGE:
        return None
    age = max(0.0, age)
    if block // length != epoch:
        return 0.0
    return (epoch + 1) * length - block - age / BLOCK_SECONDS


def validate(spec):
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise ValueError('spec missing keys: ' + ', '.join(missing))
    command = spec['command']
    if not isinstance(command, list) or not command or not all(isinstance(v, str) and v for v in command):
        raise ValueError('spec command must be a non-empty list of strings')
    if not isinstance(spec['environment'], dict):
        raise ValueError('spec environment must be an object')
    for key in ('wall_seconds', 'deadline_unix', 'reserve_blocks'):
        if (isinstance(spec[key], bool) or not isinstance(spec[key], (int, float))
                or not math.isfinite(spec[key]) or spec[key] < 0):
            raise ValueError(f'spec {key} must be a non-negative number')
    if int(spec['epoch_length']) <= 0 or int(spec['cpu_slots']) <= 0:
        raise ValueError('spec epoch_length and cpu_slots must be positive')


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_group(child, grace=GRACE_SECONDS):
    """TERM the whole group, wait ``grace``, then KILL whatever is left."""
    pgid = child.pid
    _signal_group(pgid, signal.SIGTERM)
    end = time.monotonic() + grace
    while time.monotonic() < end:
        child.poll()
        if child.returncode is not None and not _group_alive(pgid):
            return False
        time.sleep(0.05)
    _signal_group(pgid, signal.SIGKILL)
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    return True


def _cpu_preexec(slots):
    def apply():
        os.setsid()
        if hasattr(os, 'sched_setaffinity'):
            try:
                cpus = sorted(os.sched_getaffinity(0))[:slots]
                if cpus:
                    os.sched_setaffinity(0, cpus)
            except OSError:
                pass
    return apply


def supervise(spec, *, poll=POLL_SECONDS, grace=GRACE_SECONDS, containment=None):
    started_wall, started = time.time(), time.monotonic()
    receipt = {'schema': 'pr100-live-supervisor-v1', 'epoch': spec.get('epoch'),
               'started_at_unix': started_wall, 'exit_code': None, 'reason': 'not_started',
               'killed': False, 'killed_hard': False, 'elapsed_seconds': 0.0, 'stderr_tail': ''}
    receipt_path = spec.get('receipt_path')
    directory = Path(receipt_path).parent if receipt_path else Path.cwd()
    stdout_path, stderr_path = directory / 'supervisor-stdout.log', directory / 'supervisor-stderr.log'
    receipt.update(stdout_path=str(stdout_path), stderr_path=str(stderr_path))
    child = None
    cleanup_attempted = False
    interrupted = []
    handlers = {}

    def interrupted_by(signum, frame):
        interrupted.append(signum)

    try:
        validate(spec)
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted_by)
        epoch, reserve = int(spec['epoch']), float(spec['reserve_blocks'])
        cutoff = min(started_wall + float(spec['wall_seconds']), float(spec['deadline_unix']))
        receipt['cutoff_unix'] = cutoff
        left = blocks_remaining(spec['clock_path'], epoch)
        if time.time() >= cutoff:
            receipt['reason'] = 'deadline_before_start'
            return receipt
        if left is None or left <= reserve:
            receipt['reason'] = 'clock_unavailable' if left is None else 'reserve_before_start'
            return receipt
        directory.mkdir(parents=True, exist_ok=True)
        with open(stdout_path, 'ab') as out, open(stderr_path, 'ab') as err:
            child = subprocess.Popen(spec['command'], cwd=spec['cwd'],
                                     env={k: str(v) for k, v in spec['environment'].items()},
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     preexec_fn=_cpu_preexec(int(spec['cpu_slots'])))
        receipt['pid'] = child.pid
        reason = None
        clock_lost_since = None
        while child.poll() is None:
            now = time.time()
            if interrupted:
                reason = 'terminated'
            elif now >= started_wall + float(spec['wall_seconds']):
                reason = 'wall_seconds'
            elif now >= float(spec['deadline_unix']):
                reason = 'deadline_unix'
            else:
                left = blocks_remaining(spec['clock_path'], epoch, now)
                if left is None:
                    clock_lost_since = clock_lost_since if clock_lost_since is not None else now
                    if now - clock_lost_since >= CLOCK_GRACE_SECONDS:
                        reason = 'clock_unavailable'
                else:
                    clock_lost_since = None
                    if left <= reserve:
                        reason = 'reserve_blocks'
            if reason:
                receipt['killed'] = True
                cleanup_attempted = True
                receipt['killed_hard'] = stop_group(child, grace)
                break
            time.sleep(poll)
        if not reason:
            # The leader exited; reap any detached descendants left in its group.
            if _group_alive(child.pid):
                receipt['orphans_killed'] = True
                cleanup_attempted = True
                stop_group(child, grace)
        receipt['exit_code'] = child.returncode
        receipt['reason'] = reason or ('completed' if child.returncode == 0 else 'nonzero_exit')
    except Exception as error:  # the receipt must still be written
        receipt['reason'] = 'supervisor_error'
        receipt['error_type'] = type(error).__name__
        receipt['error'] = sanitized_tail_text(str(error))
    finally:
        # Service stops and unexpected exceptions must also reap the owned tree.
        # On Linux the production supervisor is a subreaper, so setsid/double-fork
        # children remain ours even after their original parent exits.
        try:
            if child is not None and not cleanup_attempted and (child.poll() is None or _group_alive(child.pid)):
                receipt['killed'] = True
                receipt['killed_hard'] = stop_group(child, grace) or receipt['killed_hard']
            if containment is not None:
                cleanup = containment.stop_descendants([child] if child is not None else (), timeout=5)
                receipt['cleanup'] = cleanup
                if cleanup['remaining_descendants']:
                    receipt['reason'] = 'cleanup_failed'
            if child is not None:
                receipt['exit_code'] = child.poll()
        except Exception as error:
            receipt.update(reason='cleanup_failed', cleanup_error_type=type(error).__name__)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt['stderr_tail'] = sanitized_tail(stderr_path)
        if receipt_path:
            atomic(receipt_path, receipt)
    return receipt


def sanitized_tail_text(text, limit=TAIL_BYTES):
    text = text[-limit:]
    return _LONG_HEX.sub('<redacted>', _SECRET.sub(lambda m: m.group(1) + '=<redacted>', text))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        spec = json.loads(args.spec.read_text())
    except (OSError, ValueError) as error:
        fallback = args.spec.parent / 'supervisor-receipt.json'
        atomic(fallback, {'schema': 'pr100-live-supervisor-v1', 'reason': 'spec_unreadable',
                          'error_type': type(error).__name__, 'exit_code': None, 'killed': False,
                          'elapsed_seconds': 0.0, 'stderr_tail': ''})
        return 2
    containment = None
    if sys.platform == 'linux':
        # Avoid an unrelated concrete scripts package in the native environment.
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        try:
            module_spec = importlib.util.spec_from_file_location(
                '_pr100_owned_processes', root / 'scripts' / 'pr100_native_probe.py')
            containment = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(containment)
            containment.dedicated_controller()
        except Exception as error:
            atomic(spec.get('receipt_path', args.spec.parent / 'supervisor-receipt.json'),
                   {'schema': 'pr100-live-supervisor-v1', 'reason': 'containment_unavailable',
                    'error_type': type(error).__name__, 'exit_code': None, 'killed': False})
            return 2
    receipt = supervise(spec, containment=containment)
    return 0 if receipt['reason'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
