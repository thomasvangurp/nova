"""Cancel an epoch's command tree before it can occupy the next epoch."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


class EpochDeadlineExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class EpochDeadline:
    chain_state: Path
    epoch: int
    reserve_blocks: int = 0
    maximum_state_age: float = 180.0

    def remaining(self) -> float:
        try:
            state = json.loads(self.chain_state.read_text())
            stamp = float(state.get('updated_at_unix', self.chain_state.stat().st_mtime))
            age = time.time() - stamp
            length, block = int(state['epoch_length']), int(state['block'])
            current = int(state['epoch'])
            if length <= 0 or current != block // length or current != self.epoch:
                raise EpochDeadlineExceeded('epoch changed; old work cancelled')
            if not 0 <= age <= self.maximum_state_age:
                raise EpochDeadlineExceeded('chain clock is stale; work cancelled')
            seconds = (length - block % length - self.reserve_blocks) * 12.0 - age
            if seconds <= 0:
                raise EpochDeadlineExceeded('epoch handoff deadline reached')
            return seconds
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise EpochDeadlineExceeded('chain clock unavailable; work cancelled') from error


def _no_live_group_members(pgid: int) -> bool:
    """A bounded process-state check; never inspect arguments or environment."""
    try:
        result = subprocess.run(['ps', '-axo', 'pgid=,stat='], capture_output=True,
                                text=True, timeout=2, check=False)
        if result.returncode:
            return False
        for line in result.stdout.splitlines():
            group, status = line.split()
            if int(group) == pgid and not status.startswith('Z'):
                return False
        return True
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin's killpg filters zombies before counting signallable members
        # and can return EPERM for a zombie-only group. Do not mask a real
        # permission failure or abandon a live descendant on any platform.
        if sys.platform != 'darwin' or not _no_live_group_members(pgid):
            raise


def _stop_group(process: subprocess.Popen) -> None:
    # Children may survive their immediate parent, so always address the group.
    _signal_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    _signal_group(process.pid, signal.SIGKILL)
    process.wait()


def run(command, *, deadline: EpochDeadline | None, timeout: float | None = None,
        check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run in a private process group, checking the live clock every second."""
    if deadline is not None:
        deadline.remaining()
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        while True:
            remaining = deadline.remaining() if deadline is not None else float('inf')
            if timeout is not None:
                remaining = min(remaining, timeout - (time.monotonic() - started))
            if remaining <= 0:
                raise EpochDeadlineExceeded('command time budget exhausted')
            try:
                stdout, stderr = process.communicate(timeout=min(1.0, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if deadline is not None:
            deadline.remaining()  # A result completed across rollover is stale.
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result
    finally:
        _stop_group(process)
