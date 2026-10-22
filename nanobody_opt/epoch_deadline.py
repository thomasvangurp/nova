"""Cancel an epoch's command tree before it can occupy the next epoch."""
from __future__ import annotations

import json
import os
import signal
import subprocess
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


def _stop_group(process: subprocess.Popen) -> None:
    # Children may survive their immediate parent, so always address the group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
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
