"""Bounded slot refill around the unchanged original native profiling adapter.

No scientific work is queued beyond the number of available slots. A slot is
reused only after the original adapter has returned, including worker cleanup.
The enclosing phase supervisor retains ownership of every process descendant.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class NotStarted:
    """Owned admission was withdrawn before invoking any scientific adapter."""
    reason: str = 'minimum_native_window_closed'


def run(next_calls, on_result, workers, *, develop=None, clock=time.monotonic):
    from nanobody_opt import pr100_filter
    develop = develop or pr100_filter.develop
    if type(workers) is not int or workers < 1:
        raise ValueError('refill workers must be a positive integer')

    async def execute(call):
        timeout = call['timeout_seconds']
        if call.get('deadline') is not None:
            timeout = min(timeout, call['deadline'] - clock())
            if timeout < call['minimum_window']:
                return NotStarted()
        values = await develop([call['sequence']], 1,
            task_timeout_seconds=timeout,
            on_result=call['record'], on_result_error=call['record_error'])
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError('incomplete native singleton result')
        return values[0]

    async def owned():
        pending = {}
        admitted = completed = maximum = never_started = 0
        started = clock()
        try:
            while True:
                free = workers - len(pending)
                calls = list(next_calls(free)) if free else []
                if len(calls) > free:
                    raise ValueError('native refill exceeded available slots')
                existing = {call['sequence'] for call in pending.values()}
                keys = [call['sequence'] for call in calls]
                if len(set(keys)) != len(keys) or existing.intersection(keys):
                    raise ValueError('native refill duplicated an active candidate')
                for call in calls:
                    pending[asyncio.create_task(execute(call))] = call
                admitted += len(calls)
                maximum = max(maximum, len(pending))
                if not pending:
                    break
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                # Stable admission order makes simultaneous completions
                # deterministic. Every completion is recorded before refill.
                for task in list(pending):
                    if task not in done:
                        continue
                    call = pending.pop(task)
                    value = task.result()
                    on_result(call, value)
                    never_started += isinstance(value, NotStarted)
                    completed += not isinstance(value, NotStarted)
        finally:
            # Cancelling an original adapter call triggers ProfileExecution
            # cleanup and awaits its worker. Do not start a replacement while
            # that cleanup is still in progress, even after callback failure.
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        return {'admitted_calls': admitted, 'completed_calls': completed,
                'never_started_calls': never_started, 'maximum_in_flight': maximum,
                'elapsed_seconds': clock() - started}

    def private_loop():
        loop = asyncio.new_event_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=workers,
            thread_name_prefix='pr100-native-refill'))
        return loop

    with asyncio.Runner(loop_factory=private_loop) as runner:
        return runner.run(owned())
