"""Make a swallowed failure loud without making it fatal.

Several pipeline steps deliberately survive a failure -- a DB write that must
not block a publishable result, a history read that must not block the
provisional publish -- and used to say so only on a stdout nobody kept
(ARCHITECTURE_AUDIT §0.4: the producer wrote nothing for ~27 epochs and the
only trace was ``MUTATION_DB_UPDATE_FAILED`` on an uncaptured pipe).

``record_failure`` keeps that "survive" behaviour but leaves three traces:

* the full traceback on stderr (the tmux pane / log the step already has);
* one JSON line in a per-host failure log
  (``/root/artifacts/pipeline_failures.jsonl`` on the boxes, override with
  ``NOVA_FAILURE_LOG``), which ``tools/check_mining_liveness.py`` reads and
  alerts on;
* nothing else -- it never raises, so it cannot itself become the failure.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

DEFAULT_BOX_LOG = Path("/root/artifacts/pipeline_failures.jsonl")


def failure_log_path() -> Path:
    override = os.environ.get("NOVA_FAILURE_LOG")
    if override:
        return Path(override)
    if DEFAULT_BOX_LOG.parent.is_dir():
        return DEFAULT_BOX_LOG
    return Path(__file__).resolve().parent.parent / "artifacts" / "pipeline_failures.jsonl"


def record_failure(component: str, marker: str, error: BaseException | None = None,
                   **context: Any) -> None:
    """Log ``marker`` loudly for ``component``; never raises."""
    try:
        detail = f"{type(error).__name__}: {error}" if error is not None else ""
        print(f"{marker} component={component} {detail}".rstrip(), file=sys.stderr, flush=True)
        if error is not None:
            traceback.print_exception(type(error), error, error.__traceback__,
                                      file=sys.stderr)
        entry = {
            "at": time.time(),
            "at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "component": component,
            "marker": marker,
            "error": detail[:500],
            "context": {k: (v if isinstance(v, (int, float, bool, type(None))) else str(v)[:300])
                        for k, v in context.items()},
        }
        path = failure_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    except Exception:  # noqa: BLE001 - the recorder must never be the failure
        pass
