"""The miner's chain view, written for a publisher that must not open a second socket.

The publisher runs as its own process in its own Pixi environment.  Giving it a
WebSocket of its own would mean two clients disagreeing about the current block,
so it reads the miner's view from a file instead.

Schema 2 adds ``neurons``.  Schema 1 carried only a block and a target, which
left the consumer to remember which uid was ours -- and uids are RECYCLED when a
hotkey is deregistered.  That is not hypothetical: the miner published into a
slot another operator had taken over for 21 days, because every check keyed on a
remembered uid silently followed the new owner.  Exporting uid/hotkey/coldkey
rows lets the publisher resolve our fleet from the coldkey on every pass and
refuse to run when it holds nothing.

Like ``publication_gates``, this module has no third-party imports: the miner,
the publisher and the tests all run in different environments, and a contract
only one of them can import is not a contract.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 2


class EmptyRosterError(ValueError):
    """Raised rather than publishing a roster nobody appears in.

    An empty roster and a real deregistration are indistinguishable downstream,
    so a bad metagraph read must never be written as though it were fact.
    """


def neuron_rows(
    hotkeys: Sequence[str] | None,
    coldkeys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """uid/hotkey/coldkey for every neuron, as plain JSON-safe values."""
    hotkeys = list(hotkeys or [])
    coldkeys = list(coldkeys or [])
    return [
        {
            "uid": index,
            "hotkey": str(hotkey),
            # A metagraph that omits coldkeys yields a fleet of zero, which reads
            # exactly like being deregistered.  Keep the absence explicit so the
            # consumer can tell the two apart.
            "coldkey": str(coldkeys[index]) if index < len(coldkeys) else None,
        }
        for index, hotkey in enumerate(hotkeys)
    ]


def normalise_target(value: Any) -> str:
    """One target as a scalar string, however the config spells it.

    ``config.nanobody_target`` is a LIST even when it holds a single target, so
    ``str()`` on it yields ``"['P20809']"`` -- which then fails an equality check
    against the live config's ``'P20809'`` and leaves the producer retrying
    forever on a string mismatch. Unwrap here so no caller has to remember.
    """
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("nanobody_target is empty")
        value = value[0]
    text = str(value).strip()
    if not text:
        raise ValueError("nanobody_target is blank")
    return text


def build_payload(
    *,
    netuid: int,
    block: int,
    epoch_length: int,
    nanobody_target: str,
    miner_uid: int,
    neurons: Sequence[Mapping[str, Any]],
    now: float,
) -> dict[str, Any]:
    """Assemble the exported view, or refuse if the roster is empty."""
    if epoch_length <= 0:
        raise ValueError(f"epoch_length must be positive: {epoch_length}")
    rows = [dict(row) for row in neurons]
    if not rows:
        raise EmptyRosterError("metagraph exposed no neurons; keeping the last good file")
    return {
        "schema_version": SCHEMA_VERSION,
        "netuid": int(netuid),
        "block": int(block),
        "epoch_length": int(epoch_length),
        "epoch": int(block) // int(epoch_length),
        "nanobody_target": normalise_target(nanobody_target),
        "miner_uid": int(miner_uid),
        "neurons": rows,
        "updated_at_unix": float(now),
    }


def write_atomic(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write JSON so a reader never observes a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=str(path.parent), delete=False, suffix=".tmp"
    )
    try:
        with handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
