"""The miner's exported chain view — the contract the publisher resolves identity from."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt import chain_state

# Real ss58 addresses from the subnet-68 leaderboard: fleet.py rejects truncated
# ones, because a partial address matches no row and is then indistinguishable
# from holding no slots at all.
COLDKEY_OURS = "5EkCwcNTfxeNbbGJcRWLGetcFaMnaoyGPvo1kDHWAnXTeLU5"
COLDKEY_RIVAL = "5Fq7TBKWQqD4vDCqCPdsJFrJjVnJLLYPvxJjJZKVBLFP7Ldy"
HOTKEY_A = "5E9MgNCNnSjef97Xjew8xBVUK1eQYYj2ziu2k8qi1LGxkFbe"
HOTKEY_B = "5DAVZojBSaL4Bqgb6Ai63hQvuYkgvKdTBFQiavV75FXgERdy"
HOTKEY_RIVAL = "5CFVtQjHeZ4iTbMf6Fug7E1aLAKeERDEAomffRhSqaNDYJtp"


def test_rows_carry_uid_hotkey_and_coldkey():
    rows = chain_state.neuron_rows(["hk0", "hk1"], ["ck0", "ck1"])
    assert rows == [
        {"uid": 0, "hotkey": "hk0", "coldkey": "ck0"},
        {"uid": 1, "hotkey": "hk1", "coldkey": "ck1"},
    ]


def test_a_missing_coldkey_is_explicit_not_omitted():
    """A fleet of zero and a deregistration read identically; keep them separable."""
    rows = chain_state.neuron_rows(["hk0", "hk1"], ["ck0"])
    assert rows[1]["coldkey"] is None
    assert "coldkey" in rows[1]


def test_payload_is_schema_2_and_derives_the_epoch():
    payload = chain_state.build_payload(
        netuid=68,
        block=9048104,
        epoch_length=361,
        nanobody_target="P20809",
        miner_uid=8,
        neurons=chain_state.neuron_rows(["hk"], ["ck"]),
        now=1.0,
    )
    assert payload["schema_version"] == 2
    # 9048104 // 361 == 25064: the epoch numbering the leaderboard uses.
    assert payload["epoch"] == 25064
    assert payload["neurons"] == [{"uid": 0, "hotkey": "hk", "coldkey": "ck"}]


def test_an_empty_roster_refuses_rather_than_publishing():
    with pytest.raises(chain_state.EmptyRosterError):
        chain_state.build_payload(
            netuid=68, block=1, epoch_length=361, nanobody_target="P20809",
            miner_uid=8, neurons=[], now=1.0,
        )


def test_zero_epoch_length_is_rejected_before_dividing_by_it():
    with pytest.raises(ValueError):
        chain_state.build_payload(
            netuid=68, block=1, epoch_length=0, nanobody_target="P20809",
            miner_uid=8, neurons=[{"uid": 0}], now=1.0,
        )


def test_write_is_atomic_and_leaves_no_temp_behind(tmp_path):
    target = tmp_path / "nested" / "chain_state.json"
    chain_state.write_atomic(target, {"block": 1})
    assert json.loads(target.read_text()) == {"block": 1}
    assert list(target.parent.glob("*.tmp")) == []


def test_a_failed_write_keeps_the_previous_file(tmp_path):
    target = tmp_path / "chain_state.json"
    chain_state.write_atomic(target, {"block": 1})

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        chain_state.write_atomic(target, {"block": Unserialisable()})
    # The reader still sees the last good view, and no debris is left.
    assert json.loads(target.read_text()) == {"block": 1}
    assert list(target.parent.glob("*.tmp")) == []


def test_the_publisher_accepts_exactly_what_the_miner_writes(tmp_path, monkeypatch):
    """The seam between the two processes, exercised end to end.

    Each side was written against a description of the other; this asserts they
    actually agree, since a silent mismatch here is indistinguishable from being
    deregistered — the failure that had us mining into another operator's slot.
    """
    import time as _time

    publisher = pytest.importorskip("tools.publish_current_nanobody")

    payload = chain_state.build_payload(
        netuid=68,
        block=9048104,
        epoch_length=361,
        nanobody_target="P20809",
        miner_uid=8,
        neurons=chain_state.neuron_rows(
            [HOTKEY_A, HOTKEY_B, HOTKEY_RIVAL],
            [COLDKEY_OURS, COLDKEY_OURS, COLDKEY_RIVAL],
        ),
        now=_time.time(),
    )
    target = tmp_path / "chain_state.json"
    chain_state.write_atomic(target, payload)

    view = publisher._chain_view(target, netuid=68, max_age=600.0)
    assert view.epoch == 25064
    assert view.block == 9048104
    assert len(view.neurons) == 3

    # ... and the fleet resolved from it holds only our own two uids.
    crew = publisher.fleet_module.discover_fleet(view.neurons, COLDKEY_OURS, netuid=68)
    assert sorted(member.uid for member in crew) == [0, 1]


def _miner_tree():
    """Parse the miner without importing it — bittensor is not installed here."""
    import ast

    source = (Path(__file__).resolve().parents[1] / "neurons" / "miner" / "miner.py")
    return ast.parse(source.read_text()), source.read_text()


def test_miner_uses_the_async_substrate_stack():
    """substrate-interface and py-scale-codec own the same `scalecodec` namespace
    as async-substrate-interface, so the chain env cannot hold both. The synchronous
    package is the one that must not come back."""
    _, source = _miner_tree()
    assert "from async_substrate_interface.sync_substrate import SubstrateInterface" in source
    assert "from substrateinterface import" not in source


def test_miner_resolves_the_endpoint_rather_than_passing_an_alias():
    """async-substrate-interface 2.x needs a real URI; `finney` only fails at the
    first query, long after startup has reported success."""
    _, source = _miner_tree()
    assert "SubstrateInterface(url=_resolved_chain_endpoint(config))" in source
    assert "SubstrateInterface(url=config.network)" not in source


def test_miner_still_calls_the_exporter():
    """A passing test of chain_state.py cannot prove the miner invokes it."""
    _, source = _miner_tree()
    assert "from nanobody_opt import chain_state" in source
    assert source.count("_publish_chain_state(state, current_block)") >= 2
    assert "await metagraph.sync()" in source
