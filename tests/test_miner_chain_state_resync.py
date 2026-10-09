"""The miner's roster export must not freeze at process start."""
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "neurons" / "miner" / "miner.py"


def test_resync_is_driven_by_epoch_change_not_an_exact_block_match():
    """`block % epoch_length == 0` requires landing on one specific block.

    Blocks arrive every ~12s and the loop does chain round-trips, so it
    routinely advances by more than one block per iteration and steps over
    the boundary. Measured on the live miner: zero "Found epoch boundary"
    lines in 5.6 days, so metagraph.sync ran once at start and never again,
    and the publisher kept assigning sequences to a uid that had been
    re-registered to another coldkey.
    """
    source = SOURCE.read_text()
    assert "epoch_now = current_block // epoch_length" in source
    assert 'state.get("last_epoch_synced") != epoch_now' in source
    # the re-sync must not sit behind the exact-match branch
    resync = source.index("epoch_now = current_block // epoch_length")
    boundary = source.index("if current_block % epoch_length == 0:")
    assert resync < boundary, "re-sync still gated behind the exact block match"


def test_chain_state_is_published_after_the_resync():
    """Exporting before the sync would publish the stale roster one last
    time under a fresh block number, which is the bug itself."""
    source = SOURCE.read_text()
    resync = source.index('state["last_epoch_synced"] = epoch_now')
    publish = source.index("_publish_chain_state(state, current_block)", resync)
    assert publish > resync


def test_a_failed_sync_does_not_end_the_epoch():
    source = SOURCE.read_text()
    block = source[source.index("epoch_now = current_block // epoch_length"):][:1200]
    assert "except Exception" in block
    assert "last_epoch_synced" in block


def test_publish_is_forced_after_resync():
    """_publish_chain_state early-returns when the block is unchanged, so the
    cached block marker has to be cleared or the fresh roster never lands."""
    source = SOURCE.read_text()
    block = source[source.index('state["last_epoch_synced"] = epoch_now'):][:600]
    assert 'state["last_chain_state_block"] = None' in block


# --- roster freshness is recorded and enforced --------------------------------

def test_payload_records_when_the_roster_was_synced():
    """`block` advances every loop; the roster only moves when sync succeeds.
    Conflating them is what made a five-day outage invisible."""
    from nanobody_opt import chain_state
    payload = chain_state.build_payload(
        netuid=68, block=9116479, epoch_length=361, nanobody_target="P20809",
        miner_uid=6, neurons=[{"uid": 6, "hotkey": "hk", "coldkey": "ck"}],
        now=1.0, roster_block=9000000)
    assert payload["roster_block"] == 9000000
    assert payload["block"] == 9116479


def test_payload_defaults_roster_block_to_the_current_block():
    """At startup the roster really is as fresh as the block."""
    from nanobody_opt import chain_state
    payload = chain_state.build_payload(
        netuid=68, block=123456, epoch_length=361, nanobody_target="P20809",
        miner_uid=6, neurons=[{"uid": 6, "hotkey": "hk", "coldkey": "ck"}],
        now=1.0)
    assert payload["roster_block"] == 123456


def test_export_carries_the_synced_roster_block():
    """Without this argument build_payload falls back to the current block,
    so the exported roster age is 0 on every write and no downstream check
    can ever see a frozen roster."""
    source = SOURCE.read_text()
    publish = source[source.index("def _publish_chain_state"):][:2000]
    assert 'roster_block=state.get("roster_block")' in publish



def test_publisher_refuses_a_frozen_roster(tmp_path):
    """Ported with the check from 2e299e6: a file that is fresh by every
    other check, whose neuron rows are far older than two epochs."""
    import json, time, importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "pub_mod", Path(__file__).resolve().parent.parent
        / "tools" / "publish_current_nanobody.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pub_mod"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        import pytest
        pytest.skip("publisher module needs its runtime deps")
    state = tmp_path / "chain_state.json"
    base = {"schema_version": 2, "netuid": 68, "block": 9116479,
            "epoch_length": 361, "epoch": 9116479 // 361,
            "nanobody_target": "P20809", "miner_uid": 6,
            "neurons": [{"uid": 6, "hotkey": "hk", "coldkey": "ck"}],
            "updated_at_unix": time.time()}

    state.write_text(json.dumps({**base, "roster_block": 9116470}))
    mod._chain_view(state, 68, 600)          # fresh roster: accepted

    limit = mod.MAX_ROSTER_AGE_BLOCKS
    state.write_text(json.dumps({**base, "roster_block": 9116479 - limit}))
    mod._chain_view(state, 68, 600)          # exactly at the limit: accepted

    for stale in (9116479 - limit - 1, 9000000):
        state.write_text(json.dumps({**base, "roster_block": stale}))
        try:
            mod._chain_view(state, 68, 600)
        except RuntimeError as error:
            assert "roster is stale" in str(error)
        else:
            raise AssertionError(f"published against a roster synced at {stale}")

    state.write_text(json.dumps(base))       # no roster_block at all
    try:
        mod._chain_view(state, 68, 600)
    except RuntimeError as error:
        assert "roster_block" in str(error)
    else:
        raise AssertionError("accepted a payload with no roster age at all")
