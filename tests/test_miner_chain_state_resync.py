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
    boundary = source.index('if state.get("last_challenge_epoch") != epoch_now:')
    assert resync < boundary, "re-sync must run before the challenge refresh"


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



# --- the challenge refresh fires once per epoch -------------------------------

def test_no_exact_block_match_remains_in_the_loop():
    """`block % epoch_length == 0` holds for every ~1.3 s poll while the
    boundary block is the chain head (~6 polls), and for none when a poll
    steps over it.  Both the re-sync and the challenge refresh must key on
    the epoch NUMBER changing instead."""
    source = SOURCE.read_text()
    loop = source[source.index("# 5) Main epoch-based loop"):]
    assert "if current_block % epoch_length == 0" not in loop
    assert 'state["last_challenge_epoch"] = epoch_now' in loop


def _load_miner(source_path=SOURCE):
    import importlib.util
    import sys
    import pytest
    root = str(SOURCE.parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("miner_under_test", source_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # bittensor and friends live in the chain env
        pytest.skip(f"miner module needs its runtime deps: {error}")
    return module


def simulate_boundary_polls(module, blocks, epoch_length=361):
    """Drive run_miner's real loop over a scripted chain-head sequence.

    Returns (challenge_epochs, search_launches, metagraph_syncs): the epoch
    of every challenge refresh, the epoch of every search launch, and the
    number of metagraph syncs.
    """
    import asyncio
    from types import SimpleNamespace

    heads = list(blocks)
    challenges, launches, syncs = [], [], []

    class Subtensor:
        async def get_current_block(self):
            if not heads:
                raise KeyboardInterrupt  # ends run_miner's loop cleanly
            return heads.pop(0)

        async def determine_block_hash(self, block):
            assert block % epoch_length == 0, f"challenge read off-boundary {block}"
            return f"0x{block:x}"

    class Metagraph:
        async def sync(self):
            syncs.append(1)

    async def setup(_config):
        return object(), Subtensor(), Metagraph(), 6, epoch_length

    def challenge(block_hash, **_):
        return {"block_hash": block_hash, "small_molecule_target": ["T"],
                "allowed_reaction": "rxn:1"}

    async def schedule(state, params, blocks_remaining):
        launches.append(params["epoch"])

    async def submit(_state):
        return True

    async def no_sleep(_seconds):
        return None

    module.setup_bittensor_objects = setup
    module._publish_chain_state = lambda state, block: None  # never touch a real file
    module.os.environ.pop("NOVA_SKIP_INITIAL_SEARCH", None)
    module.get_challenge_params_from_blockhash = challenge
    module.schedule_compound_search = schedule
    module._submit_current_candidates = submit
    module.dummy_get_candidate_products = lambda params: ("~", "~")
    module.load_github_path = lambda: "x"
    module.QuicknetBittensorDrandTimelock = lambda: None
    real_sleep = module.asyncio.sleep
    module.asyncio.sleep = no_sleep
    original_info = module.bt.logging.info

    def info(message, *args, **kwargs):
        if "Found epoch boundary" in str(message):
            challenges.append(str(message))
        return None

    module.bt.logging.info = info
    try:
        config = SimpleNamespace(small_molecule_target="T", nanobody_target="P",
                                 random_valid_reaction=True, network="test")
        asyncio.run(module.run_miner(config))
    finally:
        module.asyncio.sleep = real_sleep
        module.bt.logging.info = original_info
    return challenges, launches, syncs


def _boundary_scenario(epoch_length=361):
    start = 10 * epoch_length - 100              # mid-epoch 9
    b1, b2 = 10 * epoch_length, 11 * epoch_length
    return (
        [start, start]                            # startup + initial schedule
        + [b1 - 1] * 6                            # last block of epoch 9
        + [b1] * 6                                # ~6 polls on the boundary block
        + [b1 + 1] * 6
        + [b2 - 1] * 3
        + [b2 + 2] * 6                            # boundary b2 stepped over
    )


def test_challenge_refresh_fires_once_per_epoch():
    module = _load_miner()
    challenges, launches, syncs = simulate_boundary_polls(module, _boundary_scenario())
    # initial launch (epoch 9), then exactly one per boundary: 10 and 11,
    # including 11 whose boundary block was never observed.
    assert launches == [9, 10, 11], launches
    assert len(challenges) == 2, challenges
    assert len(syncs) == 2, syncs


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
