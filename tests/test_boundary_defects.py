"""Defects that a unit test passes and the boundary fails.

Every failure this file pins presented in production as a healthy miner: the
value under test was fine in isolation and wrong at the seam -- a call site that
skipped the validator, a deployment that pinned the constant the code had just
learned to compute, a guard that read an absent input as fact.

Nothing here imports ``bittensor`` or touches the network.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from nanobody_opt import fleet as fleet_module
from nanobody_opt import local_neighborhood as ln
from nanobody_opt import run_live_producer as rlp
from nanobody_opt import slot_manager as sm

REPOSITORY = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. The expiring lane ceiling, still pinned by the thing that actually runs.
# ---------------------------------------------------------------------------


def _producer_arguments() -> list[str]:
    """The argument list the deploy hands to ``live-producer``, as data."""
    plays = yaml.safe_load((REPOSITORY / "ansible" / "nanobody.yml").read_text())
    for play in plays:
        for task in play.get("tasks", []) or []:
            facts = task.get("ansible.builtin.set_fact") or {}
            if "nanobody_producer_arguments" in facts:
                return [str(item) for item in facts["nanobody_producer_arguments"]]
    raise AssertionError("the playbook no longer declares the producer arguments")


def test_the_deploy_does_not_pin_the_lane_ceiling_it_was_taught_to_compute():
    """The deployed producer must let the generator scale the supported-single lane.

    In production: the ceiling was moved off a fixed 8192 and onto twenty times
    the proposal budget, and nothing changed, because the launcher the deploy
    builds still passes ``--maximum-exhaustive-singles 8192`` explicitly and an
    explicit value wins.  The generator then keeps raising
    ``supported single-substitution lane exceeds --maximum-exhaustive-singles``
    every iteration -- SystemExit inside a tmux pane, 0% GPU, an epoch producing
    nothing while ``pgrep`` still shows a live producer.  This is the same shape
    as the dotenv failure: the fix landed somewhere the runtime never reads.
    """
    arguments = _producer_arguments()
    assert "--maximum-exhaustive-singles" not in arguments, (
        "the deploy pins the ceiling the generator now scales; remove the flag "
        "so EXHAUSTIVE_SINGLES_BUDGET_MULTIPLE x --proposal-limit governs"
    )


def test_no_inventory_variable_reinstates_the_expired_ceiling():
    """A var left behind is an invitation to pass the flag again next week."""
    offenders = [
        str(path.relative_to(REPOSITORY))
        for path in sorted((REPOSITORY / "ansible").rglob("*.yml"))
        if "nanobody_maximum_exhaustive_singles" in path.read_text()
    ]
    assert offenders == [], f"stale lane-ceiling variable still defined in {offenders}"


# ---------------------------------------------------------------------------
# 2. The ceiling helper stopped validating what the argument parser used to.
# ---------------------------------------------------------------------------


def test_a_non_positive_lane_ceiling_is_refused_rather_than_obeyed():
    """A zero or negative ceiling must be a startup error, not a per-iteration one.

    ``--maximum-exhaustive-singles`` used to sit in the positivity loop that
    calls ``parser.error``.  Making the default ``None`` moved the value out of
    that loop, and ``exhaustive_singles_ceiling`` returns whatever it is handed.
    So ``--maximum-exhaustive-singles 0`` now starts cleanly and then raises
    SystemExit on the first supported single of every iteration for the rest of
    the epoch -- precisely the silent-death signature the scaling default was
    written to remove, reachable again from a typo in an inventory variable.
    """
    with pytest.raises(ValueError):
        ln.exhaustive_singles_ceiling(0, 5000)
    with pytest.raises(ValueError):
        ln.exhaustive_singles_ceiling(-1, 5000)


def test_the_scaled_default_tracks_the_proposal_budget():
    """An explicit ceiling still wins; an absent one scales and never expires."""
    assert ln.exhaustive_singles_ceiling(None, 5000) == (
        ln.EXHAUSTIVE_SINGLES_BUDGET_MULTIPLE * 5000
    )
    assert ln.exhaustive_singles_ceiling(None, 1) == 160
    assert ln.exhaustive_singles_ceiling(8192, 5000) == 8192


# ---------------------------------------------------------------------------
# 3. The producer's mid-iteration re-read skipped its own staleness validator.
# ---------------------------------------------------------------------------


def _write_chain_state(path: Path, *, epoch: int, offset: int, age: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 2,
        "netuid": 68,
        "epoch": epoch,
        "epoch_length": 361,
        "block": epoch * 361 + offset,
        "nanobody_target": "P20809",
        "miner_uid": 8,
        "neurons": [{"uid": 8, "hotkey": "5H" + "k" * 45, "coldkey": "5C" + "c" * 45}],
        "updated_at_unix": time.time() - age,
    }))


def test_a_dead_miners_file_cannot_decide_the_closing_generation(tmp_path, monkeypatch):
    """The closing-window decision must come from a chain view still being written.

    ``_chain_state`` refuses a snapshot older than ``--maximum-chain-state-age``,
    but the re-read partway through an iteration went straight to
    ``json.loads(Path(args.chain_state).read_text())`` and so accepted any file
    at all as long as its ``epoch`` field matched.  If the miner process dies --
    a websocket reset, an OOM, a hot reload that did not come back -- its last
    file keeps the right epoch for up to 72 minutes while its block goes stale.

    In production: the producer reads a block from early in the epoch, computes
    ``closing=False``, passes ``--replication-rounds 0``, and the epoch ends with
    one observation per candidate instead of the twenty-plus that strict
    qualification needs.  Every log line says PRODUCER_GENERATION, the GPUs are
    busy, and the publisher quietly ships provisional every epoch.
    """
    runtime = tmp_path / "root"
    chain = tmp_path / "chain_state.json"
    epoch = 25034
    # The live view main() validated: deep inside the closing window.
    fresh = {
        "epoch": epoch,
        "epoch_length": 361,
        "block": epoch * 361 + 300,
        "nanobody_target": "P20809",
        "updated_at_unix": time.time(),
    }
    # What the dead miner left on disk: same epoch, an hour old, early block.
    _write_chain_state(chain, epoch=epoch, offset=10, age=3600.0)

    args = rlp._parser().parse_args([
        "--runtime-root", str(runtime),
        "--chain-state", str(chain),
        "--ledger", str(tmp_path / "ledger.txt"),
        "--mutation-database", str(tmp_path / "mutations.json"),
        "--replication-rounds", "22",
        "--replication-reserve-blocks", "120",
    ])

    def fake_download(url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "config.yaml":
            path.write_text(yaml.safe_dump(
                {"protein_selection": {"nanobody": {"target": "P20809"}}}
            ))
        else:
            path.write_text("sequence\n" + "A" * 120 + "\n")

    commands: list[list[str]] = []
    monkeypatch.setattr(rlp, "_download", fake_download)
    monkeypatch.setattr(rlp, "_run", lambda command: commands.append(list(command)))

    rlp._run_iteration(args=args, state=fresh, iteration=1)

    race = [item for item in commands if "nanobody_opt.sequential_exact_race" in item]
    assert len(race) == 1
    rounds = race[0][race[0].index("--replication-rounds") + 1]
    assert rounds == "22", (
        "a stale chain-state file overrode the validated snapshot and turned the "
        "closing generation into a non-closing one"
    )


# ---------------------------------------------------------------------------
# 4. A metagraph that came back empty must not be read as "we hold nothing".
# ---------------------------------------------------------------------------


class _Metagraph:
    def __init__(self, coldkeys, registration_blocks):
        self.coldkeys = coldkeys
        self.block_at_registration = registration_blocks


class _Subtensor:
    """Exactly what read_chain_state touches."""

    def __init__(self, block, recycle, regs, coldkeys, registration_blocks):
        self._block = block
        self._recycle = recycle
        self._regs = regs
        self._metagraph = _Metagraph(coldkeys, registration_blocks)

    def get_current_block(self):
        return self._block

    def recycle(self, netuid):
        return type("Balance", (), {"tao": self._recycle})()

    def query_subtensor(self, name, params=None):
        return type("Scale", (), {"value": self._regs})()

    def metagraph(self, netuid):
        return self._metagraph


OURS = "5Cour" + "c" * 44


def test_an_empty_metagraph_is_a_failed_read_not_an_empty_fleet():
    """Subnet 68 is full, so zero neurons means the read failed, not that we lost them.

    ``chain_state.build_payload`` already refuses to write an empty roster
    because "an empty roster and a real deregistration are indistinguishable
    downstream".  ``read_chain_state`` -- the one that decides whether to spend
    tau -- draws no such distinction: an empty ``coldkeys`` column yields
    ``registered_uids == ()``, ``registered == 0``, and 0 is below
    ``urgent_below``, so ``decide`` returns ``buy`` with
    ``below_target_and_short_of_a_fleet`` and skips the price floor entirely.

    In production, under ``--execute-registrations --interval``: one flaky
    metagraph call while nine healthy hotkeys are registered burns a
    registration at whatever the price happens to be, logs ``SLOT_BUY
    registered=0/9`` as though it were correct, and repeats next interval until
    the daily budget stops it -- roughly seven buys and about one tau against a
    fleet that never needed a tenth slot.
    """
    blind = _Subtensor(
        block=1_000_500, recycle=0.131, regs=3, coldkeys=[], registration_blocks=[]
    )
    with pytest.raises(sm.EmptyRosterError):
        sm.read_chain_state(blind, 68, OURS)


def test_a_truncated_coldkey_flag_is_refused_before_it_becomes_a_purchase():
    """An elided address must be a configuration error, not a reason to buy.

    ``nanobody_opt.fleet`` refuses ``"5EkCwcNTfxeN..."`` because a partial
    address matches no neuron row and gets reported as an empty fleet.  The slot
    manager accepted the same string from ``--coldkey-ss58`` and fed it straight
    into an equality test against every coldkey on the subnet.

    In production, under ``--execute-registrations``: nothing matches, so
    ``registered`` is 0 against a target of 9, which is below ``urgent_below``,
    so the price floor is skipped and the agent burns a registration an interval
    -- against a fleet that was fully held the whole time -- until the daily
    budget stops it.  Every log line reads ``SLOT_BUY reason=
    below_target_and_short_of_a_fleet registered=0/9``, which looks exactly like
    a correct cold start.
    """
    for truncated in ("5EkCwcNTfxeN...", "5EkCwcNTfxeN…", "5Eshort"):
        with pytest.raises(SystemExit):
            sm._resolve_coldkey_ss58(None, "mining_ops", truncated)
    # A full address is passed through without the wallet being touched at all.
    assert sm._resolve_coldkey_ss58(None, "mining_ops", OURS) == OURS


def test_a_full_field_that_holds_none_of_ours_is_still_an_ordinary_zero():
    """Deregistration must stay distinguishable from a broken read, and buyable."""
    field = _Subtensor(
        block=1_000_500,
        recycle=0.131,
        regs=1,
        coldkeys=["5Fother", "5Gother", "5Hother"],
        registration_blocks=[900_000, 900_000, 900_000],
    )
    state = sm.read_chain_state(field, 68, OURS)
    assert state.registered_uids == ()
    assert state.registered == 0


# ---------------------------------------------------------------------------
# 5. A neuron row with no coldkey is not a neuron row belonging to someone else.
# ---------------------------------------------------------------------------


def test_a_row_with_no_coldkey_is_refused_instead_of_becoming_the_string_None():
    """A short coldkeys column must not silently shrink the fleet we publish from.

    ``chain_state.neuron_rows`` writes ``"coldkey": None`` when the metagraph's
    coldkeys column is shorter than its hotkeys column, and says so explicitly
    so "the consumer can tell the two apart".  The consumer does not: it does
    ``str(item["coldkey"]).strip()``, so the absence becomes the literal string
    ``"None"``, which matches no coldkey and simply drops the uid.

    Two production shapes, both quiet.  A partially populated column means the
    publisher discovers four uids out of nine, prints ``fleet: 4 registered
    uid(s)`` and a FLEET_SHORTFALL line, and leaves five paid-for slots
    committing nothing -- about tau0.032 of expected value per slot per epoch,
    twenty epochs a day.  A wholly absent column raises ``EmptyFleetError``,
    whose banner says every hotkey is deregistered and tells the operator to
    register more -- so the response to a bad metagraph read is to buy
    registrations we already own.  Note the metagraph branch of the same
    function already fails closed on a column-length mismatch.
    """
    rows = [
        {"uid": 0, "hotkey": "5Hone" + "a" * 44, "coldkey": OURS},
        {"uid": 1, "hotkey": "5Htwo" + "b" * 44, "coldkey": None},
    ]
    with pytest.raises(ValueError):
        fleet_module.neuron_rows(rows)


def test_the_publisher_refuses_a_chain_state_whose_coldkeys_went_missing():
    """End to end across the seam the miner and the publisher share."""
    from nanobody_opt import chain_state

    rows = chain_state.neuron_rows(
        ["5H" + "a" * 45, "5H" + "b" * 45],
        [OURS],  # a coldkeys column one short of the hotkeys column
    )
    assert rows[1]["coldkey"] is None
    with pytest.raises(ValueError):
        fleet_module.discover_fleet(rows, OURS, netuid=68)
