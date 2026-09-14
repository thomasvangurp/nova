"""Slot policy against injected price and registration series -- no network.

The registration call itself is never reached here: every test drives
:func:`nanobody_opt.slot_manager.decide` and the log helpers, which is the whole
point of keeping the policy free of I/O.
"""
from __future__ import annotations

import ast
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt import slot_manager as sm

REPO = Path(__file__).resolve().parents[1]


def decaying(start_block: int, blocks: int, price: float, decay: float | None = None):
    """A pure-decay price series, the regime we normally buy in."""
    decay = sm.DECAY_PER_BLOCK if decay is None else decay
    return [
        sm.PriceSample(start_block + offset, price * decay**offset)
        for offset in range(blocks)
    ]


#: Enough held hotkeys that the urgency path is not in play, so a test about the
#: decay floor is actually testing the decay floor.
HOLDING_A_FLEET = tuple(range(5))


def state_at(history, *, regs=0, registered=HOLDING_A_FLEET, last=None):
    latest = history[-1]
    return sm.ChainState(
        block=latest.block,
        recycle_tao=latest.recycle_tao,
        registrations_this_interval=regs,
        registered_uids=tuple(registered),
        last_registration_block=last,
    )


# ---------------------------------------------------------------------------
# The measurement the policy is built on.
# ---------------------------------------------------------------------------


def test_the_decay_constant_reproduces_the_polled_measurement():
    """tau0.135167 -> tau0.130312 over nineteen consecutive blocks."""
    predicted = 0.135167 * sm.DECAY_PER_BLOCK**19
    assert abs(predicted - 0.130312) < 1e-5
    # ...which is the same statement as a halving in one adjustment interval.
    assert abs(sm.DECAY_PER_BLOCK**sm.ADJUSTMENT_INTERVAL - 0.5) < 1e-12
    # ...and -0.19% per block, as observed.
    assert 0.0019 < 1.0 - sm.DECAY_PER_BLOCK < 0.0020


def test_waiting_pays_at_current_prices_and_stops_paying_near_tau0_046():
    economics = sm.wait_economics(0.13, sm.SlotPolicy())
    assert economics["wait_value_ratio"] > 2.5
    assert 0.045 < economics["breakeven_recycle_tao"] < 0.048
    assert sm.wait_economics(0.03, sm.SlotPolicy())["wait_value_ratio"] < 1.0


# ---------------------------------------------------------------------------
# Buy when below target, at the floor, with a quiet interval.
# ---------------------------------------------------------------------------


def test_pure_decay_below_target_buys():
    history = decaying(1_000_000, 61, 0.135167)
    decision = sm.decide(
        state_at(history, regs=0, registered=(7, 12, 19, 24, 31)),
        history,
        [],
        sm.SlotPolicy(),
    )
    assert decision.action == sm.BUY
    assert decision.reason == sm.REASON_BUY
    assert decision.detail["shortfall"] == sm.SlotPolicy().target - 5
    # Under pure decay every sample projects to the same price, so the live
    # price sits exactly on its own floor.
    assert decision.detail["floor_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_one_competitor_registration_in_the_interval_is_still_a_buy():
    """The gate is <= 1, not == 0: a single buy per interval is the target rate."""
    history = decaying(1_000_000, 61, 0.12)
    assert sm.decide(state_at(history, regs=1), history, [], sm.SlotPolicy()).action == sm.BUY


def test_a_busy_interval_does_not_block_a_cheap_buy():
    """The competitor counter is not a price, and gating on it never buys.

    RegistrationsThisInterval does not clear until the interval turns over, and
    this field routinely registers three times an interval -- so "wait until the
    field is quiet" spent its life refusing prices less than half the ceiling.
    What matters is the price in the same snapshot.
    """
    history = decaying(1_000_000, 61, 0.12)
    decision = sm.decide(state_at(history, regs=5), history, [], sm.SlotPolicy())
    assert decision.action == sm.BUY
    assert decision.detail["registrations_this_interval"] == 5


def test_an_expensive_interval_still_refuses():
    """Removing the proxy must not remove the protection it stood in for."""
    history = decaying(1_000_000, 61, 0.42)
    decision = sm.decide(state_at(history, regs=5), history, [], sm.SlotPolicy())
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_CEILING


def test_below_a_working_fleet_an_affordable_price_beats_a_perfect_one():
    """Holding nothing, a spike we would normally sit out is still worth buying.

    The floor saves about tau0.03; each missing ticket forfeits ~tau0.032 of
    expected value every epoch, roughly twenty times a day.
    """
    history = decaying(1_000_000, 61, 0.10)
    spiked = list(history) + [sm.PriceSample(history[-1].block + 1, 0.15)]
    decision = sm.decide(
        state_at(spiked, registered=()), spiked, [], sm.SlotPolicy()
    )
    assert decision.action == sm.BUY
    assert decision.reason == sm.REASON_BUY_URGENT
    # ...and the same spike is declined once the fleet is nearly complete.
    patient = sm.decide(
        state_at(spiked, registered=tuple(range(sm.SlotPolicy().target - 1))),
        spiked, [], sm.SlotPolicy()
    )
    assert patient.action == sm.WAIT
    assert patient.reason == sm.REASON_ABOVE_FLOOR


# ---------------------------------------------------------------------------
# Competitor-driven spike.
# ---------------------------------------------------------------------------


def spiked(start_block: int, spike_at: int, blocks: int, price: float, jump: float = 1.33):
    """Pure decay, one +33% registration jump, then decay again."""
    series = []
    for offset in range(blocks):
        value = price * sm.DECAY_PER_BLOCK**offset
        if offset >= spike_at:
            value *= jump
        series.append(sm.PriceSample(start_block + offset, value))
    return series


def test_a_competitor_spike_blocks_the_buy():
    history = spiked(1_000_000, 40, 61, 0.135167)
    decision = sm.decide(state_at(history, regs=1), history, [], sm.SlotPolicy())
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_ABOVE_FLOOR
    # The pre-spike samples are what expose it: the live price is a third above
    # where pure decay would have put it.
    assert decision.detail["floor_ratio"] == pytest.approx(1.33, abs=1e-6)


def test_a_spike_delays_us_but_does_not_lock_us_out():
    """The lookback window is finite on purpose, so the floor re-anchors."""
    policy = sm.SlotPolicy(lookback_blocks=180)
    history = spiked(1_000_000, 40, 400, 0.135167)
    blocked = sm.decide(
        state_at(history[:200], regs=1), history[:200], [], policy
    )
    assert blocked.reason == sm.REASON_ABOVE_FLOOR
    # Once every pre-spike sample has aged past lookback_blocks, the remaining
    # window is all post-spike and consistent with pure decay again.
    resumed = sm.decide(state_at(history, regs=1), history, [], policy)
    assert resumed.action == sm.BUY
    # And we did not pay the spike to get there: the price fell meanwhile.
    assert resumed.detail["recycle_tao"] < blocked.detail["recycle_tao"]


def test_a_thin_window_refuses_to_guess_at_the_floor():
    history = decaying(1_000_000, 3, 0.12)
    decision = sm.decide(state_at(history), history, [], sm.SlotPolicy())
    assert decision.reason == sm.REASON_NO_HISTORY
    assert decision.detail["projected_floor_tao"] is None
    # Three samples five blocks apart is a poll burst, not a decay window.
    burst = [sm.PriceSample(1_000_000 + n * 2, 0.12) for n in range(3)]
    assert sm.decide(state_at(burst), burst, [], sm.SlotPolicy()).reason == sm.REASON_NO_HISTORY


# ---------------------------------------------------------------------------
# Rails.
# ---------------------------------------------------------------------------


def test_budget_exhaustion_refuses_rather_than_overspends():
    history = decaying(1_000_000, 61, 0.13)
    now = time.time()
    spends = [
        sm.SpendRecord(now - 3600 * hour, 0.19) for hour in range(1, 6)
    ]  # tau0.95 inside the day
    policy = sm.SlotPolicy(max_spend_tao_per_day=1.0)
    decision = sm.decide(state_at(history), history, spends, policy, now=now)
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_BUDGET
    assert decision.detail["spent_tao_last_day"] == pytest.approx(0.95)
    # Spend older than a day has rolled off, so the same series buys tomorrow.
    stale = [sm.SpendRecord(now - 86400 - 60, 0.19) for _ in range(5)]
    assert sm.decide(state_at(history), history, stale, policy, now=now).action == sm.BUY


def test_the_recycle_ceiling_refuses_an_expensive_floor():
    """A price can be honestly at its floor and still be too expensive."""
    history = decaying(1_000_000, 61, 0.9)
    policy = sm.SlotPolicy(max_recycle_tao=0.20)
    decision = sm.decide(state_at(history), history, [], policy)
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_CEILING
    assert decision.detail["floor_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_target_already_met_is_a_no_op():
    history = decaying(1_000_000, 61, 0.01)  # free, and still nothing to do
    decision = sm.decide(
        state_at(history, registered=tuple(range(sm.SlotPolicy().target))), history, [], sm.SlotPolicy()
    )
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_TARGET_MET
    assert decision.detail["shortfall"] == 0


def test_one_registration_per_adjustment_interval_at_most():
    history = decaying(1_000_000, 61, 0.12)
    block = history[-1].block
    same_interval = (block // sm.ADJUSTMENT_INTERVAL) * sm.ADJUSTMENT_INTERVAL
    decision = sm.decide(
        state_at(history, last=same_interval),
        history,
        [],
        sm.SlotPolicy(stagger_blocks=0),
    )
    assert decision.reason == sm.REASON_OWN_INTERVAL


def test_the_stagger_window_holds_expiries_apart():
    history = decaying(1_000_000, 61, 0.12)
    block = history[-1].block
    policy = sm.SlotPolicy()
    assert policy.stagger == sm.default_stagger_blocks(policy.target)
    # Probe a block that is inside the stagger window but past the adjustment
    # interval, so the stagger guard is the one that answers. The window between
    # the two narrows as the target rises, so derive the probe rather than
    # assuming a fixed margin.
    assert policy.stagger > sm.ADJUSTMENT_INTERVAL
    probe = (policy.stagger + sm.ADJUSTMENT_INTERVAL) // 2
    inside = sm.decide(
        state_at(history, last=block - probe), history, [], policy
    )
    assert inside.reason == sm.REASON_STAGGER
    outside = sm.decide(
        state_at(history, last=block - (policy.stagger + 100)), history, [], policy
    )
    assert outside.action == sm.BUY


def test_nine_staggered_buys_spread_immunity_expiries_over_the_window():
    spacing = sm.default_stagger_blocks(9)
    expiries = [n * spacing + sm.IMMUNITY_PERIOD for n in range(9)]
    gaps = [b - a for a, b in itertools.pairwise(expiries)]
    assert set(gaps) == {spacing}
    # The whole point: no two renewals land in the same adjustment interval.
    assert spacing > sm.ADJUSTMENT_INTERVAL
    assert max(expiries) - min(expiries) < sm.IMMUNITY_PERIOD


# ---------------------------------------------------------------------------
# The log is the audit trail and the agent's memory.
# ---------------------------------------------------------------------------


def test_the_log_round_trips_history_budget_and_last_purchase(tmp_path):
    log = tmp_path / "slots.jsonl"
    now = time.time()
    sm.append_record(log, {
        "time": now - 10, "action": "wait", "reason": sm.REASON_ABOVE_FLOOR,
        "executed": False, "block": 1_000_000, "recycle_tao": 0.14,
    })
    sm.append_record(log, {
        "time": now - 5, "action": "buy", "reason": sm.REASON_BUY,
        "executed": True, "block": 1_000_050, "recycle_tao": 0.13,
    })
    sm.append_record(log, {  # a dry-run buy: recorded, but not spending
        "time": now, "action": "buy", "reason": sm.REASON_BUY,
        "executed": False, "block": 1_000_100, "recycle_tao": 0.12,
    })
    records = sm.read_records(log)
    assert len(records) == 3
    assert [s.block for s in sm.price_history(records)] == [1_000_000, 1_000_050, 1_000_100]
    assert sm.spend_in_window(sm.executed_spends(records), now) == pytest.approx(0.13)
    assert sm.last_executed_block(records) == 1_000_050
    # Every skip is on the record with the price and counters that justified it.
    skip = json.loads(log.read_text().splitlines()[0])
    assert skip["reason"] == sm.REASON_ABOVE_FLOOR and skip["recycle_tao"] == 0.14


def test_a_torn_final_line_does_not_lose_the_log(tmp_path):
    log = tmp_path / "slots.jsonl"
    log.write_text('{"block": 1, "recycle_tao": 0.1}\n{"block": 2, "recyc\n')
    assert [s.block for s in sm.price_history(sm.read_records(log))] == [1]


def test_hotkey_names_are_allocated_deterministically():
    existing = ["nova68", "nova68pub.txt", "nova68-slot-1", "nova68-slot-2"]
    assert sm.next_hotkey_name(existing, "nova68-slot") == "nova68-slot-3"
    # A retry after a failed registration reuses the same name.
    assert sm.next_hotkey_name(existing, "nova68-slot") == "nova68-slot-3"


# ---------------------------------------------------------------------------
# Structural guarantees.
# ---------------------------------------------------------------------------


def test_the_policy_imports_without_bittensor_or_site_packages():
    """Proof that the decision core carries no I/O: it runs with -S."""
    program = (
        "import sys; sys.path.insert(0, '.'); "
        "from nanobody_opt import slot_manager as s; "
        "print(s.SlotPolicy().stagger)"
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-c", program],
        capture_output=True, text=True, cwd=str(REPO), check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(
        sm.default_stagger_blocks(sm.SlotPolicy().target)
    )


def test_registration_is_never_reachable_without_the_explicit_flag():
    """The only burn call sits in _register, and _register sits under the flag."""
    tree = ast.parse((REPO / "nanobody_opt" / "slot_manager.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    burns = [
        node for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "burned_register"
    ]
    assert len(burns) == 1
    invocations = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "_register"
    ]
    assert len(invocations) == 1
    guarded = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "execute"
        and any(call in ast.walk(node) for call in invocations)
    ]
    assert len(guarded) == 1


def _unlock_code() -> str:
    """The unlock function with comments and docstrings stripped."""
    source = (REPO / "nanobody_opt" / "slot_manager.py").read_text()
    # Every function the password passes through, not just the unlock: the
    # backends are where it is actually fetched, and a guard that stops at
    # _unlock_coldkey would pass while a backend leaked.
    body = "".join(
        source.split(f"def {name}")[1].split("\ndef ")[0]
        for name in (
            "_password_from_keychain",
            "_password_from_systemd_credential",
            "_coldkey_password",
            "_unlock_coldkey",
        )
    )
    lines = [
        line for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    text = "\n".join(lines)
    return text.split('"""')[0] + '"""'.join(text.split('"""')[2:])


def test_no_credential_ever_reaches_a_command_line_or_a_message():
    code = _unlock_code()
    # The Keychain service name is read from an environment variable whose name
    # is all this module ever holds, and never from a flag or a literal.
    assert "os.environ.get(service_env" in code
    # The password is captured on a pipe: never echoed, traced or written down.
    assert "capture_output=True" in code
    # read_text is how the systemd credential is fetched; writing one anywhere
    # is still forbidden, as is any shell that could echo it.
    for forbidden in ("echo", "printf", "shell=True", "-p ", "open(", "write("):
        assert forbidden not in code, forbidden
    # It is interpolated into nothing, and nothing is logged from this path.
    assert "{password" not in code
    assert "print(" not in code
    # The original exception is dropped so a keyfile library cannot leak the
    # attempted password into a traceback, and the reference is released.
    assert "from None" in code
    assert "del password" in code


def test_an_unset_keychain_service_fails_cleanly_with_no_fallback():
    source = (REPO / "nanobody_opt" / "slot_manager.py").read_text()
    # No hardcoded service name, no interactive prompt, no Keychain search.
    for forbidden in ("getpass", "input(", "find-generic-password -a", "-D "):
        assert forbidden not in source, forbidden
    assert source.count("find-generic-password") == 1
    code = _unlock_code()
    # Missing variable -> explanatory exit naming only the variable, in both stores.
    assert "raise SystemExit(" in code.split("if not service:")[1]
    assert "raise SystemExit(" in code.split("if not name:")[1]
    # The systemd credential is never substituted by an environment variable:
    # LoadCredentialEncrypted exists precisely so the value stays out of
    # `systemctl show` and /proc/<pid>/environ.
    assert "CREDENTIALS_DIRECTORY" in code
    assert "os.environ.get(name_env" in code


# ---------------------------------------------------------------------------
# Interface contract the deployment playbooks are written against.
# ---------------------------------------------------------------------------


def _cli(*argv: str):
    environment = {
        key: value for key, value in os.environ.items()
        if key != "NOVA_COLDKEY_KEYCHAIN_SERVICE"
    }
    return subprocess.run(
        [sys.executable, "-m", "nanobody_opt.slot_manager", *argv],
        capture_output=True, text=True, cwd=str(REPO), env=environment, check=False,
    )


def test_help_works_with_no_keychain_wallet_or_chain():
    """The deploy gate runs this before stopping anything; bittensor is absent."""
    completed = _cli("--help")
    assert completed.returncode == 0, completed.stderr
    for flag in ("--minimum-hotkeys", "--interval", "--execute-registrations",
                 "--max-recycle-tao", "--max-spend-tao-per-day", "--decision-log"):
        assert flag in completed.stdout, flag
    assert f"default: {sm.SlotPolicy().target}" in completed.stdout
    # Nothing was imported, unlocked or connected to in order to answer it.
    assert "bittensor" not in completed.stderr


def test_the_module_is_runnable_as_python_dash_m():
    completed = _cli("--minimum-hotkeys", "11", "--help")
    assert completed.returncode == 0, completed.stderr


def test_live_mode_refuses_before_touching_the_chain_when_the_service_is_unset():
    completed = _cli("--execute-registrations")
    assert completed.returncode != 0
    message = completed.stdout + completed.stderr
    assert "NOVA_COLDKEY_KEYCHAIN_SERVICE" in message
    # It failed on the missing variable, not on a chain connection.
    assert "bittensor is not importable" not in message


def test_dry_run_is_the_default_in_the_parsed_namespace():
    """No argument can switch spending on by default, including a future edit."""
    tree = ast.parse((REPO / "nanobody_opt" / "slot_manager.py").read_text())
    flags = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and getattr(node.args[0], "value", None) == "--execute-registrations"
    ]
    assert len(flags) == 1
    keywords = {keyword.arg: keyword for keyword in flags[0].keywords}
    assert keywords["action"].value.value == "store_true"
    assert keywords["dest"].value.value == "execute"
    # store_true defaults to False and nothing may override it.
    assert "default" not in keywords


def test_holding_nothing_on_chain_is_the_ordinary_starting_case():
    """nova68 and nova68-v2 are both deregistered, so reconcile finds zero uids."""
    history = decaying(1_000_000, 61, 0.13)
    decision = sm.decide(
        state_at(history, regs=0, registered=(), last=None), history, [], sm.SlotPolicy()
    )
    assert decision.action == sm.BUY
    assert decision.detail["registered"] == 0
    assert decision.detail["shortfall"] == sm.SlotPolicy().target
    assert decision.detail["blocks_since_our_last_registration"] is None


# ---------------------------------------------------------------------------
# Reconciliation, against an injected chain rather than a network.
# ---------------------------------------------------------------------------


class FakeMetagraph:
    def __init__(self, coldkeys, registration_blocks):
        self.coldkeys = coldkeys
        self.block_at_registration = registration_blocks


class FakeSubtensor:
    """Everything read_chain_state touches, and nothing else."""

    def __init__(self, block, recycle, regs, coldkeys, registration_blocks):
        self._block = block
        self._recycle = recycle
        self._regs = regs
        self._metagraph = FakeMetagraph(coldkeys, registration_blocks)

    def get_current_block(self):
        return self._block

    def recycle(self, netuid):
        assert netuid == 68
        return type("Balance", (), {"tao": self._recycle})()

    def query_subtensor(self, name, params=None):
        assert name == "RegistrationsThisInterval"
        return type("Scale", (), {"value": self._regs})()

    def metagraph(self, netuid):
        assert netuid == 68
        return self._metagraph


OURS = "5EkCwcNTfxeNTESTONLYnotarealaddress"


def test_reconcile_counts_only_our_uids_that_are_actually_on_chain():
    chain = FakeSubtensor(
        block=1_000_500,
        recycle=0.131,
        regs=1,
        coldkeys=["5Fother", OURS, "5Gother", OURS, "5Hother"],
        registration_blocks=[900_000, 999_000, 900_000, 1_000_100, 900_000],
    )
    state = sm.read_chain_state(chain, 68, OURS)
    assert state.registered_uids == (1, 3)
    assert state.registered == 2
    assert state.recycle_tao == pytest.approx(0.131)
    assert state.registrations_this_interval == 1
    # The stagger keys on the most recent of our registrations, not the oldest.
    assert state.last_registration_block == 1_000_100
    assert state.epoch == 1_000_500 // sm.EPOCH_LENGTH
    assert state.interval == 1_000_500 // sm.ADJUSTMENT_INTERVAL


def test_reconcile_on_an_empty_field_takes_the_last_block_from_the_log():
    """Both our hotkeys are deregistered: zero uids, memory from the log."""
    chain = FakeSubtensor(
        block=1_000_500, recycle=0.131, regs=0,
        coldkeys=["5Fother", "5Gother"], registration_blocks=[900_000, 900_000],
    )
    bare = sm.read_chain_state(chain, 68, OURS)
    assert bare.registered_uids == () and bare.last_registration_block is None
    history = decaying(1_000_440, 61, 0.131)
    same_interval = (1_000_500 // sm.ADJUSTMENT_INTERVAL) * sm.ADJUSTMENT_INTERVAL + 20
    remembered = sm.read_chain_state(chain, 68, OURS, log_last_block=same_interval)
    assert remembered.registered == 0
    assert remembered.last_registration_block == same_interval
    # A hotkey we bought and then had pruned still blocks a same-interval rebuy:
    # the interval quota was spent even though nothing survived to show for it.
    assert sm.decide(remembered, history, [], sm.SlotPolicy()).reason == (
        sm.REASON_OWN_INTERVAL
    )
    # One interval back it is the stagger, not the quota, that holds us.
    earlier = sm.read_chain_state(
        chain, 68, OURS, log_last_block=same_interval - sm.ADJUSTMENT_INTERVAL
    )
    assert sm.decide(earlier, history, [], sm.SlotPolicy()).reason == sm.REASON_STAGGER


def test_the_report_line_is_printable_for_every_decision(capsys):
    history = decaying(1_000_000, 61, 0.13)
    for state in (
        state_at(history),
        state_at(history, registered=()),
        state_at(history, registered=tuple(range(sm.SlotPolicy().target))),
        state_at(history, last=history[-1].block - 1),
    ):
        decision = sm.decide(state, history, [], sm.SlotPolicy())
        sm._report(decision, executed=False, hotkey="nova68-slot-1")
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("SLOT_BUY reason=")
    assert all("executed=false" in line for line in lines)
    assert f"reason={sm.REASON_BUY_URGENT}" in lines[1]
    assert f"reason={sm.REASON_TARGET_MET}" in lines[2]
    assert f"reason={sm.REASON_OWN_INTERVAL}" in lines[3]


def test_the_credential_source_follows_the_platform():
    """A Raspberry Pi has no Keychain; macOS has no systemd."""
    import sys as _sys

    from nanobody_opt import slot_manager as sm

    assert sm.SECRET_SOURCES == ("auto", "keychain", "systemd")
    expected = "keychain" if _sys.platform == "darwin" else "systemd"
    assert sm._default_secret_source() == expected


def test_the_systemd_credential_is_read_from_the_directory_systemd_supplies(tmp_path, monkeypatch):
    from nanobody_opt import slot_manager as sm

    monkeypatch.setenv("NOVA_TEST_CRED", "coldkey-password")
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
    (tmp_path / "coldkey-password").write_text("s3cret\n")
    assert sm._password_from_systemd_credential("NOVA_TEST_CRED") == "s3cret"


def test_running_outside_systemd_fails_with_an_explanation_not_a_fallback(tmp_path, monkeypatch):
    """No prompt, no Keychain, no plaintext file beside the unit."""
    from nanobody_opt import slot_manager as sm

    monkeypatch.setenv("NOVA_TEST_CRED", "coldkey-password")
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        sm._password_from_systemd_credential("NOVA_TEST_CRED")
    assert "CREDENTIALS_DIRECTORY" in str(excinfo.value)


def test_a_missing_or_empty_credential_is_distinguished(tmp_path, monkeypatch):
    from nanobody_opt import slot_manager as sm

    monkeypatch.setenv("NOVA_TEST_CRED", "coldkey-password")
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
    with pytest.raises(SystemExit) as missing:
        sm._password_from_systemd_credential("NOVA_TEST_CRED")
    assert "no systemd credential" in str(missing.value)
    (tmp_path / "coldkey-password").write_text("\n")
    with pytest.raises(SystemExit) as empty:
        sm._password_from_systemd_credential("NOVA_TEST_CRED")
    assert "is empty" in str(empty.value)


def test_a_new_hotkey_mnemonic_never_reaches_stdout(capsys):
    """bittensor prints the mnemonic on creation; under systemd that is the journal.

    The first live registration wrote a twelve-word private key in clear text to
    a file that survives reboots. Discarding the stream is the fix, rather than
    filtering it, because a filter must be right about every future version's
    wording.
    """
    from nanobody_opt import slot_manager as sm

    class LoudWallet:
        def create_new_hotkey(self, **kwargs):
            print("The mnemonic to the new hotkey is: correct horse battery staple")
            print("also on stderr", file=sys.stderr)

    sm._create_hotkey_quietly(LoudWallet())
    captured = capsys.readouterr()
    assert "mnemonic" not in captured.out
    assert "correct horse" not in captured.out
    assert "correct horse" not in captured.err
    assert captured.out == ""


def test_hotkey_creation_survives_a_signature_without_suppress():
    """Older bittensor has no `suppress`; the redirect must still hold."""
    from nanobody_opt import slot_manager as sm

    seen = {}

    class OldWallet:
        def create_new_hotkey(self, **kwargs):
            if "suppress" in kwargs:
                raise TypeError("unexpected keyword argument 'suppress'")
            seen.update(kwargs)
            print("mnemonic: leaked")

    sm._create_hotkey_quietly(OldWallet())
    assert seen == {"n_words": 12, "use_password": False, "overwrite": False}


class _StakeEntry:
    def __init__(self, netuid, hotkey, alpha):
        self.netuid = netuid
        self.hotkey_ss58 = hotkey
        self.stake = alpha


class _Subtensor:
    def __init__(self, entries):
        self._entries = entries

    def get_stake_info_for_coldkey(self, coldkey):
        return list(self._entries)


OURS = "5EkCwcNT"


def test_delegated_alpha_is_counted_not_only_our_own_hotkeys():
    """Our miner hotkeys held 0.000000 while the coldkey held 47.3 alpha on
    this subnet delegated to three hotkeys we do not operate. A metagraph walk
    keyed on our coldkey sees none of it and reports an empty treasury."""
    subtensor = _Subtensor([
        _StakeEntry(68, "5Gp6i5PLVV", 2.967264),
        _StakeEntry(68, "5HmBXdq8Nj", 41.697699),
        _StakeEntry(68, "5HMe8oMrVZ", 2.653606),
        _StakeEntry(2, "5Fvq8unzMX", 528.290484),   # another subnet
        _StakeEntry(68, "5Empty0000", 0.0),         # dust
    ])
    holdings = sm.alpha_holdings(subtensor, 68, OURS)
    assert holdings["alpha_total"] == pytest.approx(47.318569)
    assert [h["hotkey"] for h in holdings["holdings"]] == [
        "5HmBXdq8Nj", "5Gp6i5PLVV", "5HMe8oMrVZ"
    ], "largest position first"


def test_another_subnets_alpha_is_not_counted_as_ours_here():
    subtensor = _Subtensor([_StakeEntry(2, "5Fvq8unzMX", 1753.3)])
    assert sm.alpha_holdings(subtensor, 68, OURS)["alpha_total"] == 0.0


def test_an_sdk_without_the_stake_map_fails_loudly():
    class _Old:
        def metagraph(self, netuid):
            raise AssertionError("must not fall back to the metagraph")

    with pytest.raises(RuntimeError, match="delegated alpha"):
        sm.alpha_holdings(_Old(), 68, OURS)


def test_the_report_values_the_position_and_writes_the_command():
    holdings = {"alpha_total": 10.0,
                "holdings": [{"hotkey": "hkA", "alpha": 10.0}]}
    report = sm.treasury_report(
        holdings=holdings, price_tao=0.02151, netuid=68,
        wallet_name="mining_ops", slippage=0.02, floor_alpha=1.0,
    )
    assert report["value_tao"] == pytest.approx(0.2151)
    assert report["actionable"] is True
    assert len(report["commands"]) == 1
    assert "--amount 10.000000" in report["commands"][0]


def test_a_conversion_command_always_carries_a_price_limit():
    """A bare market order can fill the whole position into a thin book."""
    command = sm.conversion_command(
        netuid=68, hotkey="hkA", alpha=10.0, price_tao=0.02,
        wallet_name="mining_ops", slippage=0.02,
    )
    assert "--price-limit 0.01960000" in command
    assert "--tolerance 0.020" in command


def test_a_position_below_the_floor_is_reported_but_not_actionable():
    holdings = {"alpha_total": 0.4,
                "holdings": [{"hotkey": "hkA", "alpha": 0.4}]}
    report = sm.treasury_report(
        holdings=holdings, price_tao=0.02, netuid=68,
        wallet_name="mining_ops", slippage=0.02, floor_alpha=1.0,
    )
    assert report["actionable"] is False
    assert report["commands"] == []


def test_the_agent_never_executes_the_conversion_itself():
    """The agent reports and computes; the signature stays with the operator."""
    import inspect

    source = inspect.getsource(sm)
    for forbidden in ("unstake(", "remove_stake(", "swap_stake(", "sell("):
        assert forbidden not in source, f"slot_manager can execute {forbidden}"


def test_the_cli_target_default_follows_the_policy():
    """The CLI default must not be a literal that drifts from SlotPolicy."""
    import inspect

    source = inspect.getsource(sm)
    completed = subprocess.run(
        [sys.executable, "-m", "nanobody_opt.slot_manager", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert f"default: {sm.SlotPolicy().target}" in completed.stdout
    # And the literal is gone, so raising the target cannot silently miss here.
    assert "default=7," not in source


def test_the_stagger_keeps_renewals_in_separate_intervals_at_the_live_target():
    """
    The fleet size is bounded by the renewal spacing, not by taste: two
    immunity expiries inside one adjustment interval make us bid against
    ourselves. 5000 // 13 is 384 and clears 360; 5000 // 14 is 357 and does not.
    """
    policy = sm.SlotPolicy()
    assert sm.default_stagger_blocks(policy.target) > sm.ADJUSTMENT_INTERVAL, (
        f"target {policy.target} spaces renewals "
        f"{sm.default_stagger_blocks(policy.target)} blocks apart, inside the "
        f"{sm.ADJUSTMENT_INTERVAL}-block adjustment interval"
    )
    assert sm.default_stagger_blocks(13) > sm.ADJUSTMENT_INTERVAL
    assert sm.default_stagger_blocks(14) <= sm.ADJUSTMENT_INTERVAL


def test_the_deployed_target_matches_the_policy():
    """ansible/slots.yml and SlotPolicy must not disagree about the fleet size."""
    import re

    slots = (Path(__file__).resolve().parents[1] / "ansible" / "slots.yml").read_text()
    deployed = re.search(r"slot_manager_minimum_hotkeys:\s*(\d+)", slots)
    assert deployed, "slots.yml no longer sets slot_manager_minimum_hotkeys"
    assert int(deployed.group(1)) == sm.SlotPolicy().target

    spend = re.search(r"slot_manager_max_spend_tao_per_day:\s*([0-9.]+)", slots)
    assert spend, "slots.yml no longer sets slot_manager_max_spend_tao_per_day"
    assert float(spend.group(1)) == sm.SlotPolicy().max_spend_tao_per_day
