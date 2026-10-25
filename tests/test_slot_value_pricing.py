"""Value pricing: each slot is bought only up to what that slot is worth.

Pure policy tests against injected chain snapshots -- no network, no wallet.  The
historical policy is covered by ``test_slot_manager.py`` and must not change; this
file only exercises what ``SlotPolicy(value_curve=...)`` adds, the slot-life
ledger that feeds it, and the burst helper.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt import slot_manager as sm

REPO = Path(__file__).resolve().parents[1]

#: Far from an epoch edge, so the submission freeze is not in play unless a test
#: asks for it: 321 blocks remain in the epoch.
BLOCK = 361 * 25_000 + 40
WIN_VALUE = 0.832          # 41.6 alpha at tau0.020
LIFE = 18.0                # epochs
ECONOMICS = sm.SlotEconomics(win_value_tao=WIN_VALUE, slot_life_epochs=LIFE)


def value_policy(**kw):
    kw.setdefault("target", 5)
    kw.setdefault("reserve_slots", 8)
    kw.setdefault("max_recycle_tao", 0.25)
    kw.setdefault("reserve_max_recycle_tao", 0.175)
    kw.setdefault("max_spend_tao_per_day", 4.5)
    kw.setdefault("value_curve", sm.marginal_win_curve(13))
    return sm.SlotPolicy(**kw)


def state(registered: int, price: float, *, block: int = BLOCK, last=None, oldest=None, regs=0):
    return sm.ChainState(
        block=block,
        recycle_tao=price,
        registrations_this_interval=regs,
        registered_uids=tuple(range(registered)),
        last_registration_block=last,
        oldest_registration_block=oldest,
    )


def decide(registered, price, policy=None, economics=ECONOMICS, spends=(), **kw):
    st = state(registered, price, **kw)
    return sm.decide(
        st, [sm.PriceSample(st.block, st.recycle_tao)], list(spends),
        policy or value_policy(), now=1_000_000.0, economics=economics,
    )


# ---------------------------------------------------------------------------
# The schedule.
# ---------------------------------------------------------------------------


def test_the_marginal_curve_is_the_fitted_concave_curve():
    curve = sm.marginal_win_curve(13)
    assert curve[0] == pytest.approx(0.061)
    assert all(a > b for a, b in zip(curve, curve[1:])), "marginal value must fall"
    assert sum(curve) == pytest.approx(0.061 * 13**0.57)
    # 6 tickets win about 18% of epochs and 12 about 25%, as replayed.
    assert sum(curve[:6]) == pytest.approx(0.169, abs=0.01)
    assert sum(curve[:12]) == pytest.approx(0.251, abs=0.01)


def test_a_slot_cap_is_extra_win_chance_times_win_value_times_life():
    curve = sm.marginal_win_curve(13)
    cap = sm.slot_value_cap_tao(6, curve, ECONOMICS)
    assert cap == pytest.approx(curve[5] * WIN_VALUE * LIFE)
    # Linear in both measured inputs.
    longer = sm.SlotEconomics(WIN_VALUE, LIFE * 2)
    richer = sm.SlotEconomics(WIN_VALUE * 2, LIFE)
    assert sm.slot_value_cap_tao(6, curve, longer) == pytest.approx(cap * 2)
    assert sm.slot_value_cap_tao(6, curve, richer) == pytest.approx(cap * 2)
    # The schedule falls with the slot index: the 12th is worth less than the 4th.
    caps = [sm.slot_value_cap_tao(k, curve, ECONOMICS) for k in range(1, 14)]
    assert all(a > b for a, b in zip(caps, caps[1:]))
    assert caps[3] > 0.28 and caps[11] < 0.19


def test_unknown_economics_have_no_cap_and_slots_past_the_curve_are_worthless():
    curve = sm.marginal_win_curve(13)
    assert sm.slot_value_cap_tao(3, curve, None) is None
    assert sm.slot_value_cap_tao(3, curve, sm.SlotEconomics(None, LIFE)) is None
    assert sm.slot_value_cap_tao(3, curve, sm.SlotEconomics(WIN_VALUE, None)) is None
    assert sm.slot_value_cap_tao(14, curve, ECONOMICS) == 0.0
    with pytest.raises(ValueError):
        sm.slot_value_cap_tao(0, curve, ECONOMICS)


def test_a_policy_must_carry_a_curve_for_every_slot_it_may_hold():
    with pytest.raises(ValueError):
        sm.SlotPolicy(target=5, reserve_slots=8, value_curve=sm.marginal_win_curve(10))
    with pytest.raises(ValueError):
        sm.SlotPolicy(target=1, value_curve=(0.05, -0.01), reserve_slots=1)
    assert sm.SlotPolicy().value_pricing is False        # the historical policy is untouched
    assert value_policy().value_pricing is True
    assert value_policy().maximum_registered == 13


# ---------------------------------------------------------------------------
# Buying above the floor: whenever the next slot is worth its price.
# ---------------------------------------------------------------------------


def test_a_cheap_slot_above_the_floor_is_bought():
    decision = decide(registered=8, price=0.12)
    assert decision.action == sm.BUY
    assert decision.reason == sm.REASON_BUY_VALUE
    assert decision.detail["slot_index"] == 9
    assert decision.detail["slot_value_cap_tao"] == pytest.approx(
        sm.marginal_win_curve(13)[8] * WIN_VALUE * LIFE
    )
    assert decision.detail["win_value_tao"] == WIN_VALUE
    assert decision.detail["slot_life_epochs"] == LIFE


def test_a_slot_above_its_worth_is_refused_even_though_it_is_under_the_hard_ceiling():
    cap = sm.slot_value_cap_tao(11, sm.marginal_win_curve(13), ECONOMICS)
    assert cap < 0.25
    decision = decide(registered=10, price=cap + 0.01)
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_VALUE_CAP
    # ...and at exactly its worth it is bought.
    assert decide(registered=10, price=cap - 1e-9).action == sm.BUY


def test_the_same_price_buys_early_slots_and_refuses_late_ones():
    price = 0.22
    assert decide(registered=6, price=price).action == sm.BUY      # slot 7 is worth ~0.23
    assert decide(registered=11, price=price).reason == sm.REASON_VALUE_CAP   # slot 12 ~0.18


def test_no_stagger_and_no_interval_pause_in_the_value_zone():
    """A cheap price is always taken, even straight after our own purchase."""
    just_bought = dict(last=BLOCK - 1, regs=3)
    burst = decide(registered=7, price=0.15, **just_bought)
    assert burst.action == sm.BUY and burst.reason == sm.REASON_BUY_VALUE
    # The historical policy would have paced the very same snapshot.
    old = sm.SlotPolicy(target=5, reserve_slots=8, max_recycle_tao=0.25,
                        reserve_max_recycle_tao=0.175, max_spend_tao_per_day=4.5)
    st = state(7, 0.15, **just_bought)
    paced = sm.decide(st, [sm.PriceSample(st.block, st.recycle_tao)], [], old, now=1e6)
    assert paced.action == sm.WAIT
    assert paced.reason in (sm.REASON_OWN_INTERVAL, sm.REASON_STAGGER, sm.REASON_TARGET_MET, sm.REASON_RESERVE_PRICE)


def test_a_burst_ends_by_itself_when_our_own_purchases_lift_the_price():
    """Each registration multiplies the burn; the cap is the only brake needed."""
    registered, price, bought = 5, 0.11, 0
    while True:
        decision = decide(registered=registered, price=price, last=BLOCK - 1)
        if decision.action != sm.BUY:
            break
        registered += 1
        bought += 1
        price *= 1.12
        assert bought < 20, "the burst must terminate"
    assert 5 <= registered <= 13
    assert bought >= 3, "a trough should buy several slots at once"
    assert decision.reason in (sm.REASON_VALUE_CAP, sm.REASON_TARGET_MET)


def test_the_cap_of_thirteen_holds_at_any_price():
    decision = decide(registered=13, price=0.01)
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_TARGET_MET


# ---------------------------------------------------------------------------
# The floor: always at least five, paid for up to the hard ceiling.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("held", [0, 1, 2, 3, 4])
def test_below_the_floor_any_price_up_to_the_hard_ceiling_buys(held):
    decision = decide(registered=held, price=0.25, last=BLOCK - 1, regs=2)
    assert decision.action == sm.BUY
    assert decision.reason == sm.REASON_BUY_URGENT


def test_the_hard_ceiling_still_refuses_below_the_floor():
    decision = decide(registered=3, price=0.2501)
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_CEILING


def test_an_expiring_floor_slot_is_renewed_above_its_slot_value():
    """Holding exactly the floor, the oldest slot nearing the end of immunity is renewed."""
    short_lived = sm.SlotEconomics(WIN_VALUE, 12.0)
    cap6 = sm.slot_value_cap_tao(6, sm.marginal_win_curve(13), short_lived)
    price = 0.22
    assert price > cap6                       # not worth it as a spare slot...
    oldest = BLOCK - (sm.IMMUNITY_PERIOD - 100)   # 100 blocks of immunity left
    decision = decide(registered=5, price=price, economics=short_lived, oldest=oldest)
    assert decision.action == sm.BUY
    assert decision.reason == sm.REASON_BUY_RENEWAL    # ...but the floor must not dip


def test_slots_above_the_floor_are_not_renewed_before_they_are_worth_it():
    oldest = BLOCK - (sm.IMMUNITY_PERIOD - 100)
    cap9 = sm.slot_value_cap_tao(9, sm.marginal_win_curve(13), ECONOMICS)
    decision = decide(registered=8, price=cap9 + 0.02, oldest=oldest)
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_VALUE_CAP


# ---------------------------------------------------------------------------
# Rails that value pricing must not weaken.
# ---------------------------------------------------------------------------


def test_unreadable_economics_fall_back_to_the_cheap_reserve_ceiling():
    assert decide(registered=8, price=0.17, economics=None).action == sm.BUY
    refused = decide(registered=8, price=0.18, economics=None)
    assert refused.action == sm.WAIT and refused.reason == sm.REASON_VALUE_CAP
    unreadable = sm.SlotEconomics(None, LIFE)
    assert decide(registered=8, price=0.18, economics=unreadable).action == sm.WAIT


def test_the_daily_budget_still_binds():
    spends = [sm.SpendRecord(timestamp=1_000_000.0 - 60, recycle_tao=4.45)]
    decision = decide(registered=8, price=0.12, spends=spends)
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_BUDGET


def test_the_submission_freeze_still_binds():
    frozen_block = 361 * 25_000 + 361 - 20      # 20 blocks before the epoch closes
    decision = decide(registered=8, price=0.12, block=frozen_block)
    assert decision.action == sm.WAIT
    assert decision.reason == sm.REASON_SUBMISSION_FREEZE


def test_the_register_ceiling_for_a_value_buy_is_its_own_worth():
    decision = decide(registered=8, price=0.12)
    cap = decision.detail["slot_zone_cap_tao"]
    assert cap == pytest.approx(decision.detail["slot_value_cap_tao"])
    assert cap < 0.25      # the extrinsic must refuse a price above this, not just the 0.25 rail


# ---------------------------------------------------------------------------
# The slot-life ledger and estimator.
# ---------------------------------------------------------------------------


def slot(uid, hotkey, reg):
    return (uid, hotkey, reg)


def test_the_first_tick_records_a_snapshot_and_nothing_else():
    out = sm.update_life_ledger([], [slot(1, "hkA", 100), slot(2, "hkB", 200)], 1000)
    assert [r["kind"] for r in out] == [sm.LEDGER_ROSTER]
    assert out[0]["slots"] == [[1, "hkA", 100], [2, "hkB", 200]]


def test_an_unchanged_roster_writes_nothing():
    roster = [slot(1, "hkA", 100), slot(2, "hkB", 200)]
    first = sm.update_life_ledger([], roster, 1000)
    assert sm.update_life_ledger(first, roster, 1030) == []


def test_a_lost_hotkey_is_an_eviction_with_its_registration_block():
    ledger = sm.update_life_ledger([], [slot(1, "hkA", 100), slot(2, "hkB", 200)], 1000)
    out = sm.update_life_ledger(ledger, [slot(2, "hkB", 200)], 6000)
    departed = [r for r in out if r["kind"] == sm.LEDGER_DEPARTED]
    assert len(departed) == 1
    assert departed[0]["hotkey"] == "hkA" and departed[0]["registered_block"] == 100
    assert departed[0]["block"] == 6000 and departed[0]["replaced_by_us"] is False
    assert out[-1]["kind"] == sm.LEDGER_ROSTER


def test_a_slot_replaced_by_our_own_purchase_is_not_an_eviction():
    ledger = sm.update_life_ledger([], [slot(32, "hkOld", 100), slot(2, "hkB", 200)], 1000)
    out = sm.update_life_ledger(ledger, [slot(32, "hkNew", 5990), slot(2, "hkB", 200)], 6000)
    departed = [r for r in out if r["kind"] == sm.LEDGER_DEPARTED]
    assert departed[0]["hotkey"] == "hkOld" and departed[0]["replaced_by_us"] is True
    lives = sm.slot_lives(ledger + out, [slot(32, "hkNew", 5990), slot(2, "hkB", 200)])
    old = next(life for life in lives if life.registered_block == 100)
    assert old.end_block == 6000 and old.evicted is False


def test_an_empty_roster_is_a_failed_read_and_is_never_recorded():
    ledger = sm.update_life_ledger([], [slot(1, "hkA", 100)], 1000)
    assert sm.update_life_ledger(ledger, [], 2000) == []


def test_slot_lives_join_finished_and_running_slots_without_duplicates():
    ledger = sm.update_life_ledger([], [slot(1, "hkA", 100), slot(2, "hkB", 200)], 1000)
    ledger += sm.update_life_ledger(ledger, [slot(2, "hkB", 200)], 6000)
    lives = sm.slot_lives(ledger + ledger, [slot(2, "hkB", 200)])     # duplicated log lines
    assert sorted((l.registered_block, l.end_block, l.evicted) for l in lives) == [
        (100, 6000, True), (200, None, False)]


def lives_with(n, *, evicted_after_epochs, now=1_000_000, evicted=True):
    """``n`` finished slots, each registered recently and ended after a given exposure."""
    out = []
    for i in range(n):
        reg = now - 50_000 - i * 10
        end = reg + sm.IMMUNITY_PERIOD + int(evicted_after_epochs * sm.EPOCH_LENGTH)
        out.append(sm.SlotLife(reg, end, evicted))
    return out


def test_with_no_observations_the_estimate_is_the_prior():
    est = sm.estimate_slot_life([], 1_000_000)
    assert est["slot_life_epochs"] == pytest.approx(18.0)
    assert est["events"] == 0.0


def test_observed_lifespans_at_the_prior_leave_the_estimate_alone():
    prior_post = 18.0 - sm.IMMUNITY_EPOCHS
    est = sm.estimate_slot_life(lives_with(60, evicted_after_epochs=prior_post), 1_000_000)
    assert est["slot_life_epochs"] == pytest.approx(18.0, abs=0.05)


def test_sustained_faster_eviction_shortens_the_life_estimate():
    est = sm.estimate_slot_life(lives_with(60, evicted_after_epochs=0.0), 1_000_000)
    assert est["slot_life_epochs"] < 15.5
    # and a few quick evictions only nudge it, because the prior is worth 20 events
    few = sm.estimate_slot_life(lives_with(3, evicted_after_epochs=0.0), 1_000_000)
    assert 16.5 < few["slot_life_epochs"] < 18.0


def test_slots_still_held_add_exposure_but_never_an_eviction():
    alive = [sm.SlotLife(1_000_000 - 90_000 + i, None) for i in range(10)]
    est = sm.estimate_slot_life(alive, 1_000_000)
    assert est["events"] == 0.0 and est["exposure_epochs"] > 0
    assert est["slot_life_epochs"] > 18.0            # survivors say slots last longer, not shorter


def test_a_renewal_we_chose_is_not_evidence_about_rivals():
    chosen = sm.estimate_slot_life(lives_with(40, evicted_after_epochs=0.0, evicted=False), 1_000_000)
    forced = sm.estimate_slot_life(lives_with(40, evicted_after_epochs=0.0, evicted=True), 1_000_000)
    assert chosen["events"] == 0.0 and forced["events"] == 40.0
    assert chosen["slot_life_epochs"] > forced["slot_life_epochs"]


def test_old_registrations_fall_out_of_the_window_and_the_estimate_is_clamped():
    ancient = [sm.SlotLife(1, 6000, True)] * 50
    assert sm.estimate_slot_life(ancient, 1_000_000)["slots_in_window"] == 0.0
    huge = [sm.SlotLife(1_000_000 - 99_000, None)] * 200
    assert sm.estimate_slot_life(huge, 1_000_000)["slot_life_epochs"] <= sm.LIFE_CEILING_EPOCHS


def test_the_estimate_feeds_the_cap_end_to_end():
    longer = sm.estimate_slot_life([sm.SlotLife(1_000_000 - 90_000 + i, None) for i in range(10)], 1_000_000)
    curve = sm.marginal_win_curve(13)
    base = sm.slot_value_cap_tao(8, curve, sm.SlotEconomics(WIN_VALUE, 18.0))
    grown = sm.slot_value_cap_tao(8, curve, sm.SlotEconomics(WIN_VALUE, longer["slot_life_epochs"]))
    assert grown > base


# ---------------------------------------------------------------------------
# Burst control and the command line.
# ---------------------------------------------------------------------------


def test_only_a_landed_value_buy_triggers_another_look_and_the_burst_is_bounded():
    assert sm.should_burst(executed=True, bursts_done=0, burst_max=8, value_pricing=True)
    assert not sm.should_burst(executed=False, bursts_done=0, burst_max=8, value_pricing=True)
    assert not sm.should_burst(executed=True, bursts_done=0, burst_max=8, value_pricing=False)
    assert not sm.should_burst(executed=True, bursts_done=7, burst_max=8, value_pricing=True)
    assert not sm.should_burst(executed=True, bursts_done=0, burst_max=1, value_pricing=True)


def namespace(**kw):
    base = dict(
        target=5, reserve_slots=0, reserve_max_recycle_tao=0.175, submission_freeze_blocks=90,
        urgent_below=None, max_recycle_tao=0.25, max_spend_tao_per_day=4.5, stagger_blocks=None,
        lookback_blocks=180, floor_tolerance=0.05, value_pricing=True, maximum_hotkeys=13,
        win_curve=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_the_cli_builds_a_floor_five_cap_thirteen_policy():
    policy = sm._policy_from_args(namespace())
    assert policy.target == 5 and policy.maximum_registered == 13 and policy.reserve_slots == 8
    assert policy.value_pricing and len(policy.value_curve) == 13
    assert policy.max_recycle_tao == 0.25


def test_the_cli_refuses_a_cap_below_the_floor_and_a_short_curve():
    with pytest.raises(SystemExit):
        sm._policy_from_args(namespace(maximum_hotkeys=4))
    with pytest.raises(SystemExit):
        sm._policy_from_args(namespace(win_curve="0.06,0.03"))
    with pytest.raises(SystemExit):
        sm._policy_from_args(namespace(win_curve="a,b"))
    custom = sm._policy_from_args(namespace(win_curve=",".join(["0.02"] * 13)))
    assert custom.value_curve == (0.02,) * 13


def test_without_the_flag_the_historical_policy_is_built_unchanged():
    policy = sm._policy_from_args(namespace(value_pricing=False, target=12, reserve_slots=1,
                                            max_recycle_tao=0.20))
    assert policy.value_pricing is False
    assert policy.target == 12 and policy.reserve_slots == 1


def test_help_documents_the_new_flags_without_importing_the_chain_stack():
    completed = subprocess.run(
        [sys.executable, "-m", "nanobody_opt.slot_manager", "--help"],
        capture_output=True, text=True, check=True, cwd=REPO,
    )
    for flag in ("--value-pricing", "--maximum-hotkeys", "--win-payout-alpha",
                 "--win-curve", "--slot-life-prior-epochs", "--life-ledger", "--burst-max"):
        assert flag in completed.stdout


# ---------------------------------------------------------------------------
# main() wired end to end against a fake chain: ledger, economics, burst.
# ---------------------------------------------------------------------------

import json
import types

COLDKEY = "5" + "E" * 47        # long enough to pass the truncated-address guard


class FakeChain:
    """Just enough of the SDK for ``main`` and ``read_chain_state``."""

    def __init__(self, held, price, *, alpha_price=0.02, block=BLOCK):
        self.block = block
        self.price = price
        self.alpha_price = alpha_price
        self.ours = list(held)               # [(uid, hotkey, registration_block)]
        self.registered_calls = 0
        self.size = 64

    # -- what read_chain_state reads ------------------------------------------
    def metagraph(self, netuid):
        coldkeys = ["5Rival" + str(i) * 40 for i in range(self.size)]
        hotkeys = ["hk-rival-" + str(i) for i in range(self.size)]
        regs = [1] * self.size
        for uid, hotkey, reg in self.ours:
            coldkeys[uid], hotkeys[uid], regs[uid] = COLDKEY, hotkey, reg
        return types.SimpleNamespace(coldkeys=coldkeys, hotkeys=hotkeys, block_at_registration=regs)

    def get_current_block(self):
        return self.block

    def recycle(self, netuid):
        return types.SimpleNamespace(tao=self.price)

    def query_subtensor(self, name, params=None):
        return 0

    def subnet(self, netuid):
        return types.SimpleNamespace(price=self.alpha_price)

    # -- what the patched _register does --------------------------------------
    def register(self):
        self.registered_calls += 1
        before = self.price
        free = next(uid for uid in range(self.size) if uid not in {u for u, _, _ in self.ours})
        self.ours.append((free, f"hk-new-{self.registered_calls}-{self.block}", self.block))
        self.block += 1
        self.price *= 1.12
        return True, before


def run_main(monkeypatch, tmp_path, chain, *extra, execute=True):
    monkeypatch.setattr(sm, "_import_bittensor", lambda: types.SimpleNamespace(
        Subtensor=lambda network=None: chain,
        Wallet=lambda **kw: pytest.fail('empty inventory must not load a wallet')))
    monkeypatch.setattr(sm, "_hotkey_names", lambda wallet: [])
    monkeypatch.setattr(sm, "_register", lambda **kw: chain.register())
    monkeypatch.setattr(sm.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("NOVA_TEST_KEYCHAIN", "svc")
    argv = [
        "--coldkey-ss58", COLDKEY, "--decision-log", str(tmp_path / "slots.jsonl"),
        "--value-pricing", "--minimum-hotkeys", "5", "--maximum-hotkeys", "13",
        "--max-recycle-tao", "0.25", "--max-spend-tao-per-day", "4.5",
        "--keychain-service-env", "NOVA_TEST_KEYCHAIN", "--credential-source", "keychain",
        "--interval", "0", *extra,
    ]
    if execute:
        argv.append("--execute-registrations")
    sm.main(argv)
    return [json.loads(line) for line in (tmp_path / "slots.jsonl").read_text().splitlines()]


def test_a_cheap_price_is_burst_bought_in_one_run_and_ends_at_the_value_cap(monkeypatch, tmp_path):
    held = [(uid, f"hk-held-{uid}", BLOCK - 3000 - uid) for uid in range(6)]
    chain = FakeChain(held, price=0.11)
    records = run_main(monkeypatch, tmp_path, chain, "--burst-max", "8")
    buys = [r for r in records if r.get("executed")]
    assert 3 <= len(buys) <= 7, [r["reason"] for r in records]
    assert chain.registered_calls == len(buys)
    assert 6 < len(chain.ours) <= 13
    assert all(r["reason"] == sm.REASON_BUY_VALUE for r in buys)
    last = records[-1]
    assert last["action"] == sm.WAIT and last["reason"] in (sm.REASON_VALUE_CAP, sm.REASON_TARGET_MET)
    # every purchase was at or under the worth of the slot it bought
    for r in buys:
        assert r["quoted_recycle_tao"] <= r["slot_value_cap_tao"] + 1e-9
    # the log carries the measured inputs behind each decision
    assert buys[0]["win_value_tao"] == pytest.approx(41.6 * 0.02)
    assert buys[0]["slot_life_epochs"] == pytest.approx(18.0, abs=0.5)
    ledger = [json.loads(l) for l in (tmp_path / "subnet68_slot_life.jsonl").read_text().splitlines()]
    assert ledger and ledger[0]["kind"] == sm.LEDGER_ROSTER


def test_without_burst_max_one_run_buys_once(monkeypatch, tmp_path):
    held = [(uid, f"hk-held-{uid}", BLOCK - 3000 - uid) for uid in range(6)]
    chain = FakeChain(held, price=0.11)
    run_main(monkeypatch, tmp_path, chain)
    assert chain.registered_calls == 1


def test_a_dry_run_never_registers_or_bursts(monkeypatch, tmp_path):
    held = [(uid, f"hk-held-{uid}", BLOCK - 3000 - uid) for uid in range(6)]
    chain = FakeChain(held, price=0.11)
    records = run_main(monkeypatch, tmp_path, chain, "--burst-max", "8", execute=False)
    assert chain.registered_calls == 0
    assert len(records) == 1 and records[0]["dry_run"] is True and records[0]["action"] == sm.BUY


def test_a_price_above_the_next_slots_worth_buys_nothing(monkeypatch, tmp_path):
    held = [(uid, f"hk-held-{uid}", BLOCK - 3000 - uid) for uid in range(11)]
    chain = FakeChain(held, price=0.23)
    records = run_main(monkeypatch, tmp_path, chain, "--burst-max", "8")
    assert chain.registered_calls == 0
    assert records[-1]["reason"] == sm.REASON_VALUE_CAP


def test_an_unreadable_alpha_price_falls_back_to_the_reserve_ceiling(monkeypatch, tmp_path):
    held = [(uid, f"hk-held-{uid}", BLOCK - 3000 - uid) for uid in range(8)]
    chain = FakeChain(held, price=0.18, alpha_price=0.0)
    records = run_main(monkeypatch, tmp_path, chain)
    assert chain.registered_calls == 0
    assert records[-1]["reason"] == sm.REASON_VALUE_CAP
    assert records[-1]["win_value_tao"] is None


def test_a_lost_slot_between_runs_lands_in_the_ledger_and_moves_the_estimate(monkeypatch, tmp_path):
    held = [(uid, f"hk-held-{uid}", BLOCK - 5200 - uid) for uid in range(8)]
    chain = FakeChain(held, price=0.30)               # too dear to buy: runs only observe
    run_main(monkeypatch, tmp_path, chain)
    chain.ours = chain.ours[:-2]                      # two slots evicted by rivals
    chain.block += 25
    records = run_main(monkeypatch, tmp_path, chain)
    ledger = [json.loads(l) for l in (tmp_path / "subnet68_slot_life.jsonl").read_text().splitlines()]
    departed = [r for r in ledger if r["kind"] == sm.LEDGER_DEPARTED]
    assert len(departed) == 2 and all(r["replaced_by_us"] is False for r in departed)
    assert records[-1]["slot_life_events"] == 2.0
