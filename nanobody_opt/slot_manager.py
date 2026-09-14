"""Hold a target number of subnet-68 registrations by buying into the price decay.

Subnet 68 is full (``MaxAllowedUids`` 64, ``n`` 64, six validator permits), so
every registration deregisters the lowest-pruning-score non-immune uid and our
own holdings decay out of the field at roughly the same rate.  Keeping nine
hotkeys registered is therefore a standing purchase schedule, not a one-off
buy, and the only question worth answering on each poll is *not now or never*
but *now or a few hundred blocks from now*.

Three measured facts drive the whole policy, and none of them is the textbook
difficulty formula -- ``LastAdjustmentBlock`` reads about a million blocks
stale on this subnet, so that formula demonstrably is not what moves ``Burn``
here.  Measured by polling instead:

* The recycle price *decays* and does not ratchet.  Over nineteen consecutive
  blocks it fell tau0.135167 -> tau0.130312, which is -0.192% per block and a
  halving every 360 blocks -- exactly one ``adjustment_interval``, 72 minutes.
  :data:`DECAY_PER_BLOCK` is written as that halving so the constant states the
  observation rather than a fitted decimal.
* It *jumps* on registration: +33% observed from three competitors buying
  inside one interval.  A jump decays away on the same clock, so the cost of
  buying straight after one is real and bounded.
* ``target_regs_per_interval`` is 1 and ``max_regs_per_block`` is 1.  Buying
  nine slots quickly compounds our own jumps and bids the price up against
  ourselves; one purchase every couple of hours lets our own impact decay
  several-fold first.

Waiting is cheap enough to be worth doing.  A ticket is worth about tau0.032 per
epoch (a nanobody epoch win pays ~tau1.34 at a field of ~42), which is
tau8.9e-5 per block of not holding it, while the decay saves ``price * 0.00192``
per block -- tau2.5e-4 at a price of tau0.13.  Waiting pays about 2.8x the
foregone mining revenue down to a break-even price near tau0.046
(:func:`wait_economics` computes this and every decision records it).  That is
the quantitative reason the price gate exists at all.

Layering:

* Everything above :func:`decide` is pure.  It takes a chain snapshot, a price
  history, and a spend window, and returns a :class:`Decision`.  It imports
  nothing outside the standard library -- ``bittensor`` is imported lazily
  inside the I/O helpers -- which is what makes the policy unit-testable with
  injected series and no network.
* Everything below it reads the chain, creates hotkeys, unlocks the coldkey and
  submits ``burned_register``.  That half is inert unless
  ``--execute-registrations`` is passed.  Dry run is the deployed default, so
  turning on live spending is a deliberate manual act by the operator and can
  never be a deployment side effect; running with no arguments at all cannot
  spend.

The agent runs on the operator's Mac -- the only machine the coldkey is allowed
to exist on, which is why it is deliberately absent from every GPU host -- as
``python -m nanobody_opt.slot_manager``, single-shot or with ``--interval``.

``--help`` is answered from argparse alone: no import of ``bittensor``, no
Keychain lookup, no wallet load and no ``Subtensor`` construction happens at
import time or during argument parsing, so a deploy gate can interrogate the
CLI on a machine that has none of them.

The coldkey password is never handled by this module as a value that can be
seen.  :func:`_unlock_coldkey` shells out to the macOS Keychain, captures the
password on a pipe straight into a local variable, hands it to the keyfile, and
drops it.  The only credential-adjacent string this module will ever print is
the *name of the environment variable* that names the Keychain service.
"""
from __future__ import annotations

import argparse
import json
import os
import contextlib
import io
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Measured chain constants.  Each one is an observation, not a derivation.
# ---------------------------------------------------------------------------

#: ``adjustment_interval`` as read from the subnet hyperparameters.
ADJUSTMENT_INTERVAL = 360
#: ``epoch = block // 361`` on this subnet.  Used for logging and ticket EV.
EPOCH_LENGTH = 361
#: ``immunity_period``: 5000 blocks, 13.85 epochs, 16.7 hours.
IMMUNITY_PERIOD = 5000
#: 12-second blocks.
BLOCKS_PER_DAY = 7200
#: Recycle decay, written as the measured halving rather than a fitted decimal:
#: the price halves in exactly one adjustment interval.  Over nineteen blocks
#: this reproduces the observed tau0.135167 -> tau0.130312 to within 1e-6.
DECAY_PER_BLOCK = 0.5 ** (1.0 / ADJUSTMENT_INTERVAL)
#: ss58 addresses are 47-49 characters.  Kept in step with
#: ``nanobody_opt.fleet.MINIMUM_SS58_LENGTH``; duplicated rather than imported so
#: the decision core keeps importing with the standard library alone.
MINIMUM_SS58_LENGTH = 40
#: Expected value of holding one registered ticket for one epoch, in tau.
#: 155.6 alpha/epoch * 40% track share * ~tau0.0215/alpha ~= tau1.34 per win,
#: spread over a field of about 42.
TICKET_TAO_PER_EPOCH = 0.032

# Decision actions.
BUY = "buy"
WAIT = "wait"

# Reason codes.  Every skip records one, so the log says why we did not buy.
REASON_BUY = "below_target_at_decay_floor"
REASON_BUY_URGENT = "below_target_and_short_of_a_fleet"
REASON_TARGET_MET = "target_met"
REASON_OWN_INTERVAL = "already_registered_this_interval"
REASON_STAGGER = "inside_stagger_window"
REASON_NO_HISTORY = "insufficient_price_history"
REASON_ABOVE_FLOOR = "price_above_decay_floor"
REASON_CEILING = "price_above_recycle_ceiling"
REASON_BUDGET = "daily_budget_would_be_exceeded"


def default_stagger_blocks(target: int) -> int:
    """Spacing that spreads ``target`` immunity expiries evenly, in blocks.

    Nine slots bought together lose immunity at the same block and all need
    renewal at once -- a self-inflicted price spike every 16.7 hours, paid for
    by us.  Spacing purchases ``immunity_period // target`` apart puts the
    expiries at uniform intervals around the immunity clock instead.  At the
    default target that is 555 blocks, about 1.9 hours, which also lets one of
    our own +33% jumps decay roughly 2.9x before the next buy.
    """
    if target < 1:
        raise ValueError("target must be at least one hotkey")
    return max(1, IMMUNITY_PERIOD // target)


@dataclass(frozen=True)
class PriceSample:
    """One poll of the recycle price."""

    block: int
    recycle_tao: float


@dataclass(frozen=True)
class ChainState:
    """What one poll of the chain tells us, reduced to what the policy needs."""

    block: int
    recycle_tao: float
    registrations_this_interval: int
    registered_uids: tuple[int, ...] = ()
    #: Block at which *we* most recently registered anything, from the
    #: metagraph's ``block_at_registration`` for our uids, unioned with our own
    #: decision log.  ``None`` when we hold nothing and have bought nothing.
    last_registration_block: int | None = None

    @property
    def registered(self) -> int:
        return len(self.registered_uids)

    @property
    def interval(self) -> int:
        return self.block // ADJUSTMENT_INTERVAL

    @property
    def epoch(self) -> int:
        return self.block // EPOCH_LENGTH


@dataclass(frozen=True)
class SpendRecord:
    """An executed purchase, for the rolling daily budget."""

    timestamp: float
    recycle_tao: float


@dataclass(frozen=True)
class SlotPolicy:
    """Every knob the decision takes, so a decision is reproducible from a log."""

    target: int = 7
    max_recycle_tao: float = 0.20
    #: Below this many held hotkeys, take any affordable price rather than wait
    #: for a decay floor.  A forfeited epoch costs more than a dear registration:
    #: the floor saves ~tau0.03, a missing ticket forfeits ~tau0.032 of expected
    #: value every epoch, and there are ~20 epochs a day.
    urgent_below: int = 3
    max_spend_tao_per_day: float = 1.0
    #: ``None`` means :func:`default_stagger_blocks` of the target.
    stagger_blocks: int | None = None
    #: How far back the decay floor looks.  Deliberately finite: a competitor
    #: spike must delay us, not lock us out.  Once the spike leaves the window
    #: the floor re-anchors and buying resumes.
    lookback_blocks: int = 180
    min_history_samples: int = 3
    min_history_span_blocks: int = 30
    #: How far above the projected floor the live price may sit and still count
    #: as "at the floor".  Absorbs poll jitter and decay-fit error.
    floor_tolerance: float = 0.05
    decay_per_block: float = DECAY_PER_BLOCK
    ticket_tao_per_epoch: float = TICKET_TAO_PER_EPOCH

    def __post_init__(self) -> None:
        if self.target < 1:
            raise ValueError("target must be at least one hotkey")
        if self.max_recycle_tao <= 0:
            raise ValueError("recycle ceiling must be positive")
        if self.max_spend_tao_per_day <= 0:
            raise ValueError("daily budget must be positive")
        if not 0.0 < self.decay_per_block < 1.0:
            raise ValueError("decay per block must lie strictly inside (0, 1)")
        if self.floor_tolerance < 0.0:
            raise ValueError("floor tolerance cannot be negative")
        if self.lookback_blocks < 1 or self.min_history_samples < 2:
            raise ValueError("the decay floor needs a real window to be measured in")

    @property
    def stagger(self) -> int:
        if self.stagger_blocks is None:
            return default_stagger_blocks(self.target)
        return max(0, self.stagger_blocks)


@dataclass(frozen=True)
class Decision:
    """What to do, and the numbers that justified it."""

    action: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.action == BUY


def wait_economics(price_tao: float, policy: SlotPolicy) -> dict[str, float]:
    """Per-block value of waiting one more block, against the cost of waiting.

    The saving is the decay on the price we are about to pay; the cost is one
    ticket-epoch of mining EV prorated to a block.  Below the break-even price
    waiting stops paying and buying immediately is correct.  This is advisory:
    it is recorded on every decision so the log carries the calibration that
    the thresholds were chosen against, but it does not gate anything -- the
    hard constraint is holding the target, not minimising spend.
    """
    saving = price_tao * (1.0 - policy.decay_per_block)
    foregone = policy.ticket_tao_per_epoch / EPOCH_LENGTH
    return {
        "decay_saving_tao_per_block": saving,
        "foregone_ticket_tao_per_block": foregone,
        "wait_value_ratio": saving / foregone if foregone > 0 else float("inf"),
        "breakeven_recycle_tao": foregone / (1.0 - policy.decay_per_block),
    }


def projected_floor_tao(
    history: Sequence[PriceSample], block: int, policy: SlotPolicy
) -> float | None:
    """Lowest price the current block could hold if nothing had registered.

    Each sample in the lookback window is carried forward to ``block`` at the
    measured decay, giving one estimate of "where the price would be now had it
    only decayed since then".  The minimum over the window is the floor.

    This is what separates a decayed price from a jumped one.  After a
    competitor registers, the samples taken *before* the jump still project to
    a low value, so the live price sits visibly above the floor and the gate
    holds.  Samples taken after the jump project to the jumped level, so once
    the pre-jump samples age out of ``lookback_blocks`` the floor re-anchors --
    a spike costs us a bounded delay rather than a permanent lockout.

    Returns ``None`` when the window is too thin to mean anything, which is the
    correct answer on the first poll after a restart.
    """
    window = [
        sample
        for sample in history
        if 0 <= block - sample.block <= policy.lookback_blocks
    ]
    if len(window) < policy.min_history_samples:
        return None
    span = max(s.block for s in window) - min(s.block for s in window)
    if span < policy.min_history_span_blocks:
        return None
    return min(
        sample.recycle_tao * policy.decay_per_block ** (block - sample.block)
        for sample in window
    )


def spend_in_window(
    spends: Iterable[SpendRecord], now: float, window_seconds: float = 86400.0
) -> float:
    """Executed tau spent inside the trailing window.  Dry runs never count."""
    return sum(
        record.recycle_tao
        for record in spends
        if now - record.timestamp <= window_seconds
    )


def decide(
    state: ChainState,
    history: Sequence[PriceSample],
    spends: Sequence[SpendRecord],
    policy: SlotPolicy,
    *,
    now: float | None = None,
) -> Decision:
    """Buy or wait, from a chain snapshot and our own history.

    ``history`` should already contain the current sample; the floor is
    computed from it while the live price is taken from ``state`` so the two
    cannot drift apart.

    The gates, in the order they are reported:

    1. we are at or above target -- nothing to do;
    2. we already registered inside this adjustment interval -- never more than
       one of our registrations per interval, whatever the price does;
    3. we are inside the stagger window -- buying now would bunch an immunity
       expiry against one we already hold;
    4. the price is above the hard recycle ceiling;
    5. the purchase would breach the rolling daily budget;
    6. the price history is too thin to locate the floor;
    7. the price sits above the projected decay floor.

    Gates 6 and 7 are a REFINEMENT and are skipped while we hold fewer than
    ``policy.urgent_below`` hotkeys.  Waiting for a decay floor saves at most
    about a third of a roughly tau0.10 registration -- call it tau0.03 -- while
    every epoch spent below a full fleet forfeits ~tau0.032 of expected value
    per missing ticket, and there are about twenty epochs a day.  Haggling is
    only rational once most of the fleet is already held.

    There is deliberately no gate on ``RegistrationsThisInterval``.  It was one
    of the first things tried and it is a proxy for "the price is elevated",
    which is worth nothing when the price itself is right there in the same
    snapshot.  Worse, it does not clear: the counter sits at its value for the
    whole 360-block interval, and this field routinely registers three times an
    interval, so a "buy only when the field is quiet" rule spends most of its
    life refusing -- observed refusing tau0.092 against a tau0.20 ceiling.  A
    purchaser that never purchases is the expensive failure here, not a slightly
    dear registration.
    """
    now = time.time() if now is None else now
    shortfall = policy.target - state.registered
    floor = projected_floor_tao(history, state.block, policy)
    spent = spend_in_window(spends, now)
    blocks_since_ours = (
        None
        if state.last_registration_block is None
        else state.block - state.last_registration_block
    )
    detail: dict[str, Any] = {
        "block": state.block,
        "epoch": state.epoch,
        "interval": state.interval,
        "recycle_tao": state.recycle_tao,
        "registrations_this_interval": state.registrations_this_interval,
        "registered": state.registered,
        "registered_uids": list(state.registered_uids),
        "target": policy.target,
        "shortfall": shortfall,
        "projected_floor_tao": floor,
        "floor_ratio": (
            None if floor in (None, 0) else state.recycle_tao / floor
        ),
        "floor_tolerance": policy.floor_tolerance,
        "blocks_since_our_last_registration": blocks_since_ours,
        "stagger_blocks": policy.stagger,
        "max_recycle_tao": policy.max_recycle_tao,
        "spent_tao_last_day": spent,
        "max_spend_tao_per_day": policy.max_spend_tao_per_day,
        "budget_headroom_tao": policy.max_spend_tao_per_day - spent,
        **wait_economics(state.recycle_tao, policy),
    }

    def verdict(action: str, reason: str) -> Decision:
        return Decision(action=action, reason=reason, detail=detail)

    if shortfall <= 0:
        return verdict(WAIT, REASON_TARGET_MET)
    if (
        state.last_registration_block is not None
        and state.last_registration_block // ADJUSTMENT_INTERVAL == state.interval
    ):
        return verdict(WAIT, REASON_OWN_INTERVAL)
    if blocks_since_ours is not None and blocks_since_ours < policy.stagger:
        return verdict(WAIT, REASON_STAGGER)
    # Hard rails first: both refuse absolutely and cannot be traded away.
    if state.recycle_tao > policy.max_recycle_tao:
        return verdict(WAIT, REASON_CEILING)
    if spent + state.recycle_tao > policy.max_spend_tao_per_day:
        return verdict(WAIT, REASON_BUDGET)
    # Below a working fleet, an affordable price beats a perfect one.
    if state.registered < policy.urgent_below:
        return verdict(BUY, REASON_BUY_URGENT)
    if floor is None:
        return verdict(WAIT, REASON_NO_HISTORY)
    if state.recycle_tao > floor * (1.0 + policy.floor_tolerance):
        return verdict(WAIT, REASON_ABOVE_FLOOR)
    return verdict(BUY, REASON_BUY)


# ---------------------------------------------------------------------------
# Persisted decision log.  It is both the audit trail and the agent's memory:
# the price history that the decay floor needs, the executed spend that the
# budget needs, and the last purchase block that the stagger needs all come
# back out of it, so single-shot runs accumulate state exactly like a loop.
# ---------------------------------------------------------------------------


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line durably.  The log is the only thing we trust."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path: Path) -> list[dict[str, Any]]:
    """Every well-formed line of the log.  A torn last line is skipped."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def price_history(records: Iterable[dict[str, Any]]) -> list[PriceSample]:
    """Price samples recovered from the log, oldest first and de-duplicated."""
    by_block: dict[int, float] = {}
    for record in records:
        try:
            block = int(record["block"])
            price = float(record["recycle_tao"])
        except (KeyError, TypeError, ValueError):
            continue
        by_block[block] = price
    return [PriceSample(block, by_block[block]) for block in sorted(by_block)]


def executed_spends(records: Iterable[dict[str, Any]]) -> list[SpendRecord]:
    """Purchases that actually hit the chain.  Dry runs are not spending."""
    spends = []
    for record in records:
        if record.get("action") != BUY or not record.get("executed"):
            continue
        try:
            spends.append(
                SpendRecord(
                    timestamp=float(record["time"]),
                    recycle_tao=float(record["recycle_tao"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return spends


def last_executed_block(records: Iterable[dict[str, Any]]) -> int | None:
    """Block of our most recent executed purchase, per the log."""
    blocks = [
        int(record["block"])
        for record in records
        if record.get("action") == BUY
        and record.get("executed")
        and isinstance(record.get("block"), int)
    ]
    return max(blocks) if blocks else None


# ---------------------------------------------------------------------------
# Chain and wallet I/O.  Nothing above this line touches either.
# ---------------------------------------------------------------------------


def _import_bittensor():
    try:
        # Deliberately lazy: see the layering note in the module docstring.
        import bittensor
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "bittensor is not importable in this environment; the decision "
            "core runs without it but chain reads and registration need it"
        ) from error
    return bittensor


def _scalar(value: Any) -> Any:
    return getattr(value, "value", value)


def _recycle_tao(subtensor: Any, netuid: int) -> float:
    """Current recycle price in tau, across SDK spellings."""
    for name in ("recycle", "burn"):
        accessor = getattr(subtensor, name, None)
        if accessor is None:
            continue
        balance = accessor(netuid)
        return float(getattr(balance, "tao", balance))
    raw = _scalar(subtensor.query_subtensor("Burn", params=[netuid]))
    return float(raw) / 1e9


#: Alpha holdings below this are not worth a manual conversion.
DEFAULT_ALPHA_REPORT_FLOOR = 1.0


def alpha_holdings(subtensor: Any, netuid: int, coldkey: str) -> dict[str, Any]:
    """Every alpha position this coldkey holds on the subnet, and its value.

    Asked of the stake map, not the metagraph. A metagraph walk keeping rows
    whose coldkey is ours only sees alpha sitting on hotkeys we registered,
    and that is not where the treasury is: on 2026-09-13 our own miner hotkeys
    held 0.000000 while the coldkey held 47.3 alpha on this subnet delegated to
    three hotkeys we do not operate -- taostats files exactly that as "other
    alpha". A reporter blind to it reports an empty treasury.

    Read-only. The agent reports the position and the command that would
    convert it; the conversion itself is executed by the operator, because an
    unattended process holding the coldkey should not be able to move the
    treasury on its own.
    """
    accessor = getattr(subtensor, "get_stake_info_for_coldkey", None)
    if accessor is None:
        raise RuntimeError(
            "this bittensor SDK exposes no per-coldkey stake map; the "
            "metagraph alone cannot see delegated alpha"
        )
    entries = accessor(coldkey)
    if not isinstance(entries, list):
        entries = [entries]
    holdings: list[dict[str, Any]] = []
    total = 0.0
    for entry in entries:
        if int(getattr(entry, "netuid", -1)) != int(netuid):
            continue
        stake = getattr(entry, "stake", None)
        amount = float(getattr(stake, "tao", stake) or 0.0)
        if amount <= 0:
            continue
        hotkey = str(getattr(entry, "hotkey_ss58", ""))
        total += amount
        holdings.append({"hotkey": hotkey, "alpha": amount})
    holdings.sort(key=lambda item: item["alpha"], reverse=True)
    return {"holdings": holdings, "alpha_total": total}


def conversion_command(
    *, netuid: int, hotkey: str, alpha: float, price_tao: float,
    wallet_name: str, slippage: float,
) -> str:
    """The exact command the operator runs to convert alpha to tau.

    Rendered with a limit price rather than a bare market order, so a thin
    book cannot fill the whole position at a price nobody intended.
    """
    floor = price_tao * (1.0 - slippage)
    return (
        f"btcli stake remove --netuid {netuid} --wallet.name {wallet_name} "
        f"--wallet.hotkey {hotkey} --amount {alpha:.6f} "
        f"--tolerance {slippage:.3f} --price-limit {floor:.8f}"
    )


def treasury_report(
    *, holdings: dict[str, Any], price_tao: float, netuid: int,
    wallet_name: str, slippage: float, floor_alpha: float,
) -> dict[str, Any]:
    """What we hold, what it is worth, and the command to convert it."""
    total = float(holdings["alpha_total"])
    report = {
        "kind": "treasury",
        "alpha_total": total,
        "alpha_price_tao": price_tao,
        "value_tao": total * price_tao,
        "holdings": holdings["holdings"],
        "actionable": total >= floor_alpha,
        "commands": [],
    }
    if report["actionable"]:
        report["commands"] = [
            conversion_command(
                netuid=netuid, hotkey=item["hotkey"], alpha=item["alpha"],
                price_tao=price_tao, wallet_name=wallet_name, slippage=slippage,
            )
            for item in holdings["holdings"] if item["alpha"] > 0
        ]
    return report


def _registrations_this_interval(subtensor: Any, netuid: int) -> int:
    raw = _scalar(subtensor.query_subtensor("RegistrationsThisInterval", params=[netuid]))
    return int(raw)


def read_chain_state(
    subtensor: Any, netuid: int, coldkey_ss58: str, *, log_last_block: int | None = None
) -> ChainState:
    """One poll, reduced to a :class:`ChainState`.

    Our registrations are discovered by coldkey off the metagraph rather than
    by remembering which hotkeys we created: a hotkey we made and never
    registered, or one that was deregistered while we were not looking, must
    not be counted.  ``block_at_registration`` for those uids, unioned with the
    decision log, is what the stagger and one-per-interval gates run on.
    """
    metagraph = subtensor.metagraph(netuid)
    coldkeys = list(getattr(metagraph, "coldkeys", []) or [])
    if not coldkeys:
        # Fail closed before anything downstream can read "no neurons" as
        # "we hold nothing" and buy a slot we already own.
        raise EmptyRosterError(
            f"metagraph for netuid {netuid} exposed no coldkeys; the subnet is "
            "full, so this is a failed read and not an empty fleet"
        )
    ours = tuple(
        uid for uid, coldkey in enumerate(coldkeys) if coldkey == coldkey_ss58
    )
    registration_blocks = list(getattr(metagraph, "block_at_registration", []) or [])
    ours_blocks = [
        int(registration_blocks[uid]) for uid in ours if uid < len(registration_blocks)
    ]
    if log_last_block is not None:
        ours_blocks.append(int(log_last_block))
    return ChainState(
        block=int(subtensor.get_current_block()),
        recycle_tao=_recycle_tao(subtensor, netuid),
        registrations_this_interval=_registrations_this_interval(subtensor, netuid),
        registered_uids=ours,
        last_registration_block=max(ours_blocks) if ours_blocks else None,
    )


class EmptyRosterError(RuntimeError):
    """Raised when the metagraph exposed no neurons at all.

    Subnet 68 is full -- ``MaxAllowedUids`` 64, ``n`` 64 -- so a coldkeys column
    of length zero is a failed read, never a fact about the field.  Read as fact
    it means "we hold nothing", which is below ``urgent_below`` and therefore an
    immediate purchase at whatever the price happens to be.  The miner's
    chain-state exporter refuses an empty roster for exactly this reason; the
    component that spends tau must refuse it at least as hard.
    """


def _hotkey_names(wallet_name: str) -> list[str]:
    """Hotkey names in the wallet.  Names only -- keyfiles are never opened."""
    directory = Path.home() / ".bittensor" / "wallets" / wallet_name / "hotkeys"
    if not directory.is_dir():
        return []
    # iterdir() only ever yields names here; no hotkey file is opened.
    return sorted(entry.name for entry in directory.iterdir() if entry.is_file())


def next_hotkey_name(existing: Sequence[str], prefix: str) -> str:
    """First unused ``prefix-N``.  Deterministic so a retry reuses the name."""
    taken = set(existing)
    index = 1
    while f"{prefix}-{index}" in taken:
        index += 1
    return f"{prefix}-{index}"


SECRET_SOURCES = ("auto", "keychain", "systemd")


def _default_secret_source() -> str:
    """macOS has a Keychain; a headless Raspberry Pi does not."""
    return "keychain" if sys.platform == "darwin" else "systemd"


def _password_from_keychain(service_env: str) -> str:
    """Read the password from the macOS Keychain, by service name only."""
    service = os.environ.get(service_env, "").strip()
    if not service:
        raise SystemExit(
            f"set {service_env} to the Keychain service name holding the "
            "coldkey password; it is never read from a flag or a file"
        )
    completed = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        # completed.stderr is discarded unread: security(1) repeats the query
        # back in its diagnostics.
        raise SystemExit(f"no Keychain password for the service named by {service_env}")
    password = completed.stdout.rstrip("\n")
    if not password:
        raise SystemExit(f"Keychain item named by {service_env} is empty")
    return password


def _password_from_systemd_credential(name_env: str) -> str:
    """Read the password from a systemd-supplied encrypted credential.

    This is the store for an always-on headless node such as a Raspberry Pi,
    where there is no Keychain and no unlocked desktop keyring to talk to.  The
    operator encrypts the password once with ``systemd-creds encrypt``, bound to
    that host; systemd decrypts it at service start into ``$CREDENTIALS_DIRECTORY``
    on a tmpfs, mode 0400, owned by the service user.

    It is deliberately NOT passed as an environment variable: the whole point of
    LoadCredential over Environment= is that the value never appears in the unit
    file, in ``systemctl show``, in ``/proc/<pid>/environ``, or in a core dump of
    a child process.  Only the credential's *name* is configuration.
    """
    name = os.environ.get(name_env, "").strip()
    if not name:
        raise SystemExit(
            f"set {name_env} to the systemd credential name holding the coldkey "
            "password (see LoadCredentialEncrypted= in the unit)"
        )
    directory = os.environ.get("CREDENTIALS_DIRECTORY", "").strip()
    if not directory:
        raise SystemExit(
            "CREDENTIALS_DIRECTORY is unset: this process was not started by "
            "systemd with LoadCredentialEncrypted=, so the credential is not "
            "available. Run it under the unit rather than by hand."
        )
    path = Path(directory) / name
    if not path.is_file():
        raise SystemExit(f"no systemd credential named by {name_env} in the credentials directory")
    try:
        password = path.read_text(encoding="utf-8").rstrip("\n")
    except OSError as error:
        raise SystemExit(f"systemd credential unreadable: {type(error).__name__}") from None
    if not password:
        raise SystemExit(f"systemd credential named by {name_env} is empty")
    return password


def _coldkey_password(source: str, service_env: str) -> str:
    """Fetch the password from whichever store this host actually has."""
    if source == "auto":
        source = _default_secret_source()
    if source == "keychain":
        return _password_from_keychain(service_env)
    if source == "systemd":
        return _password_from_systemd_credential(service_env)
    raise SystemExit(f"unknown credential source: {source}")


def _unlock_coldkey(wallet: Any, service_env: str, source: str = "auto") -> None:
    """Decrypt the coldkey with the password held in the host's secret store.

    The password exists in this process as one local ``str`` for the duration
    of one ``decrypt`` call and nowhere else.  It is never an argument, never
    an environment variable, never written down, and never formatted into a
    message -- including an error message, which is why the original exception
    is dropped rather than chained: a keyfile library is entitled to put the
    attempted password in its own text, and that text would land in the log.

    The only string here that may ever be printed is ``service_env``, the
    *name* of the environment variable that names the item in the store --
    a Keychain service on macOS, a systemd credential on a headless node.
    """
    keyfile = wallet.coldkey_file
    if not keyfile.is_encrypted():
        return
    password = _coldkey_password(source, service_env)
    try:
        keyfile.decrypt(password)
    except Exception as error:  # noqa: BLE001 -- the class is the whole point
        raise SystemExit(f"coldkey unlock failed: {type(error).__name__}") from None
    finally:
        # Best effort: CPython offers no way to wipe a str, but dropping the
        # only reference keeps it out of tracebacks and locals dumps.
        del password


def _create_hotkey_quietly(wallet: Any) -> None:
    """Create a hotkey without its mnemonic reaching stdout.

    bittensor prints the new key's twelve-word mnemonic on creation.  Under a
    systemd unit stdout is the journal, so the first live registration wrote a
    private key in clear text to a file that survives reboots and is readable by
    anyone who can read the journal.  That is a key-material leak, not noise.

    The output is discarded rather than filtered: a filter has to be right about
    every future version's wording, while a discarded stream cannot leak
    whatever the next release decides to print.  ``suppress=True`` is passed as
    well where the installed version understands it, so the mnemonic is not
    formatted in the first place.
    """
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                wallet.create_new_hotkey(
                    n_words=12, use_password=False, overwrite=False, suppress=True
                )
            except TypeError:
                # Older signatures have no `suppress`; the redirect still holds.
                wallet.create_new_hotkey(
                    n_words=12, use_password=False, overwrite=False
                )
    finally:
        # Drop the captured text unread. Nothing in it is worth a leak.
        buffer.close()


def _register(
    *,
    wallet_name: str,
    hotkey_name: str,
    netuid: int,
    network: str,
    keychain_service_env: str,
    credential_source: str = "auto",
) -> bool:
    """Create the hotkey if needed and burn-register it.

    Only ``--execute-registrations`` reaches this function, and nothing in the
    test suite does.  The SDK spellings below (``Wallet``/``wallet``,
    ``Subtensor``/``subtensor``, ``Keyfile.decrypt``) have *not* been exercised
    against a live node from this checkout -- there is no GPU host and
    ``bittensor`` is not importable on this machine -- so the first live run
    should be a single ``--execute-registrations`` invocation watched by hand,
    not an unattended ``--interval`` loop.
    """
    bittensor = _import_bittensor()
    wallet_factory = getattr(bittensor, "Wallet", None) or bittensor.wallet
    wallet = wallet_factory(name=wallet_name, hotkey=hotkey_name)
    if hotkey_name not in _hotkey_names(wallet_name):
        _create_hotkey_quietly(wallet)
    _unlock_coldkey(wallet, keychain_service_env, credential_source)
    subtensor_factory = getattr(bittensor, "Subtensor", None) or bittensor.subtensor
    subtensor = subtensor_factory(network=network)
    return bool(
        subtensor.burned_register(
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
    )


def _policy_from_args(args: argparse.Namespace) -> SlotPolicy:
    return SlotPolicy(
        target=args.target,
        urgent_below=args.urgent_below,
        max_recycle_tao=args.max_recycle_tao,
        max_spend_tao_per_day=args.max_spend_tao_per_day,
        stagger_blocks=args.stagger_blocks,
        lookback_blocks=args.lookback_blocks,
        floor_tolerance=args.floor_tolerance,
    )


def _report(decision: Decision, *, executed: bool, hotkey: str | None) -> None:
    detail = decision.detail
    floor = detail.get("projected_floor_tao")
    print(
        f"SLOT_{decision.action.upper()} reason={decision.reason} "
        f"block={detail['block']} epoch={detail['epoch']} "
        f"recycle={detail['recycle_tao']:.6f} "
        f"floor={'na' if floor is None else format(floor, '.6f')} "
        f"regs_this_interval={detail['registrations_this_interval']} "
        f"registered={detail['registered']}/{detail['target']} "
        f"spent_24h={detail['spent_tao_last_day']:.4f}/"
        f"{detail['max_spend_tao_per_day']:.4f} "
        f"executed={str(executed).lower()} hotkey={hotkey or 'na'}",
        flush=True,
    )


def _preflight(args: argparse.Namespace) -> None:
    """Fail before touching the chain if live mode is missing its prerequisites.

    Only the *name* of the environment variable is inspected here, and only its
    emptiness: no Keychain lookup, no wallet load, no fallback service name, no
    prompt.  An unset variable is an operator mistake with a one-line fix, not
    something to go searching the Keychain for.
    """
    if not args.execute:
        return
    source = args.credential_source
    if source == "auto":
        source = _default_secret_source()
    store = "Keychain service" if source == "keychain" else "systemd credential"
    if not os.environ.get(args.keychain_service_env, "").strip():
        raise SystemExit(
            f"--execute-registrations needs {args.keychain_service_env} set to the "
            f"{store} name holding the coldkey password"
        )
    if source == "systemd" and not os.environ.get("CREDENTIALS_DIRECTORY", "").strip():
        # Fail here rather than after a chain round trip: without systemd having
        # decrypted the credential there is nothing to unlock the coldkey with.
        raise SystemExit(
            "--execute-registrations with the systemd credential source must run "
            "under the unit (CREDENTIALS_DIRECTORY is unset)"
        )


def _resolve_coldkey_ss58(bittensor: Any, wallet_name: str, given: str | None) -> str:
    """Our coldkey address, from the flag or the wallet's *public* key.

    ``coldkeypub`` is public material and is the only part of the wallet this
    module ever looks at.  The encrypted coldkey file is opened exactly once,
    by ``bittensor``, inside :func:`_unlock_coldkey`, and never read here.
    """
    if given:
        # A coldkey pasted from a dashboard arrives elided -- "5EkCwcNTfxeN..."
        # -- and an address that matches no neuron row is indistinguishable here
        # from holding nothing, which is below urgent_below and therefore a
        # purchase.  nanobody_opt.fleet refuses the same shape for the same
        # reason; there it only costs a refusal to publish, here it costs tau.
        candidate = str(given).strip()
        if (
            "…" in candidate
            or "..." in candidate
            or len(candidate) < MINIMUM_SS58_LENGTH
        ):
            raise SystemExit(
                f"--coldkey-ss58 {candidate!r} is truncated; a partial address "
                "matches no uid and would be read as an empty fleet, which buys"
            )
        return candidate
    wallet_factory = getattr(bittensor, "Wallet", None) or bittensor.wallet
    wallet = wallet_factory(name=wallet_name)
    return str(wallet.coldkeypub.ss58_address)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m nanobody_opt.slot_manager",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--wallet", default="mining_ops")
    parser.add_argument(
        "--coldkey-ss58",
        default=None,
        help="default: read from the wallet's public coldkeypub",
    )
    parser.add_argument("--hotkey-prefix", default="nova68-slot")
    parser.add_argument(
        "--minimum-hotkeys",
        "--target",
        dest="target",
        type=int,
        default=7,
        help="registrations to hold on subnet 68 (default: 7)",
    )
    parser.add_argument(
        "--decision-log",
        type=Path,
        default=Path.home() / ".nova" / "subnet68_slots.jsonl",
        help="append-only audit trail; also the agent's price and spend memory",
    )
    parser.add_argument("--max-recycle-tao", type=float, default=0.20)
    parser.add_argument("--max-spend-tao-per-day", type=float, default=1.0)
    parser.add_argument(
        "--stagger-blocks",
        type=int,
        default=None,
        help="default: immunity_period // target, which spreads expiries evenly",
    )
    parser.add_argument(
        "--urgent-below",
        type=int,
        default=3,
        help="below this many held hotkeys, buy at any affordable price rather "
        "than waiting for a decay floor",
    )
    parser.add_argument("--lookback-blocks", type=int, default=180)
    parser.add_argument("--floor-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--credential-source",
        choices=SECRET_SOURCES,
        default="auto",
        help="where the coldkey password lives: keychain (macOS), systemd "
        "(LoadCredentialEncrypted on a headless node), or auto by platform",
    )
    parser.add_argument(
        "--keychain-service-env",
        default="NOVA_COLDKEY_KEYCHAIN_SERVICE",
        help="name of the environment variable holding the Keychain service name",
    )
    parser.add_argument(
        "--interval",
        "--poll-seconds",
        dest="interval",
        type=float,
        default=0.0,
        help="watch-loop period in seconds; 0 evaluates once and exits "
        "(the decision log carries history between single-shot runs)",
    )
    parser.add_argument(
        "--execute-registrations",
        dest="execute",
        action="store_true",
        help="actually spend tau and register; omitting it -- which is the "
        "deployed default -- makes every purchase a logged dry run",
    )
    args = parser.parse_args(argv)
    policy = _policy_from_args(args)
    _preflight(args)
    # Nothing above this line imports bittensor, reads the Keychain, loads a
    # wallet or opens a connection, so --help works on a bare machine.
    bittensor = _import_bittensor()
    coldkey_ss58 = _resolve_coldkey_ss58(bittensor, args.wallet, args.coldkey_ss58)
    subtensor_factory = getattr(bittensor, "Subtensor", None) or bittensor.subtensor
    subtensor = subtensor_factory(network=args.network)

    while True:
        records = read_records(args.decision_log)
        try:
            state = read_chain_state(
                subtensor,
                args.netuid,
                coldkey_ss58,
                log_last_block=last_executed_block(records),
            )
        except EmptyRosterError as error:
            # A failed read is a reason to poll again, never a reason to buy.
            print(f"SLOT_CHAIN_READ_REFUSED {error}", flush=True)
            if args.interval <= 0:
                raise SystemExit(str(error)) from error
            time.sleep(args.interval)
            continue
        history = price_history(records) + [
            PriceSample(state.block, state.recycle_tao)
        ]
        decision = decide(
            state, history, executed_spends(records), policy, now=time.time()
        )
        hotkey = None
        executed = False
        if decision.is_buy:
            hotkey = next_hotkey_name(_hotkey_names(args.wallet), args.hotkey_prefix)
            if args.execute:
                executed = _register(
                    wallet_name=args.wallet,
                    hotkey_name=hotkey,
                    netuid=args.netuid,
                    network=args.network,
                    keychain_service_env=args.keychain_service_env,
                    credential_source=args.credential_source,
                )
        append_record(
            args.decision_log,
            {
                "time": time.time(),
                "action": decision.action,
                "reason": decision.reason,
                "executed": executed,
                "dry_run": not args.execute,
                "hotkey": hotkey,
                "netuid": args.netuid,
                "coldkey_ss58": coldkey_ss58,
                **decision.detail,
            },
        )
        _report(decision, executed=executed, hotkey=hotkey)
        if args.interval <= 0:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
