"""Per-subnet profit and loss, in TAO and USD per day.

Earned alpha is what the balance gained that was not transferred in as a
purchase.  Mining rewards arrive as alpha emission (``is_transfer`` stake
transfers are swept rewards, counted as income), so:

    earned_alpha = d(balance) + sold_alpha - bought_alpha
    bought_alpha counts only DELEGATE rows that are NOT stake transfers

Two corrections keep revenue from reading as a burn-down that does not exist:

* SN68's balance-to-balance alpha oscillates because the slot manager recycles
  its own registrations (an evicted uid's stake returns to free TAO and the
  re-burn moves it back).  That is a change of asset, not a mining loss, so a
  negative daily delta is floored to zero -- revenue is the emission, which
  cannot go negative.
* The registration burn (``burned_register``, ~2.5 TAO/day) is a real TAO cost
  of holding the slots -- the recycle TAO leaves the coldkey.  It is subtracted
  from net alongside infrastructure and shown in the slot-purchases panel.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import statistics as st

SUBNET_NAMES = {68: "SN68 NOVA", 2: "SN2 DSperse", 62: "SN62", 85: "SN85", 107: "SN107"}
#: A DELEGATE transfer-in above this many alpha is a one-time swept event (e.g.
#: an unrelated blueprint win) rather than the normal compound/nanobody reward
#: churn; it is treated as not-earned so it can't spike the daily yield.
LARGE_TRANSFER_ALPHA = 1000.0


def _block_clock(connection: sqlite3.Connection):
    """Return a block -> UTC datetime mapper from observed chain timestamps.

    Historical balance snapshots were fetched together, so their ``sampled_at``
    values describe collection time rather than the block's day.  Delegations
    and slot decisions carry both a block and a real timestamp.  The median
    ``timestamp - block*12`` offset gives every balance sample its actual day
    without trusting one potentially delayed API row.
    """
    offsets: list[float] = []
    for block, stamp in connection.execute(
        "SELECT block, timestamp FROM delegations "
        "WHERE block IS NOT NULL AND timestamp IS NOT NULL"
    ):
        try:
            when = dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
            offsets.append(when.timestamp() - int(block) * 12)
        except (TypeError, ValueError, OSError):
            continue
    for block, stamp in connection.execute(
        "SELECT block, time FROM slots WHERE block IS NOT NULL AND time IS NOT NULL"
    ):
        try:
            offsets.append(float(stamp) - int(block) * 12)
        except (TypeError, ValueError):
            continue
    offset = st.median(offsets) if offsets else 0.0

    def at(block: int) -> dt.datetime:
        return dt.datetime.fromtimestamp(int(block) * 12 + offset, dt.timezone.utc)

    return at


def subnet_pnl_history(connection: sqlite3.Connection,
                       tao_usd: float) -> list[dict]:
    """One true balance-to-balance daily observation for every subnet.

    Each interval uses only flows inside ``(start_block, end_block]``.  Reward
    stake transfers remain earned balance; alpha purchases are removed and
    sales are added back, matching :func:`subnet_pnl` without smearing eight
    days into one average.
    """
    balances: dict[int, list[tuple[int, float, float, float]]] = {}
    for block, netuid, alpha, price, free in connection.execute(
        "SELECT block, netuid, alpha, alpha_price_tao, free_tao "
        "FROM balances ORDER BY netuid, block"
    ):
        balances.setdefault(int(netuid), []).append(
            (int(block), float(alpha or 0), float(price or 0), float(free or 0))
        )

    flows: dict[int, list[tuple[int, str, float, int]]] = {}
    for netuid, action, alpha, block, transfer in connection.execute(
        "SELECT netuid, action, alpha, block, is_transfer FROM delegations "
        "WHERE netuid IS NOT NULL AND block IS NOT NULL"
    ):
        flows.setdefault(int(netuid), []).append(
            (int(block), str(action), float(alpha or 0), int(bool(transfer)))
        )
    purchases = [
        (int(block), float(cost if cost is not None else recycle or 0))
        for block, cost, recycle in connection.execute(
            "SELECT block, decision_recycle_tao, recycle_tao FROM slots "
            "WHERE executed = 1 AND block IS NOT NULL"
        )
    ]
    block_time = _block_clock(connection)
    out: list[dict] = []
    for netuid, series in balances.items():
        if netuid != 68:
            continue  # SN68 (compound + nanobody) only; skip SN2/SN62/SN85/SN107 noise
        for before, after in zip(series, series[1:]):
            b0, a0, _, _ = before
            b1, a1, price, free = after
            elapsed_days = (b1 - b0) / 7200.0
            if elapsed_days <= 0:
                continue
            bought = sold = 0.0
            for block, action, alpha, transfer in flows.get(netuid, []):
                if not b0 < block <= b1:
                    continue
                if action == "UNDELEGATE":
                    sold += alpha
                elif action == "DELEGATE" and (
                        not transfer or alpha > LARGE_TRANSFER_ALPHA):
                    bought += alpha
            earned_alpha = (a1 - a0) + sold - bought
            # This is the alpha-stake delta (a "stake" view, not the headline):
            # SN68's stake oscillates because the slot manager recycles its own
            # registrations (an evicted uid's stake returns to free TAO).  The
            # portfolio (free + staked) is the real income and lives in
            # daily_money_history.
            gross_tao = earned_alpha * price / elapsed_days
            # The registration burn is a real TAO cost of holding the slots
            # (the TAO leaves the coldkey), so it is subtracted from net.
            registration = 0.0
            if netuid == 68:
                registration = sum(
                    cost for block, cost in purchases if b0 < block <= b1
                ) / elapsed_days
            net_tao = gross_tao - registration
            out.append(dict(
                day=block_time(b1).date().isoformat(), block_start=b0, block_end=b1,
                netuid=netuid, name=SUBNET_NAMES.get(netuid, f"SN{netuid}"),
                elapsed_days=elapsed_days, earned_alpha=earned_alpha,
                gross_tao_per_day=gross_tao,
                registration_tao_per_day=registration,
                net_tao_per_day=net_tao,
                gross_usd_per_day=gross_tao * tao_usd,
                net_usd_per_day=net_tao * tao_usd,
                holding_tao=a1 * price, free_tao=free,
            ))
    return sorted(out, key=lambda row: (row["day"], row["netuid"]))


def daily_money_history(connection: sqlite3.Connection, tao_usd: float) -> list[dict]:
    """Daily mining income vs cost, per day.

    Income is the stake transferred in from our mining hotkeys (DELEGATE
    transfer-in), excluding one-time large swept events such as the blueprint
    win.  Costs are the registration burn (TAO) plus the fiat infra (AWS/Vast,
    from the costs table).  TAO or alpha moved out to *fund* infra is funding,
    not a cost, so it is not subtracted -- only the actual AWS/Vast/registration
    spend is.  This replaces the old portfolio-delta view, which wrongly read
    infra funding and mark-to-market as losses.
    """
    clock = _block_clock(connection)

    income_by_day: dict[str, float] = {}
    for block, action, alpha, tao, transfer in connection.execute(
        "SELECT block, action, alpha, tao, is_transfer FROM delegations "
        "WHERE block IS NOT NULL"
    ):
        day = clock(block).date().isoformat()
        if action == "DELEGATE" and transfer and (alpha or 0.0) <= LARGE_TRANSFER_ALPHA:
            income_by_day[day] = income_by_day.get(day, 0.0) + (tao or 0.0)

    reg_by_day: dict[str, float] = {}
    for block, cost, recycle in connection.execute(
        "SELECT block, decision_recycle_tao, recycle_tao FROM slots "
        "WHERE executed = 1 AND block IS NOT NULL"
    ):
        day = clock(block).date().isoformat()
        reg_by_day[day] = reg_by_day.get(day, 0.0) + (
            float(cost) if cost is not None else float(recycle or 0.0))

    infra_by_day: dict[str, float] = {}
    infra_items_by_day: dict[str, dict[str, float]] = {}
    for day, item, usd in connection.execute(
        "SELECT day, item, COALESCE(usd, 0) FROM costs ORDER BY day, item"
    ):
        key = str(day)
        infra_by_day[key] = infra_by_day.get(key, 0.0) + float(usd or 0.0)
        infra_items_by_day.setdefault(key, {})[str(item)] = float(usd or 0.0)

    # API silence is not zero income. A successful complete query is the
    # coverage watermark; an event timestamp alone does not prove a full day.
    try:
        sync = connection.execute(
            "SELECT through_at,ok FROM economics_sync WHERE source='delegations'").fetchone()
    except sqlite3.OperationalError:
        sync = None
    if sync and sync[0]:
        through_day = dt.datetime.fromtimestamp(sync[0], dt.timezone.utc).date().isoformat()
    else:
        latest = connection.execute("SELECT MAX(timestamp) FROM delegations").fetchone()[0]
        through_day = str(latest or "")[:10]
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()

    # Keep the last sample of each position, not the sum of intraday snapshots.
    # A missing portfolio observation remains unknown instead of becoming zero.
    portfolio_by_day: dict[str, dict] = {}
    for block, netuid, alpha, price, free in connection.execute(
        "SELECT block, netuid, alpha, alpha_price_tao, free_tao FROM balances "
        "ORDER BY block"
    ):
        day = clock(block).date().isoformat()
        p = portfolio_by_day.setdefault(day, {"free": 0.0, "positions": {}})
        p["free"] = float(free or 0.0)
        p["positions"][netuid] = float(alpha or 0.0) * float(price or 0.0)

    days = sorted(set(income_by_day) | set(reg_by_day)
                  | set(infra_by_day) | set(portfolio_by_day))
    out: list[dict] = []
    for day in days:
        income = income_by_day.get(day, 0.0)
        reg = reg_by_day.get(day, 0.0)
        infra_usd = infra_by_day.get(day, 0.0)
        covered = bool(through_day and day < through_day and day < today)
        net_tao = income - reg - (infra_usd / tao_usd) if covered else None
        p = portfolio_by_day.get(day)
        staked = sum(p["positions"].values()) if p is not None else None
        out.append({
            "day": day,
            "gross_tao": income,
            "registration_tao": reg,
            "net_tao": net_tao,
            "gross_usd": income * tao_usd,
            "infra_usd": infra_usd,
            "net_usd": net_tao * tao_usd if net_tao is not None else None,
            "income_status": "recorded" if covered else "incomplete",
            "rewards_through": through_day or None,
            "registration_usd": reg * tao_usd,
            "infrastructure": infra_items_by_day.get(day, {}),
            "holding_tao": staked,
            "free_tao": p["free"] if p is not None else None,
            "portfolio_tao": p["free"] + staked if p is not None else None,
        })
    return out


def infrastructure_cost_history(connection: sqlite3.Connection) -> list[dict]:
    """Every recorded operational infrastructure charge, grouped by UTC day."""
    by_day: dict[str, dict] = {}
    for day, item, usd in connection.execute(
        "SELECT day, item, COALESCE(usd, 0) FROM costs ORDER BY day, item"
    ):
        bucket = by_day.setdefault(str(day), {
            "day": str(day), "total_usd": 0.0, "infrastructure": {},
        })
        amount = float(usd or 0)
        bucket["total_usd"] += amount
        bucket["infrastructure"][str(item)] = amount
    return [by_day[day] for day in sorted(by_day)]


def subnet_pnl(connection: sqlite3.Connection, tao_usd: float,
               registration_tao_per_day: float = 0.0) -> list[dict]:
    """One row per subnet: earned alpha, TAO/day, USD/day, and registration cost."""
    balances: dict[int, list] = {}
    for block, netuid, alpha, price, free, _ in connection.execute(
            "SELECT block, netuid, alpha, alpha_price_tao, free_tao, sampled_at "
            "FROM balances ORDER BY block"):
        balances.setdefault(netuid, []).append((block, alpha, price))

    # Flows MUST be windowed to the same blocks as the balance samples.
    # Summing all-time undelegations against an 8-day balance delta reported
    # 64 TAO/day of yield on a 19 TAO position.
    span = {n: (s[0][0], s[-1][0]) for n, s in balances.items() if len(s) >= 2}
    flows: dict[int, dict] = {}
    for netuid, action, alpha, tao, block, transfer in connection.execute(
            "SELECT netuid, action, alpha, tao, block, is_transfer FROM delegations"):
        window = span.get(netuid)
        if window is None or block is None or not (window[0] <= block <= window[1]):
            continue
        f = flows.setdefault(netuid, dict(bought=0.0, sold=0.0, sold_tao=0.0))
        if action == "UNDELEGATE":
            f["sold"] += alpha or 0.0
            f["sold_tao"] += tao or 0.0
        elif action == "DELEGATE" and (
                not transfer or (alpha or 0.0) > LARGE_TRANSFER_ALPHA):
            # bought with TAO, or a one-time swept reward unrelated to mining
            f["bought"] += alpha or 0.0

    out = []
    for netuid, series in balances.items():
        if netuid != 68:
            continue  # SN68 (compound + nanobody) only
        if len(series) < 2:
            continue
        (b0, a0, _), (b1, a1, p1) = series[0], series[-1]
        days = max((b1 - b0) / 7200.0, 1e-6)
        f = flows.get(netuid, dict(bought=0.0, sold=0.0, sold_tao=0.0))
        earned_alpha = (a1 - a0) + f["sold"] - f["bought"]
        # This is the alpha-stake delta (a "stake" view, not the headline): a
        # negative SN68 delta is the recycle churn, not a mining loss.
        tao_day = earned_alpha * (p1 or 0.0) / days
        out.append(dict(
            netuid=netuid, name=SUBNET_NAMES.get(netuid, f"SN{netuid}"),
            days=round(days, 2), alpha_start=round(a0, 2), alpha_end=round(a1, 2),
            earned_alpha=round(earned_alpha, 2),
            earned_alpha_per_day=round(earned_alpha / days, 2),
            alpha_price_tao=p1, tao_per_day=round(tao_day, 4),
            usd_per_day=round(tao_day * tao_usd, 2),
            holding_tao=round(a1 * (p1 or 0.0), 3)))
    out.sort(key=lambda r: -r["tao_per_day"])
    # Registration burn is a real TAO cost of holding the slots, so net is
    # gross minus the registration burn for SN68.
    for row in out:
        cost = registration_tao_per_day if row["netuid"] == 68 else 0.0
        row["cost_tao_per_day"] = round(cost, 4)
        row["net_tao_per_day"] = round(row["tao_per_day"] - cost, 4)
        row["net_usd_per_day"] = round(row["net_tao_per_day"] * tao_usd, 2)
    return out


def portfolio_tao_per_day(connection: sqlite3.Connection) -> float:
    """Net TAO/day from total portfolio (free + staked), balance to balance.

    This is the number Taostats shows as the steady total-TAO trend: free TAO
    plus every subnet's staked alpha valued at its current price.  It cancels
    the SN68 slot-recycle churn (alpha <-> free TAO), so it reads as the
    genuine balance change rather than a fake -2.5 TAO/day burn-down.
    """
    rows = connection.execute(
        "SELECT block, netuid, alpha, alpha_price_tao, free_tao "
        "FROM balances ORDER BY block"
    ).fetchall()
    if not rows:
        return 0.0
    by_block: dict[int, dict[str, float]] = {}
    for block, netuid, alpha, price, free in rows:
        bucket = by_block.setdefault(int(block), {"free": 0.0, "staked": 0.0})
        bucket["free"] = float(free or 0.0)
        bucket["staked"] += float(alpha or 0.0) * float(price or 0.0)
    blocks = sorted(by_block)
    days = (blocks[-1] - blocks[0]) / 7200.0
    if days <= 0:
        return 0.0
    first = by_block[blocks[0]]
    last = by_block[blocks[-1]]
    return ((last["free"] + last["staked"]) - (first["free"] + first["staked"])) / days


def registration_burn(connection: sqlite3.Connection, days: float = 8.0) -> float:
    """TAO/day recycled buying uids, from the slot manager's executed buys."""
    since = dt.datetime.now(dt.timezone.utc).timestamp() - days * 86400
    total = connection.execute(
        "SELECT COALESCE(SUM(COALESCE(decision_recycle_tao, recycle_tao, 0)), 0) "
        "FROM slots "
        "WHERE executed = 1 AND time >= ?", (since,)).fetchone()[0]
    return float(total) / days


def infra_usd_per_day(connection: sqlite3.Connection, days: int = 7) -> dict:
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = connection.execute(
        "SELECT item, AVG(usd) FROM costs WHERE day >= ? GROUP BY item",
        (since,)).fetchall()
    return {item: round(float(value or 0), 2) for item, value in rows}
