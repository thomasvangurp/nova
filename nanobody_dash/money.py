"""Per-subnet profit and loss, in TAO and USD per day.

Earned alpha is what the balance gained that was not transferred in as a
purchase.  On this coldkey the mining rewards arrive as stake TRANSFERS
(``is_transfer`` true), so they must be counted as income, while the
UNDELEGATE rows are the slot manager selling alpha to pay registration burn
and are therefore not income at all -- they are a change of asset.

    earned_alpha = d(balance) + sold_alpha - bought_alpha
    bought_alpha counts only DELEGATE rows that are NOT stake transfers

Getting this wrong in either direction turns a profitable subnet into a
burning one or vice versa, which is why the split is computed here once and
every figure reads it.
"""
from __future__ import annotations

import datetime as dt
import sqlite3

SUBNET_NAMES = {68: "SN68 NOVA", 2: "SN2 DSperse", 62: "SN62", 85: "SN85", 107: "SN107"}


def subnet_pnl(connection: sqlite3.Connection, tao_usd: float,
               registration_tao_per_day: float = 0.0) -> list[dict]:
    """One row per subnet: earned alpha, TAO/day, USD/day, and costs."""
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
        elif action == "DELEGATE" and not transfer:
            # bought with TAO; a transfer-in is a swept reward, not a purchase
            f["bought"] += alpha or 0.0

    out = []
    for netuid, series in balances.items():
        if len(series) < 2:
            continue
        (b0, a0, _), (b1, a1, p1) = series[0], series[-1]
        days = max((b1 - b0) / 7200.0, 1e-6)
        f = flows.get(netuid, dict(bought=0.0, sold=0.0, sold_tao=0.0))
        earned_alpha = (a1 - a0) + f["sold"] - f["bought"]
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
    # registration burn is a subnet-68 cost, applied after the sort so the
    # ordering reflects gross yield and the table shows net beside it
    for row in out:
        cost = registration_tao_per_day if row["netuid"] == 68 else 0.0
        row["cost_tao_per_day"] = round(cost, 4)
        row["net_tao_per_day"] = round(row["tao_per_day"] - cost, 4)
        row["net_usd_per_day"] = round(row["net_tao_per_day"] * tao_usd, 2)
    return out


def registration_burn(connection: sqlite3.Connection, days: float = 8.0) -> float:
    """TAO/day destroyed buying uids, from the slot manager's executed buys."""
    since = dt.datetime.now(dt.timezone.utc).timestamp() - days * 86400
    total = connection.execute(
        "SELECT COALESCE(SUM(decision_recycle_tao), 0) FROM slots "
        "WHERE executed = 1 AND time >= ?", (since,)).fetchone()[0]
    return float(total) / days


def infra_usd_per_day(connection: sqlite3.Connection, days: int = 7) -> dict:
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = connection.execute(
        "SELECT item, AVG(usd) FROM costs WHERE day >= ? GROUP BY item",
        (since,)).fetchall()
    return {item: round(float(value or 0), 2) for item, value in rows}
