"""One-off: reconstruct per-day balances across the 2026-09-24 -> now gap.

The #money daily chart froze on 09-24 when the Taostats API 429'd.  The last
archive-node balance sample per subnet sits at block 9138968; the next sample is
the fresh ``collect_current_balances`` snapshot inserted by the economics
collector.  ``subnet_pnl_history`` treats those two samples as a single
interval, so several days of yield smear into one daily figure.

For each subnet we reconstruct one synthetic balance sample per missing day from
the delegation flows:

    alpha(day) = alpha@9138968 + sum(DELEGATE alpha) - sum(UNDELEGATE alpha)
                 over flows in (9138968, day_end_block]

which is exact wherever mining rewards arrive as DELEGATE stake transfers
(is_transfer) in ``delegation/v1`` -- true for SN68.  A subnet whose rewards
accrue directly to the stake without a delegation extrinsic (observed for SN2)
cannot be reconstructed from flows; for those we insert only the fresh snapshot
and record the ambiguity in data/money_backfill_note.md.

Usage:
    python data/_money_backfill_reconstruct.py --database data/nanobody_dash.sqlite          # dry run
    python data/_money_backfill_reconstruct.py --database data/nanobody_dash.sqlite --apply  # write
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import time
from pathlib import Path

ANCHOR_BLOCK = 9138968  # last archive balance sample (2026-09-24 17:16 UTC)


def _iso_to_ts(value: str) -> float | None:
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OSError):
        return None


def _anchor_day(connection, netuid, anchor_block) -> dt.date:
    row = connection.execute(
        "SELECT sampled_at FROM balances WHERE netuid=? AND block=?",
        (netuid, anchor_block)).fetchone()
    if row and row[0]:
        return dt.datetime.fromtimestamp(float(row[0]), dt.timezone.utc).date()
    return dt.date(2026, 9, 24)


def reconstruct(connection: sqlite3.Connection, anchor_block: int = ANCHOR_BLOCK):
    anchors: dict[int, tuple[float, float, float]] = {}
    fresh: dict[int, tuple[int, float, float, float, float]] = {}
    for block, netuid, alpha, price, free, sampled_at in connection.execute(
        "SELECT block, netuid, alpha, alpha_price_tao, free_tao, sampled_at "
        "FROM balances ORDER BY netuid, block"
    ):
        netuid = int(netuid)
        if int(block) <= anchor_block:
            anchors[netuid] = (float(alpha or 0), float(price or 0), float(free or 0))
        else:
            fresh[netuid] = (int(block), float(alpha or 0), float(price or 0),
                             float(free or 0), float(sampled_at or 0))

    flows: dict[int, list[tuple[int, str, float, str]]] = {}
    for netuid, action, alpha, block, timestamp in connection.execute(
        "SELECT netuid, action, alpha, block, timestamp FROM delegations "
        "WHERE netuid IS NOT NULL AND block IS NOT NULL AND block > ? "
        "ORDER BY block", (anchor_block,)
    ):
        flows.setdefault(int(netuid), []).append(
            (int(block), str(action), float(alpha or 0), str(timestamp or ""))
        )

    rows_to_insert: list[tuple] = []
    report: list[str] = []

    for netuid in sorted(set(anchors) | set(fresh)):
        if netuid not in anchors or netuid not in fresh:
            report.append(
                f"netuid {netuid}: SKIPPED (anchor={'yes' if netuid in anchors else 'no'}, "
                f"fresh={'yes' if netuid in fresh else 'no'})")
            continue
        alpha_start, price_start, free_start = anchors[netuid]
        fresh_block, alpha_now, _, _, fresh_sampled_at = fresh[netuid]
        sub_flows = flows.get(netuid, [])

        recon_now = alpha_start + sum(
            (a if act == "DELEGATE" else -a) for _, act, a, _ in sub_flows)
        total_delta = alpha_now - alpha_start
        flow_delta = recon_now - alpha_start
        residual = total_delta - flow_delta

        report.append(
            f"netuid {netuid}: anchor alpha={alpha_start:.4f}@9138968, "
            f"fresh alpha={alpha_now:.4f}@{fresh_block}, "
            f"flows={len(sub_flows)}, flow_delta={flow_delta:.4f}, "
            f"total_delta={total_delta:.4f}, residual={residual:.4f}")

        anchor_day = _anchor_day(connection, netuid, anchor_block)
        fresh_day = (dt.datetime.fromtimestamp(fresh_sampled_at, dt.timezone.utc).date()
                     if fresh_sampled_at else anchor_day + dt.timedelta(days=4))
        missing_days = []
        day = anchor_day + dt.timedelta(days=1)
        while day < fresh_day:
            missing_days.append(day)
            day += dt.timedelta(days=1)

        if not missing_days:
            report.append(f"netuid {netuid}: no missing days")
            continue

        # Only spread across days when the flows actually explain the observed
        # change.  Otherwise the residual (invisible mining rewards) cannot be
        # attributed per day without fabricating a distribution.
        if abs(residual) > max(1e-6, abs(total_delta) * 0.02):
            report.append(
                f"netuid {netuid}: AMBIGUOUS -- flows explain {flow_delta:.4f} of "
                f"{total_delta:.4f} alpha; not spreading (no fabrication)")
            continue

        # One sample per missing day, placed at the last delegation block that
        # falls on that day (or a 7200-block grid step when the day has none).
        day_starts = {
            day: dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()
            for day in missing_days}
        placed: dict[dt.date, tuple[int, float, float]] = {}
        cum = alpha_start
        for block, action, alpha, timestamp in sub_flows:
            cum += alpha if action == "DELEGATE" else -alpha
            ts = _iso_to_ts(timestamp)
            if ts is None:
                continue
            for day in missing_days:
                if day_starts[day] <= ts < day_starts[day] + 86400:
                    placed[day] = (block, cum, ts)

        for idx, day in enumerate(missing_days, start=1):
            if day in placed:
                block, cum, ts = placed[day]
                sampled_at = ts
            else:
                block = anchor_block + 7200 * idx
                cum = alpha_start + sum(
                    (a if act == "DELEGATE" else -a)
                    for b, act, a, _ in sub_flows if b <= block)
                sampled_at = block * 12
            # Never overwrite the real fresh snapshot; clamp below it.
            if block >= fresh_block:
                block = fresh_block - 1
            rows_to_insert.append(
                (block, netuid, cum, price_start, free_start, sampled_at))
            report.append(
                f"netuid {netuid}: day {day.isoformat()} -> block {block}, "
                f"alpha {cum:.4f}, sampled_at "
                f"{dt.datetime.fromtimestamp(sampled_at, dt.timezone.utc).isoformat()}")

    return rows_to_insert, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/nanobody_dash.sqlite"))
    parser.add_argument("--apply", action="store_true", help="write rows (default dry run)")
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    rows, report = reconstruct(connection)
    print("\n".join(report))
    print(f"\n{'WOULD insert' if not args.apply else 'Inserting'} {len(rows)} rows")
    if args.apply and rows:
        connection.executemany(
            "INSERT OR REPLACE INTO balances "
            "(block, netuid, alpha, alpha_price_tao, free_tao, sampled_at) "
            "VALUES (?,?,?,?,?,?)", rows)
        connection.commit()
        print("committed.")
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
