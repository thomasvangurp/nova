from __future__ import annotations

import datetime as dt
import sqlite3

from nanobody_dash import economics, funding
from nanobody_opt.infra_sweep import decide


def _connection():
    connection = sqlite3.connect(":memory:")
    connection.executescript(economics.SCHEMA)
    connection.execute("""
        CREATE TABLE slots (
            time REAL, block INTEGER, executed INTEGER,
            decision_recycle_tao REAL, recycle_tao REAL
        )
    """)
    return connection


def test_target_covers_forward_aws_and_all_live_vast_but_not_more_than_earnings(monkeypatch):
    connection = _connection()
    monkeypatch.setattr(funding, "daily_mining_earnings", lambda *_: [
        {"day": "2026-09-22", "gross_tao": 1.2, "registration_tao": 0.2},
        {"day": "2026-09-23", "gross_tao": 1.0, "registration_tao": 0.2},
    ])
    target = funding.build_target(
        connection,
        {"aws": {"projected_month_end_usd": 300.0,
                 "operating_usd_per_day": 10.0},
         "vast": {"sn68_usd_per_day": 12.0,
                  "sn68_production_usd_per_day": 12.0,
                  "research_usd_per_day": 3.0,
                  "unattributed_usd_per_day": 0.0}},
        100.0,
        now=dt.datetime(2026, 9, 24, 12, tzinfo=dt.timezone.utc),
        maximum_tao=0.5,
    )
    assert target["operating_infra_usd"] == 22.0
    assert target["cost_tao"] == 0.22
    assert target["earnings_tao_per_day"] == 0.9
    assert target["authorized_tao"] == 0.22


def test_target_is_hard_capped_even_when_cost_and_earnings_are_higher(monkeypatch):
    connection = _connection()
    monkeypatch.setattr(funding, "daily_mining_earnings", lambda *_: [
        {"day": "2026-09-23", "gross_tao": 2.0, "registration_tao": 0.0},
    ])
    target = funding.build_target(
        connection,
        {"aws": {"projected_month_end_usd": 3000.0}, "vast": {}},
        100.0,
        now=dt.datetime(2026, 9, 24, 12, tzinfo=dt.timezone.utc),
        maximum_tao=0.5,
    )
    assert target["authorized_tao"] == 0.5


def test_pi_refuses_stale_duplicate_and_over_cap_targets():
    now = dt.datetime(2026, 9, 24, 12, tzinfo=dt.timezone.utc).timestamp()
    target = {
        "schema_version": 1, "target_day": "2026-09-24",
        "generated_at": int(now - 60), "valid_until": int(now + 3600),
        "authorized_tao": 0.2, "cost_tao": 0.2,
        "earnings_tao_per_day": 0.8,
    }
    assert decide(target, [], now=now, maximum_tao=0.5)["action"] == "transfer"
    duplicate = [{"target_day": "2026-09-24", "state": "transfer_started"}]
    assert decide(target, duplicate, now=now, maximum_tao=0.5)["reason"] == \
        "already_attempted_today"
    target["authorized_tao"] = 0.6
    target["cost_tao"] = 0.6
    assert decide(target, [], now=now, maximum_tao=0.5)["reason"] == \
        "hard_cap_exceeded"
    target["target_day"] = "2026-09-23"
    assert decide(target, [], now=now, maximum_tao=0.5)["reason"] == \
        "target_not_for_today"


def test_target_builds_end_to_end_without_stubs():
    """No stand-ins: a signature change in money.py must fail this test."""
    connection = _connection()
    day0 = dt.datetime(2026, 9, 22, tzinfo=dt.timezone.utc).timestamp()
    connection.executemany(
        "INSERT INTO balances (block, netuid, alpha, alpha_price_tao, free_tao, sampled_at) VALUES (?,?,?,?,?,?)",
        [(1000, 68, 100.0, 0.02, 1.0, day0), (8200, 68, 150.0, 0.02, 1.0, day0 + 86400)])
    connection.execute("INSERT INTO slots VALUES (?,?,?,?,?)", (day0 + 3600, 4000, 1, 0.2, 0.2))
    rows = funding.daily_mining_earnings(connection, 300.0)
    assert len(rows) == 1 and abs(rows[0]["gross_tao"] - 1.0) < 1e-9
    assert abs(rows[0]["registration_tao"] - 0.2) < 1e-9
    target = funding.build_target(
        connection, {"aws": {"operating_usd_per_day": 30.0}, "vast": {}}, 300.0,
        now=dt.datetime(2026, 9, 30, 12, tzinfo=dt.timezone.utc), maximum_tao=0.5)
    assert abs(target["earnings_tao_per_day"] - 0.8) < 1e-9
    assert abs(target["authorized_tao"] - 0.1) < 1e-9
