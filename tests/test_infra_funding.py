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
    monkeypatch.setattr(funding, "subnet_pnl_history", lambda *_: [])
    monkeypatch.setattr(funding, "daily_money_history", lambda *_: [
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
    monkeypatch.setattr(funding, "subnet_pnl_history", lambda *_: [])
    monkeypatch.setattr(funding, "daily_money_history", lambda *_: [
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
