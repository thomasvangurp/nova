from __future__ import annotations

import datetime as dt
import json
import sqlite3
import subprocess

import plotly.graph_objects as go

from nanobody_dash import build, collect, economics, infra, money, panels, refresh
from nanobody_dash.outcomes import annotate_field_beating_entries


def _entry(epoch: int, rank: int, uid: int = 7, hotkey: str = "hk") -> dict:
    return {
        "epoch": epoch,
        "rank": rank,
        "uid": uid,
        "hotkey": hotkey,
        "score": 10.0 + rank,
    }


def test_missed_epoch_does_not_reduce_quality_win_rate():
    won = _entry(1, 1)
    lost = _entry(3, 2)
    data = {
        "epochs": {1: {}, 2: {}, 3: {}},
        "by_epoch": {1: [won], 3: [lost]},
    }

    figure = build.fig_wins(data)

    assert list(figure.data[0].y) == [1, 1, 1]
    assert list(figure.data[1].y) == [100.0, 100.0, 50.0]
    assert "scored epochs" in figure.data[1].name


def test_fleet_age_uses_latest_scored_cohort_after_a_missed_round():
    entry = _entry(8, 4)
    data = {
        "epochs": {8: {}, 9: {}},
        "by_epoch": {8: [entry]},
        "ours": [entry],
        "first_seen": {"hk": 8},
    }

    figure = build.fig_ages(data)

    assert list(figure.data[0].y) == ["uid 7 · hk…"]
    assert any("epoch 8" in annotation.text for annotation in figure.layout.annotations)


def test_every_epoch_axis_is_rendered_as_an_exact_integer():
    figure = go.Figure(go.Scatter(x=[25270, 25271], y=[1, 2]))
    figure.update_xaxes(title="epoch")

    build._exact_epoch_axis(figure)

    assert figure.layout.xaxis.tickformat == "d"
    assert figure.layout.xaxis.exponentformat == "none"
    assert figure.layout.xaxis.showexponent == "none"


def test_efficiency_bubbles_use_nearest_point_hover():
    data = {
        "entries": [
            {"coldkey": "ours", "epoch": 1, "rank": 2, "score": 20},
            {"coldkey": "5EvpaL11", "epoch": 1, "rank": 1, "score": 10},
        ],
        "coldkey": "ours",
    }

    figure = build.fig_competitor_efficiency(data)

    assert figure.layout.hovermode == "closest"
    assert figure.layout.hoverdistance == -1
    assert "5EvpaL11..." in list(figure.data[0].text)


def test_two_entries_ahead_of_every_rival_both_count_as_entry_wins():
    entries = [
        {"coldkey": "ours", "epoch": 1, "rank": 1, "score": 10.0},
        {"coldkey": "ours", "epoch": 1, "rank": 2, "score": 11.0},
        {"coldkey": "rival", "epoch": 1, "rank": 3, "score": 12.0},
    ]

    annotate_field_beating_entries(entries)

    assert [row["entry_win"] for row in entries] == [True, True, False]


def test_rank_two_loses_when_rank_one_belongs_to_a_rival():
    entries = [
        {"coldkey": "rival", "epoch": 1, "rank": 1, "score": 10.0},
        {"coldkey": "ours", "epoch": 1, "rank": 2, "score": 11.0},
    ]

    annotate_field_beating_entries(entries)

    assert [row["entry_win"] for row in entries] == [True, False]


def test_competitor_stats_separates_entry_wins_from_winning_epochs():
    entries = [
        {"coldkey": "ours", "epoch": 1, "rank": 1, "score": 10.0},
        {"coldkey": "ours", "epoch": 1, "rank": 2, "score": 11.0},
        {"coldkey": "rival", "epoch": 1, "rank": 3, "score": 12.0},
    ]

    ours = next(
        row for row in build.competitor_stats({"entries": entries, "coldkey": "ours"})
        if row["ours"]
    )

    assert ours["wins"] == 2
    assert ours["winning_epochs"] == 1
    assert ours["win_per_entry"] == 100.0


def test_purchase_history_keeps_each_price_and_marks_purchases():
    data = {
        "slots": [
            {"time": 1.0, "epoch": 10, "registered": 8,
             "recycle_tao": 0.2, "decision_recycle_tao": None,
             "executed": 0, "reason": "wait"},
            {"time": 2.0, "epoch": 10, "registered": 8,
             "recycle_tao": 0.1, "decision_recycle_tao": 0.1,
             "executed": 1, "reason": "buy"},
        ]
    }

    figure = build.fig_slot_history(data)

    assert list(figure.data[0].y) == [0.2, 0.1]
    assert list(figure.data[1].y) == [0.1]
    assert figure.data[1].name == "purchased"


def test_paid_fleet_uses_last_epoch_roster_not_a_median_that_hides_churn(tmp_path):
    database = tmp_path / "dashboard.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(collect.SCHEMA)
    connection.execute(
        "INSERT INTO epochs VALUES (?,?,?,?,?,?)",
        (25293, 52, 65.0, 171.0, 127.0, 4.0),
    )
    connection.executemany(
        "INSERT INTO slots "
        "(time,epoch,block,registered,target,recycle_tao,"
        "decision_recycle_tao,executed,reason,hotkey) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (1.0, 25293, 1, 11, 12, 0.19, None, 0, "wait", None),
            (2.0, 25293, 2, 12, 12, 0.18, None, 0, "wait", None),
            (3.0, 25293, 3, 12, 12, 0.17, None, 0, "wait", None),
            (4.0, 25293, 4, 11, 12, 0.16, None, 0, "wait", None),
        ],
    )
    connection.commit()
    connection.close()

    data = build.load(database, "ours")

    assert data["registered"][25293] == 11


def test_headline_distinguishes_epoch_close_from_current_chain_total():
    data = {
        "epochs": {25293: {"field_size": 52, "winner": 65.0}},
        "by_epoch": {25293: [_entry(25293, 1)]},
        "ours": [_entry(25293, 1)],
        "registered": {25293: 11},
        "current_held": 12,
    }

    scoring = next(card for card in build.headline(data) if card[0] == "uids scoring")

    assert scoring == ("uids scoring", "1", "11 at epoch close · 12 now")


def test_daily_slot_spend_keeps_legacy_purchases_without_a_decision_price():
    data = {
        "slots": [{
            "time": 1_790_000_000.0, "executed": 1,
            "recycle_tao": 0.16, "decision_recycle_tao": None,
        }]
    }

    figure = build.fig_slots(data)

    assert list(figure.data[0].y) == [0.16]
    assert list(figure.data[0].customdata) == [1]
    assert figure.data[1].name == "median price per uid"


def _money_connection():
    connection = sqlite3.connect(":memory:")
    connection.executescript(economics.SCHEMA)
    connection.execute(
        "CREATE TABLE slots (time REAL, block INTEGER, recycle_tao REAL, "
        "decision_recycle_tao REAL, executed INTEGER)"
    )
    return connection


def test_daily_money_history_uses_interval_local_flows_and_costs():
    connection = _money_connection()
    # Exactly one day apart: alpha rises by two, but one was purchased, so only
    # one alpha is earned. At tau0.5/alpha that is tau0.5 gross.
    connection.executemany(
        "INSERT INTO balances VALUES (?,?,?,?,?,?)",
        [(100, 68, 10.0, 0.5, 1.0, 1.0),
         (7300, 68, 12.0, 0.5, 1.0, 2.0)],
    )
    connection.execute(
        "INSERT INTO delegations "
        "(id,timestamp,block,netuid,action,alpha,tao,alpha_price_tao,is_transfer) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("d1", "2026-09-02T00:00:00Z", 5000, 68, "DELEGATE", 1.0,
         0.5, 0.5, 0),
    )
    connection.execute(
        "INSERT INTO slots VALUES (?,?,?,?,?)",
        (1_788_307_200.0, 6000, 0.125, 0.1, 1),
    )

    history = money.subnet_pnl_history(connection, tao_usd=10.0)
    assert len(history) == 1
    assert history[0]["gross_tao_per_day"] == 0.5
    assert history[0]["registration_tao_per_day"] == 0.1
    # Registration burn is a real cost, so net is gross minus the burn (0.4).
    assert history[0]["net_tao_per_day"] == 0.4

    connection.execute(
        "INSERT INTO costs VALUES (?,?,?,?)",
        (history[0]["day"], "vast.ai", 2.0, "invoiced"),
    )
    daily = money.daily_money_history(connection, history, tao_usd=10.0)
    assert daily[0]["gross_usd"] == 5.0
    assert daily[0]["registration_usd"] == 1.0
    assert daily[0]["infra_usd"] == 2.0
    assert daily[0]["infrastructure"] == {"vast.ai": 2.0}
    # Net USD is mining revenue minus registration and infrastructure (2.0).
    assert daily[0]["net_usd"] == 2.0
    assert daily[0]["portfolio_tao"] == 7.0


def test_aws_cost_agent_is_read_only_and_keeps_daily_sn2_costs():
    calls = []

    def fetch(arguments):
        calls.append(tuple(arguments[:2]))
        if arguments[:2] == ["sts", "get-caller-identity"]:
            return {"Account": infra.AWS_ACCOUNT}
        if arguments[:2] == ["ce", "get-cost-and-usage"]:
            return {"ResultsByTime": [
                {"TimePeriod": {"Start": "2026-09-23", "End": "2026-09-24"},
                 "Total": {"UnblendedCost": {"Amount": "25.50", "Unit": "USD"}}},
                {"TimePeriod": {"Start": "2026-09-24", "End": "2026-09-25"},
                 "Total": {"UnblendedCost": {"Amount": "30.25", "Unit": "USD"}}},
            ]}
        return {"Total": {"Amount": "100.00", "Unit": "USD"}}

    result = infra.aws_cost_explorer(
        now=dt.date(2026, 9, 24), days=2, fetch=fetch,
    )

    assert set(calls) <= infra.AWS_ALLOWED_READS
    assert result["bill_to_date_usd"] == 55.75
    assert result["mtd_usd"] == 55.75
    assert result["provider_forecast_remaining_usd"] == 100.0
    assert result["operating_usd_per_day"] == 25.5
    assert result["forecast_remaining_usd"] == 178.5
    assert result["forecast_remainder_usd"] == 178.5
    assert result["projected_month_end_usd"] == 204.0
    assert result["estimated_month_usd"] == 204.0
    assert result["daily"][0]["item"] == "aws:sn2:total"


def test_aws_cost_agent_breaks_down_daily_cost_and_excludes_migration():
    def fetch(arguments):
        if arguments[:2] == ["sts", "get-caller-identity"]:
            return {"Account": infra.AWS_ACCOUNT}
        if arguments[:2] == ["ce", "get-cost-and-usage"]:
            return {"ResultsByTime": [{
                "TimePeriod": {"Start": "2026-09-01", "End": "2026-09-02"},
                "Groups": [
                    {"Keys": ["EC2 - Other", "USE2-DataTransfer-Regional-Bytes"],
                     "Metrics": {"UnblendedCost": {"Amount": "21.83"}}},
                    {"Keys": ["Amazon Elastic Compute Cloud - Compute",
                              "USE2-BoxUsage:r7i.4xlarge"],
                     "Metrics": {"UnblendedCost": {"Amount": "10.00"}}},
                ],
            }]}
        return {"Total": {"Amount": "20.00", "Unit": "USD"}}

    result = infra.aws_cost_explorer(
        now=dt.date(2026, 9, 1), days=1, fetch=fetch,
        operational_exclusions=[{
            "days": ["2026-09-01"],
            "usage_type_contains": "DataTransfer-Regional-Bytes",
            "reason": "migration",
        }],
    )

    assert result["bill_to_date_usd"] == 31.83
    assert result["operating_mtd_usd"] == 10.0
    assert result["one_time_excluded_usd"] == 21.83
    assert result["provider_forecast_remaining_usd"] == 20.0
    assert result["operating_usd_per_day"] == 10.0
    assert result["projected_month_end_usd"] == 300.0
    assert result["operating_breakdown_mtd_usd"] == {"compute": 10.0}
    assert result["daily"] == [{
        "day": "2026-09-01", "item": "aws:sn2:compute", "usd": 10.0,
        "note": "AWS Cost Explorer recurring operations; one-time costs excluded by explicit policy",
    }]


def test_cloud_cost_panel_keeps_aws_and_vast_scopes_explicit():
    figure = panels.fig_cloud_costs({
        "aws": {"bill_to_date_usd": 400,
                "operating_mtd_usd": 375,
                "one_time_excluded_usd": 25,
                "projected_month_end_usd": 654.07,
                "operating_breakdown_mtd_usd": {"compute": 300,
                                                 "data-transfer": 75},
                "status": "live"},
        "vast": {"sn68_usd_per_day": 40, "account_mtd_usd": 300,
                 "status": "live"},
        "payments": {"cloud_funding_source": "Krak virtual card",
                     "real_card_allowed": False},
    })

    values = figure.data[0].cells.values
    assert values[0] == ["AWS Bills", "AWS operations", "Vast.ai", "Funding guard"]
    assert "SN2" in values[1][0]
    assert "SN68" in values[1][2]
    assert values[2][0] == "$400.00 accrued bill"
    assert values[2][1] == "$375.00 operating MTD"
    assert values[3][1] == "$654.07 projected month-end"
    assert "$25.00 one-time excluded" in values[4][1]
    assert values[3][3] == "agent cannot charge a real card"


def test_aws_breakdown_matches_bills_categories_and_shows_usage():
    figure = panels.fig_aws_cost_breakdown({"aws": {
        "bill_to_date_usd": 654.08,
        "operating_mtd_usd": 628.54,
        "one_time_excluded_usd": 25.54,
        "breakdown_mtd_usd": {
            "compute": 308.13, "storage": 104.13,
            "data-transfer": 111.43, "lightsail": 11.22,
            "vpc": 5.64, "tax": 113.51, "other": 0.02,
        },
        "operating_breakdown_mtd_usd": {
            "compute": 308.13, "storage": 104.13,
            "data-transfer": 85.89, "lightsail": 11.22,
            "vpc": 5.64, "tax": 113.51, "other": 0.02,
        },
        "usage_breakdown_mtd": [
            {"usage_type": "USE2-DataTransfer-Out-Bytes",
             "quantity": 1054.261, "unit": "GB", "usd": 85.88},
            {"usage_type": "USE2-DataTransfer-Regional-Bytes",
             "quantity": 2555.468, "unit": "GB", "usd": 25.54},
            {"usage_type": "USE2-DataTransfer-In-Bytes",
             "quantity": 836.251, "unit": "GB", "usd": 0},
        ],
    }})

    values = figure.data[0].cells.values
    assert values[0][0] == "Elastic Compute Cloud"
    assert values[1][0] == "$412.26"
    assert values[0][1] == "Data Transfer"
    assert values[1][1] == "$111.43"
    assert values[2][1] == "$85.89"
    assert "1,054.261 GB" in values[3][1]


def test_infrastructure_daily_hover_is_one_compact_nonzero_breakdown():
    figure = panels.fig_infrastructure_cost_history([{
        "day": "2026-09-24",
        "infrastructure": {
            "aws:sn2:compute": 12.0,
            "aws:sn2:data-transfer": 3.0,
            "aws:sn2:tax": 0.0,
            "vast.ai:research": 2.5,
        },
    }])

    assert figure.layout.hovermode == "closest"
    hover = figure.data[0].customdata[0]
    assert "total $17.50" in hover
    assert "AWS $15.00" in hover
    assert "Vast.ai $2.50" in hover
    assert "SN2 tax" not in hover


def test_subnet_yield_hover_is_one_compact_trace():
    figure = panels.fig_subnet_pnl_history([
        {
            "day": "2026-09-24", "netuid": 2, "name": "SN2 DSperse",
            "net_tao_per_day": 0.5444, "gross_tao_per_day": 0.5444,
            "registration_tao_per_day": 0.0, "holding_tao": 5.836,
            "elapsed_days": 0.77,
        },
        {
            "day": "2026-09-24", "netuid": 68, "name": "SN68 NOVA",
            "net_tao_per_day": 0.4441, "gross_tao_per_day": 3.3895,
            "registration_tao_per_day": 2.9454, "holding_tao": 22.188,
            "elapsed_days": 0.77,
        },
    ])

    assert figure.layout.hovermode == "closest"
    assert figure.layout.hoverdistance == -1
    assert "holding" not in figure.data[0].hovertemplate
    assert "interval" not in figure.data[0].hovertemplate
    assert "registration" not in figure.data[0].hovertemplate
    assert "registration" in figure.data[1].hovertemplate


def test_vast_runtime_separates_production_research_and_unattributed():
    connection = _money_connection()
    live = {"instances": [
        {"label": "nova-nanobody-pixi", "state": "running", "dph": 1.0,
         "num_gpus": 4},
        {"label": "nova-nanobody-h2-fast", "state": "running", "dph": 2.0,
         "num_gpus": 8},
        {"label": "nova-boltz2-seed-noise-test", "state": "running", "dph": 0.7,
         "num_gpus": 1},
        {"label": None, "state": "loading", "dph": 0.5, "num_gpus": 1},
        {"label": "old-research", "state": "exited", "dph": 99.0,
         "num_gpus": 8},
    ]}

    scopes = infra.record_vast_runtime(
        connection, live, ("nova-nanobody-pixi",),
        ("nova-boltz2-seed-noise-test",), sampled_at=1_798_156_800,
    )

    assert scopes["sn68-prod"]["usd_per_hour"] == 1.0
    assert scopes["sn68-research"]["usd_per_hour"] == 0.7
    assert scopes["research"]["usd_per_hour"] == 2.0
    assert scopes["unattributed"]["usd_per_hour"] == 0.5
    costs = dict(connection.execute(
        "SELECT item, usd FROM costs ORDER BY item"
    ).fetchall())
    assert costs == {
        "vast.ai:research": 48.0,
        "vast.ai:sn68-research": 16.8,
        "vast.ai:sn68-prod": 24.0,
        "vast.ai:unattributed": 12.0,
    }


def test_tao_price_falls_back_to_coingecko_and_caches_last_good_value():
    connection = _money_connection()

    def unavailable():
        raise RuntimeError("provider unavailable")

    price = economics.resolve_tao_usd(
        connection, now=1_000.0, taostats_fetch=unavailable,
        coingecko_fetch=lambda: 291.08,
    )
    assert price == 291.08
    assert connection.execute(
        "SELECT usd, source FROM market_prices WHERE asset = 'TAO'"
    ).fetchone() == (291.08, "coingecko")

    # A fresh cache performs no network request.
    assert economics.resolve_tao_usd(
        connection, now=1_100.0, taostats_fetch=unavailable,
        coingecko_fetch=unavailable,
    ) == 291.08
    # Even after it becomes stale, provider failure reuses a real historical
    # price rather than publishing a fictional zero-revenue dashboard.
    assert economics.resolve_tao_usd(
        connection, now=100_000.0, taostats_fetch=unavailable,
        coingecko_fetch=unavailable,
    ) == 291.08


def test_tao_price_refuses_zero_when_no_valid_source_exists():
    connection = _money_connection()

    def unavailable():
        raise RuntimeError("provider unavailable")

    try:
        economics.resolve_tao_usd(
            connection, now=1_000.0, taostats_fetch=unavailable,
            coingecko_fetch=unavailable,
        )
    except RuntimeError as error:
        assert "no valid TAO/USD price" in str(error)
    else:
        raise AssertionError("zero must never replace an unavailable price")


def test_current_balance_collection_aggregates_hotkeys_per_subnet(monkeypatch):
    connection = _money_connection()
    same_day = dt.datetime(2026, 9, 22, 0, 1, tzinfo=dt.timezone.utc).timestamp()
    connection.execute(
        "INSERT INTO balances VALUES (?,?,?,?,?,?)",
        (8995, 68, 2.9, 0.5, 3.0, same_day),
    )
    runtime_locator = object()
    monkeypatch.setattr(economics, "_api_key", lambda: runtime_locator)

    def fake_get(path, params, **kwargs):
        assert kwargs["key"] is runtime_locator
        if path == "account/latest/v1":
            return {"data": [{"balance_free": "3000000000"}]}
        return {"data": [
            {"netuid": 68, "block_number": 9000, "balance": "1000000000",
             "balance_as_tao": "500000000", "timestamp": "2026-09-22T00:00:00Z"},
            {"netuid": 68, "block_number": 9000, "balance": "2000000000",
             "balance_as_tao": "1000000000", "timestamp": "2026-09-22T00:00:00Z"},
        ]}

    monkeypatch.setattr(economics, "get", fake_get)

    assert economics.collect_current_balances(connection, "public-coldkey") == 1
    row = connection.execute(
        "SELECT alpha, alpha_price_tao, free_tao FROM balances"
    ).fetchone()
    assert row == (3.0, 0.5, 3.0)
    assert connection.execute("SELECT COUNT(*) FROM balances").fetchone()[0] == 1


def test_expiry_panel_shows_current_chain_total(tmp_path):
    detail = tmp_path / "hotkeys.json"
    detail.write_text(json.dumps({
        "block": 100,
        "immunity_period": 5000,
        "hotkeys": [{"uid": 7, "age_epochs": 2, "immune": True}],
    }))

    figure = panels.fig_hotkey_expiry(detail, current_total=11)

    assert any("11 hotkeys currently registered" in annotation.text
               for annotation in figure.layout.annotations)


def test_pi_snapshot_is_allowlisted_and_atomically_saved(tmp_path, monkeypatch):
    local = tmp_path / "slots.jsonl"
    staged = tmp_path / "slots.jsonl.next"
    payload = (
        json.dumps({"time": 1, "epoch": 2, "registered": 11,
                    "recycle_tao": 0.12, "token": "must-not-persist"}) + "\n"
    ).encode()
    monkeypatch.setattr(refresh, "SLOT_SNAPSHOT", local)
    monkeypatch.setattr(refresh, "SLOT_SNAPSHOT_NEXT", staged)
    monkeypatch.setattr(
        refresh.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, payload, b""),
    )

    assert refresh.sync_slot_snapshot() == 1
    saved = json.loads(local.read_text())
    assert saved == {"epoch": 2, "recycle_tao": 0.12,
                     "registered": 11, "time": 1}


def test_easynano_panel_is_not_rendered():
    assert all(name != "easynano_status"
               for _, _, entries in build.TABS for _, _, name in entries)
