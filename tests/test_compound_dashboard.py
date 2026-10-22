import json
import sqlite3

import pytest

from nanobody_dash import compound


def test_dynamic_floor_excludes_own_coldkey_and_uses_calibration():
    cache = {
        "1": [
            {"coldkey": "ours", "reaction": 3, "score": 3.0},
            {"coldkey": "rival", "reaction": 3, "score": 1.0},
        ],
        "2": [
            {"coldkey": "ours", "reaction": 3, "score": 3.0},
            {"coldkey": "rival", "reaction": 3, "score": 1.0},
        ],
        "_self_calibration": {
            "coldkeys": ["ours"],
            "evidence": [
                {"epoch": 1, "reaction": 3, "optimism": 0.10},
                {"epoch": 2, "reaction": 3, "optimism": 0.10},
            ],
        },
    }

    floors, details = compound._dynamic_floors_from_cache(cache, next_epoch=3)

    assert floors[3] == 1.0682
    assert details[3]["winner"] == 1.0
    assert details[3]["owned_coldkeys_excluded"] == 1
    assert details[3]["own_evidence_epochs"] == 2


def test_final_fleet_rejection_is_not_counted_as_an_allocation():
    lines = [
        '{"at": "12:00:00", "event": "window_open", "epoch": 100, "reaction": 3}',
        '{"event": "allocated", "uids": 1, "epoch": 100, "reaction": 3, '
        '"best": 1.7, "worst": 1.7, "entropy_min": 0.26, "distinct": 20}',
        json.dumps({
            "event": "fleet_below_baseline", "raw_floor": 1.8, "worst_raw": 1.7,
            "prediction_offset": 0.09, "realized_floor": 1.71,
            "worst_predicted_realized": 1.61,
        }),
    ]

    parsed = compound._parse_loop_log("\n".join(lines))

    assert parsed["allocated"] == []
    assert parsed["blocked"] == [{
        "epoch": 100, "reaction": 3, "selected_score": 1.7, "baseline": 1.8,
        "prediction_offset": 0.09, "predicted_realized": 1.61,
        "realized_floor": 1.71, "final_gate": True,
    }]


def test_strategy_verdict_does_not_call_relative_lift_a_breakthrough():
    ab = {
        "arms": {
            "control": {"n": 40, "mean_source_delta": -0.08, "ge_entry": 0},
            "treatment": {"n": 200, "mean_source_delta": -0.06, "ge_entry": 0},
            "comparison": {"mean_source_delta_difference": 0.02},
        },
        "verdict": {"minimum_per_arm": 20, "status": "better"},
    }
    lanes = {"lanes": {
        "competitor_motif": {"max": 0.09},
        "frontier": {"max": 0.11},
    }}

    verdict = compound._strategy_verdict(ab, lanes)

    assert verdict["status"] == "relative_only"
    assert verdict["label"] == "relative signal; no entry hit"
    assert verdict["best_lane"] == "frontier"


@pytest.mark.parametrize("policy", ["field-recombination-v1", "field-recombination-v2"])
def test_field_recombination_uses_pre_proxy_random_arm_as_control(policy):
    lanes = {
        "active_strategy": policy,
        "active_geometry": "known-block-triple-pair-novel",
        "block_share": 300,
        "entry_score": 0.12,
        "lanes": {
            "block_bandit/field_recombination/random": {
                "scored": 25, "mean": 0.07, "max": 0.10, "ge_entry": 0,
            },
            "block_bandit/field_recombination/proxy": {
                "scored": 100, "mean": 0.08, "max": 0.13, "ge_entry": 1,
            },
            "block_bandit": {"max": 0.13},
        },
    }

    verdict = compound._strategy_verdict({}, lanes)

    assert verdict["mode"] == "field_recombination"
    assert verdict["status"] == "breakthrough"
    assert verdict["relative_lift"] == pytest.approx(0.01)
    assert verdict["control_n"] == 25
    assert verdict["treatment_n"] == 100


def test_live_gate_table_makes_reused_value_visibly_stale():
    status = {
        "pi_reachable": True,
        "gap_checks": {
            "5": {
                "selected_score": 2.12,
                "floor": 2.25,
                "gap": -0.13,
                "clears_now": False,
                "checked_at": 1_700_000_000,
                "refresh_error": "timeout_or_unreachable",
                "refresh_failed_at": 1_700_000_300,
            }
        },
    }

    rendered = compound.compound_status_table(status)

    assert "STALE — latest refresh failed" in rendered
    assert "2.120 vs 2.25" in rendered


def test_partial_live_gate_refresh_is_failed_and_reuses_value_visibly(
        monkeypatch, tmp_path):
    previous = {
        "gap_checks": {
            "5": {
                "selected_score": 2.12,
                "floor": 2.25,
                "gap": -0.13,
                "clears_now": False,
                "checked_at": 1_700_000_000,
            }
        }
    }
    status_path = tmp_path / "compound_status.json"
    status_path.write_text(json.dumps(previous))
    connection = sqlite3.connect(":memory:")
    connection.executescript(compound.SCHEMA)

    monkeypatch.setattr(compound, "STATUS_PATH", status_path)
    monkeypatch.setattr(compound, "connect", lambda: connection)
    monkeypatch.setattr(compound, "_refresh_boxes_from_pi", lambda: None)
    monkeypatch.setattr(compound, "_pull_submitted_archive", lambda: True)
    monkeypatch.setattr(compound, "_ssh", lambda *args, **kwargs: "loop")
    monkeypatch.setattr(compound, "_parse_loop_log", lambda _text: {
        "window_epoch_reaction": {100: 1},
        "allocated": [], "blocked": [], "final_gates": {},
        "ordered_window_events": [],
    })
    monkeypatch.setattr(compound, "_dynamic_floors", lambda _epoch: (
        {reaction: values[2] for reaction, values in compound.REACTION_BOXES.items()}, {}))
    monkeypatch.setattr(compound, "_search_strategy_check", lambda _reaction: None)

    def gap(reaction, _epoch):
        if reaction == 5:
            return None
        return {
            "selected_score": 1.0, "floor": 1.1, "gap": -0.1,
            "clears_now": False, "pool_size": 100,
        }

    monkeypatch.setattr(compound, "_live_gap_check", gap)
    monkeypatch.setattr(compound, "LEGACY_BOX_CHECKS", True)

    status = compound.collect(live_gap_check=True)

    assert status["gap_refresh_complete"] is False
    assert status["gap_refresh_failed_reactions"] == [5]
    assert status["gap_checks"]["5"]["selected_score"] == 2.12
    assert status["gap_checks"]["5"]["refresh_error"] == "timeout_or_unreachable"


def test_gap_history_plot_merges_newer_status_snapshot():
    connection = sqlite3.connect(":memory:")
    connection.executescript(compound.SCHEMA)
    connection.execute(
        "INSERT INTO gap_history VALUES (?,?,?,?,?,?,?)",
        (100.0, 4, 1.69, 1.79, -0.10, 0, 2000),
    )
    status = {"gap_checks": {"4": {
        "checked_at": 200.0,
        "selected_score": 1.708,
        "floor": 1.79,
        "gap": -0.082,
        "clears_now": False,
        "pool_size": 2652,
    }}}

    rows = compound.gap_history_rows(connection, status)

    assert [row["selected_score"] for row in rows] == [1.69, 1.708]
    assert rows[-1]["checked_at"] == 200.0


SANQI_DUMP = """##reaction_count.json
{"before": 5, "count": 44, "from_epoch": 25693}
##reaction_health.json
{"dead": [9, 14, 30, 41, 42]}
##sprint_state.json
{"epoch": 25697, "valid_until": 1791926333.1}
##sprint_trigger.log
{"at": "2026-10-13 10:06:23", "event": "sprint_started", "epoch": 25692, "reaction": 41, "minutes": 42.6, "boxes": 2, "pushed": 1}
{"at": "2026-10-13 11:20:03", "event": "sprint_started", "epoch": 25693, "reaction": 31, "minutes": 41.8, "boxes": 2, "pushed": 1}
{"at": "2026-10-13 11:21:03", "event": "sprint_started", "epoch": 25693, "reaction": 31, "minutes": 40.8, "boxes": 2, "pushed": 2}
{"at": "2026-10-13 14:34:36", "event": "sprint_forced", "reaction": 35, "minutes": 3.0, "boxes": 3, "pushed": 2}
{"at": "2026-10-13 14:55:45", "event": "sprint_started", "epoch": 25696, "reaction": 22, "minutes": 42.4, "boxes": 4, "pushed": 3}
{"at": "2026-10-13 15:05:14", "event": "sprint_joined_late", "epoch": 25696, "box": 1, "attempt": 23, "seconds_left": 1932}
{"at": "2026-10-13 16:50:00", "event": "dead_reaction", "epoch": 25698, "reaction": 9, "why": "x"}
"""


def test_parse_sanqi_reads_config_dead_list_and_per_epoch_sprints():
    parsed = compound._parse_sanqi(SANQI_DUMP)

    assert parsed["config"]["count"] == 44
    assert parsed["dead"] == [9, 14, 30, 41, 42]
    assert parsed["current_epoch"] == 25697
    by_epoch = {row["epoch"]: row for row in parsed["sprints"]}
    assert by_epoch[25693]["pushed"] == 2 and by_epoch[25693]["minutes"] == 41.8   # duplicate start: first budget, max pushed
    assert by_epoch[25696]["late_joins"] == 1
    assert by_epoch[25698]["status"] == "dead" and by_epoch[25698]["reaction"] == 9
    assert [row["epoch"] for row in parsed["sprints"]] == sorted(by_epoch, reverse=True)   # the forced rehearsal has no epoch


def test_sanqi_tables_show_drawn_reactions_beyond_the_legacy_five():
    status = {
        "pi_reachable": True,
        "draws_by_reaction": {"1": 92, "22": 2},
        "clears_by_reaction": {"1": 9, "22": 1},
        "wins_by_reaction": {"1": 0},
        "recent_history": [
            {"epoch": 25696, "reaction": 22, "outcome": "cleared", "selected_score": 1.3239},
            {"epoch": 25698, "reaction": 9, "outcome": "dead"},
        ],
        "sanqi": compound._parse_sanqi(SANQI_DUMP),
    }

    status_html = compound.compound_status_table(status)
    sprint_html = compound.compound_strategy_table(status)
    history_html = compound.compound_history_table(status)

    assert "rxn 22" in status_html and "3/4 boxes" in status_html and "1.324" in status_html.replace("1.3239", "1.324")
    assert "9, 14, 30, 41, 42" in status_html
    assert "legacy bank" in status_html                      # rxn 1 has history but no SANQI sprint
    assert "skipped: dead reaction" in sprint_html and "awaiting leaderboard" in sprint_html
    assert "submitted, awaiting leaderboard" in history_html and "can never validate" in history_html
    assert "nova-sanqi-N" in compound.compound_caveat_note(status)


def test_unknown_reaction_baseline_does_not_crash_collect_path():
    # reactions outside 1-5 have no REACTION_BOXES entry; the history code must not index it
    assert compound.REACTION_BOXES.get(22, (None, None, 0.0))[2] == 0.0


def test_sanqi_boxes_count_as_compound_spend():
    from nanobody_dash import infra

    assert infra.cost_scope("nova-sanqi-4") == "sn68-compound"
    assert infra.cost_scope("nova-rxn3-nim-4x4090") == "sn68-compound"
    assert infra.cost_scope("nova-sanqi-extra-test") == "research"
