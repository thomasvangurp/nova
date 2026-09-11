import json

import pandas as pd

from nanobody_opt import backtest_budget_allocation as budget
from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL


METRICS = {
    "design_iiptm": 0.78,
    "design_ptm": 0.88,
    "design_to_target_iptm": 0.68,
    "min_design_to_target_pae": 2.2,
    "interaction_pae": 10.5,
    "plip_hbonds_refolded": 5.0,
    "plip_saltbridge_refolded": 2.0,
    "delta_sasa_refolded": 740.0,
    "liability_score": 265.0,
    "liability_num_violations": 29.0,
}


def _exact_document(sequences):
    return {
        "target": "P20809",
        "scorer_protocol": SCORER_PROTOCOL,
        "n": len(sequences),
        "per_nanobody_components": {
            "0": {sequence: {"P20809": dict(METRICS)} for sequence in sequences}
        },
    }


def test_collect_observations_separates_screens_from_replicates(tmp_path):
    race = tmp_path / "epoch25052" / "producer" / "iteration_01" / "race"
    screen = race / "screen" / "tranche_01"
    screen.mkdir(parents=True)
    (screen / "exact_all.json").write_text(
        json.dumps(_exact_document(["AAA", "BBB"]))
    )
    for replicate in (2, 3):
        repeat = race / "candidates" / ("a" * 8) / f"repeat_{replicate:02d}"
        repeat.mkdir(parents=True)
        (repeat / "exact_all.json").write_text(json.dumps(_exact_document(["AAA"])))

    frame = budget.collect_observations(tmp_path)

    assert len(frame) == 4
    assert sorted(frame.kind.unique()) == ["repeat", "screen"]
    assert set(frame[frame.kind == "repeat"].replicate) == {2, 3}
    assert set(frame.epoch) == {25052}


def test_replication_beats_pure_breadth_at_an_equal_budget():
    # With observation noise comparable to the spread of true candidate quality,
    # selecting the best of many single observations mostly selects noise.
    breadth_only = budget.two_stage_expectation(
        1000, 1, 1, sigma_quality=30.0, sigma_observation=34.0, trials=1500
    )
    two_stage = budget.two_stage_expectation(
        800, 25, 9, sigma_quality=30.0, sigma_observation=34.0, trials=1500
    )

    assert two_stage < breadth_only - 5.0


def test_more_replicates_never_hurt_the_selected_candidate():
    single = budget.two_stage_expectation(
        400, 20, 1, sigma_quality=30.0, sigma_observation=34.0, trials=1500
    )
    replicated = budget.two_stage_expectation(
        400, 20, 8, sigma_quality=30.0, sigma_observation=34.0, trials=1500
    )

    assert replicated < single


def test_wall_clock_plan_respects_the_epoch_window():
    plans = budget.wall_clock_plan(
        sigma_quality=30.0,
        sigma_observation=34.0,
        load_seconds=33.0,
        seconds_per_sequence=14.5,
        gpu_count=10,
        window_seconds=3672.0,
        bookkeeping_seconds=84.0,
        overlap=True,
        trials=400,
    )

    assert plans
    assert all(plan["wall_seconds"] <= 3672.0 for plan in plans)
    assert all(plan["replicates"] >= 2 for plan in plans)


def test_serial_bookkeeping_buys_fewer_replication_rounds():
    shared = dict(
        sigma_quality=30.0,
        sigma_observation=34.0,
        load_seconds=33.0,
        seconds_per_sequence=14.5,
        gpu_count=10,
        window_seconds=3672.0,
        bookkeeping_seconds=84.0,
        trials=400,
    )
    overlapped = budget.wall_clock_plan(overlap=True, **shared)[0]
    serial = budget.wall_clock_plan(overlap=False, **shared)[0]

    assert overlapped["observations"] > serial["observations"]


def test_measured_policy_reports_what_was_actually_spent():
    frame = pd.DataFrame(
        [
            {"epoch": 1, "kind": "screen", "replicate": 1, "sequence": "A"},
            {"epoch": 1, "kind": "screen", "replicate": 1, "sequence": "B"},
            {"epoch": 1, "kind": "repeat", "replicate": 2, "sequence": "A"},
            {"epoch": 1, "kind": "repeat", "replicate": 3, "sequence": "A"},
        ]
    )

    measured = budget.measured_policy(frame)

    assert measured["breadth"] == 2.0
    assert measured["shortlist"] == 1.0
    assert measured["replicates"] == 2.0
