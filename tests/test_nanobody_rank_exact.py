import json
import sys

import numpy as np
import pandas as pd
import pytest

from nanobody_opt import db, rank_exact
from nanobody_opt.mutation_network import record_exact_run, update_observation_store
from nanobody_opt.simulate_round import (
    METRICS,
    category_rank_sums,
    rank_sum_in_field,
)


def test_diversity_reserve_reaches_beyond_a_clustered_prefix():
    ranked = [
        {"sequence": "AAAAAAAAAAAAAAAAAAAA", "rank": 1},
        {"sequence": "AAAAAAAAAAAAAAAAAAAC", "rank": 2},
        {"sequence": "CCCCCCCCCCCCCCCCCCCC", "rank": 3},
        {"sequence": "DDDDDDDDDDDDDDDDDDDD", "rank": 4},
    ]

    selected, added = rank_exact._with_diversity_reserve(
        ranked, ranked[:2], 3
    )

    assert [item["rank"] for item in selected] == [1, 2, 3, 4]
    assert added == 2


def _metrics(value):
    return {name: float(value) for name in METRICS}


def test_database_history_reads_only_complete_normalized_raw_runs(tmp_path):
    database = tmp_path / "observations.sqlite"
    db.connect_local(database)
    raw_id = "local-exact:sealed:0:RAW"
    partial_id = "local-exact:partial:0:PARTIAL"
    update_observation_store(database, [
        {
            "observation_id": raw_id,
            "sequence": "RAW",
            "score": 20,
            "score_kind": "local_exact_validator_replay_rank_sum",
            "source": "local_exact_raw",
            "target": "P20809",
            "metrics": _metrics(2),
        },
        {
            "observation_id": partial_id,
            "sequence": "PARTIAL",
            "score": 30,
            "score_kind": "local_exact_validator_replay_rank_sum",
            "source": "local_exact_raw",
            "target": "P20809",
            "metrics": _metrics(3),
        },
        {
            "observation_id": "live_exact_screen:epoch1:DERIVED",
            "sequence": "DERIVED",
            "score": 1,
            "score_kind": "local_validator_replay_rank_sum",
            "source": "live_exact_screen",
            "target": "P20809",
            "metrics": _metrics(1),
        },
    ])
    record_exact_run(
        database,
        target="P20809",
        run_id="sealed",
        epoch=1,
        source_path=tmp_path / "sealed.json",
        file_sha256="sealed-sha",
        scorer_protocol="validator-fast-v1",
        observation_ids=[raw_id],
    )
    record_exact_run(
        database,
        target="P20809",
        run_id="partial",
        epoch=1,
        source_path=tmp_path / "partial.json",
        file_sha256="partial-sha",
        scorer_protocol="validator-fast-v1",
        observation_ids=[partial_id, "missing"],
    )

    found, added = rank_exact._historical_observations(
        database, "P20809", {}
    )

    assert found == {"RAW": [_metrics(2)]}
    assert added == 1


def test_database_history_deduplicates_a_current_raw_run(tmp_path):
    database = tmp_path / "observations.sqlite"
    db.connect_local(database)
    observation_id = "local-exact:same-run:0:RAW"
    update_observation_store(database, [{
        "observation_id": observation_id,
        "sequence": "RAW",
        "score": 20,
        "score_kind": "local_exact_validator_replay_rank_sum",
        "source": "local_exact_raw",
        "target": "P20809",
        "metrics": _metrics(2),
    }])
    record_exact_run(
        database,
        target="P20809",
        run_id="same-run",
        epoch=1,
        source_path=tmp_path / "same.json",
        file_sha256="same-sha",
        scorer_protocol=None,
        observation_ids=[observation_id],
    )

    found, added = rank_exact._historical_observations(
        database, "P20809", {}, excluded_run_ids={"same-run"}
    )

    assert found == {}
    assert added == 0


def test_rank_sum_uses_validator_dense_ties():
    # With average ranks five identical rows would score 30 total. The validator
    # assigns every tied value dense rank 1, split 5 + 3 + 2 by category.
    mine = _metrics(1)
    field = pd.DataFrame([_metrics(1) for _ in range(4)])

    assert category_rank_sums(mine, field) == {
        "confidence": 5.0,
        "physical_interaction": 3.0,
        "developability": 2.0,
    }
    assert rank_sum_in_field(mine, field) == 10.0


def test_compiled_dense_ranks_match_validator_randomized_fields():
    rng = np.random.default_rng(20260911)
    for _ in range(200):
        size = int(rng.integers(1, 100))
        field = pd.DataFrame({
            metric: rng.choice(
                np.asarray([-np.inf, -2.0, -0.0, 0.0, 1.0, 4.0, np.inf]),
                size=size,
                replace=True,
            )
            for metric in METRICS
        })
        for metric in METRICS:
            field.loc[rng.random(size) < 0.1, metric] = np.nan
        mine = {
            metric: float(rng.choice([-3.0, -0.0, 0.0, 1.0, 5.0]))
            for metric in METRICS
        }

        expected = category_rank_sums(mine, field)
        actual = rank_exact._compiled_category_rank_sums(
            mine, rank_exact._compile_validator_field(field)
        )

        assert actual == expected
        assert sum(actual.values()) == rank_sum_in_field(mine, field)


def test_dashboard_payload_becomes_unique_validator_field():
    first = _metrics(1)
    second = _metrics(2)
    payload = {
        "success": True,
        "data": [
            {"nanobodies": [{
                **first,
                "protein_name": "P20809",
                "sequence": "AAAA",
                "sequence_hash": "same",
            }]},
            {"nanobodies": [{
                **second,
                "protein_name": "P20809",
                "sequence": "AAAA",
                "sequence_hash": "same",
            }]},
            {"nanobodies": [{
                **second,
                "protein_name": "OTHER",
                "sequence": "BBBB",
                "sequence_hash": "other",
            }]},
        ],
    }

    field = rank_exact.leaderboard_field(payload, "P20809")

    assert len(field) == 1
    assert field.iloc[0].to_dict() == first


def test_duplicate_scoring_sweep_is_only_one_observation(tmp_path):
    payload = {
        "target": "P20809",
        "observation_id": "same-sweep",
        "per_nanobody_components": {
            "0": {"FRESH": {"P20809": _metrics(1)}}
        },
    }
    first = tmp_path / "first.json"
    duplicate = tmp_path / "duplicate.json"
    first.write_text(json.dumps(payload))
    payload["per_nanobody_components"]["0"]["FRESH"]["P20809"] = _metrics(9)
    duplicate.write_text(json.dumps(payload))

    observations = rank_exact._observations([first, duplicate], "P20809")

    assert observations == {"FRESH": [_metrics(1)]}


def test_published_metric_vector_uses_validator_score_share_mean(
    tmp_path, monkeypatch
):
    archive = tmp_path / "archive.csv"
    pd.DataFrame([
        {"sequence": f"ARCHIVE{index:02d}", **_metrics(index + 1)}
        for index in range(60)
    ]).to_csv(archive, index=False)
    exact_args = []
    for index, value in enumerate((1, 1, 10)):
        path = tmp_path / f"exact-{index}.json"
        path.write_text(json.dumps({
            "target": "P20809",
            "observation_id": f"observation-{index}",
            "per_nanobody_components": {
                "0": {"FRESH": {"P20809": _metrics(value)}}
            },
        }))
        exact_args.extend(["--exact", str(path)])
    output = tmp_path / "ranking.json"
    monkeypatch.setattr(sys, "argv", [
        "rank_exact.py", *exact_args,
        "--archive", str(archive),
        "--target", "P20809",
        "--output", str(output),
        "--recent", "50",
        "--field-size", "10",
        "--minimum-observations", "3",
    ])

    rank_exact.main()

    candidate = json.loads(output.read_text())["candidates"][0]
    assert candidate["metric_aggregation"] == "arithmetic_mean"
    assert set(candidate["metrics"].values()) == {4.0}
    assert list(candidate["metric_standard_errors"].values()) == pytest.approx(
        [3.0] * len(METRICS)
    )


def test_ranker_excludes_full_history_and_published_ledger(tmp_path, monkeypatch):
    archive = tmp_path / "archive.csv"
    rows = [
        {"sequence": f"ARCHIVE{index:02d}", **_metrics(index + 1)}
        for index in range(60)
    ]
    rows[0]["sequence"] = "HISTORICAL"
    pd.DataFrame(rows).to_csv(archive, index=False)

    exact = tmp_path / "exact.json"
    exact.write_text(
        json.dumps(
            {
                "target": "P20809",
                "per_nanobody_components": {
                    "0": {
                        sequence: {"P20809": _metrics(value)}
                        for sequence, value in (
                            ("HISTORICAL", 1),
                            ("PUBLISHED", 2),
                            ("FRESH", 3),
                        )
                    }
                },
            }
        )
    )
    ledger = tmp_path / "published.txt"
    ledger.write_text("# prior epochs\nPUBLISHED\n")
    output = tmp_path / "ranking.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rank_exact.py",
            "--exact",
            str(exact),
            "--archive",
            str(archive),
            "--target",
            "P20809",
            "--output",
            str(output),
            "--exclude-file",
            str(ledger),
            "--recent",
            "50",
            "--field-size",
            "10",
        ],
    )

    rank_exact.main()

    payload = json.loads(output.read_text())
    assert payload["candidate_count"] == 1
    assert payload["sequences"] == ["FRESH"]
    assert payload["excluded_sequence_count"] == 1
    assert payload["rank_method"] == "dense"
    assert payload["selection_order"] == "competitive-rate"
    assert payload["qualification"] == {
        "competitive_score_threshold": 75.0,
        "minimum_competitive_rate": 0.0,
        "minimum_observation_competitive_rate": 0.0,
        "maximum_p90_rank_sum": None,
        "maximum_p90_category_rank_sums": {
            "confidence": None,
            "physical_interaction": None,
            "developability": None,
        },
        "pose_stability": {
            "minimum_structure_observations": 0,
            "minimum_median_contact_jaccard": 0.0,
            "maximum_cdr3_rmsd": None,
            "minimum_epitope_occupancy": 0.0,
        },
    }
    candidate = payload["candidates"][0]
    assert 0 <= candidate["competitive_rate"] <= 1
    assert 0 <= candidate["observation_competitive_rate"] <= 1
    assert candidate["selection_score"] == candidate["score_p90"]
    assert "per_observation_field_rank_sums" not in candidate
    assert candidate["field_replay_matrix_shape"] == [1, 64]


def test_ranker_rejects_structurally_unstable_observations(tmp_path, monkeypatch):
    archive = tmp_path / "archive.csv"
    pd.DataFrame([
        {"sequence": f"ARCHIVE{index:02d}", **_metrics(index + 1)}
        for index in range(60)
    ]).to_csv(archive, index=False)
    good = {
        name: (100.0 if direction == "max" else 0.0)
        for name, direction in rank_exact.METRICS.items()
    }
    bad = {
        name: (0.0 if direction == "max" else 100.0)
        for name, direction in rank_exact.METRICS.items()
    }
    exact_paths = []
    for index, metrics in enumerate([good] * 4 + [bad] * 4):
        path = tmp_path / f"exact-{index}.json"
        path.write_text(json.dumps({
            "target": "P20809",
            "observation_id": f"observation-{index}",
            "per_nanobody_components": {
                "0": {"UNSTABLE": {"P20809": metrics}}
            },
        }))
        exact_paths.extend(["--exact", str(path)])
    output = tmp_path / "ranking.json"
    monkeypatch.setattr(sys, "argv", [
        "rank_exact.py", *exact_paths,
        "--archive", str(archive),
        "--target", "P20809",
        "--output", str(output),
        "--recent", "50",
        "--field-size", "10",
        "--minimum-observations", "8",
        "--minimum-observation-competitive-rate", "0.75",
    ])

    try:
        rank_exact.main()
    except SystemExit as error:
        assert "cleared qualification" in str(error)
    else:
        raise AssertionError("four good and four bad observations were accepted")


def test_selection_orders_on_the_plain_rank_sum():
    """
    Selection orders on the unweighted sum of dense ranks -- the same quantity
    the validator's own score is built from.

    A weighted variant ran for eleven epochs in two shapes and was reverted:
    best-rank medians across unweighted, weighted, and weighted-plus-generator-
    fixes were 3, 8 and 18, mean confidence gaps +3, +27 and +25, and the only
    win in that stretch belongs to the unweighted ordering. Reintroducing a
    weighting needs evidence that did not exist then, so this pins the contract.
    """
    import numpy as np
    from nanobody_opt.rank_exact import (
        _compiled_category_rank_sums,
        _compiled_metric_ranks,
    )
    from nanobody_opt.simulate_round import METRIC_CATEGORIES, METRICS

    rng = np.random.default_rng(23)
    field = {metric: np.sort(rng.normal(size=64)) for metric in METRICS}
    point = {metric: float(rng.normal()) for metric in METRICS}

    ranks = _compiled_metric_ranks(point, field)
    categories = _compiled_category_rank_sums(point, field)

    # Every category is a plain sum of its members' dense ranks...
    for category, names in METRIC_CATEGORIES.items():
        assert categories[category] == pytest.approx(sum(ranks[n] for n in names))
    # ...and the score the fleet is ordered by is the plain sum of those, with
    # no metric counting for more than any other.
    assert sum(categories.values()) == pytest.approx(sum(ranks.values()))


def test_no_per_metric_weighting_remains():
    """The weighting is gone, not merely unused."""
    from nanobody_opt import rank_exact

    assert not hasattr(rank_exact, "selection_metric_weight")
    assert not hasattr(rank_exact, "SELECTION_METRIC_WEIGHTS")
    assert not hasattr(rank_exact, "_compiled_selection_score")



def _exact_bank(tmp_path, observations_by_sequence):
    """An exact score bank holding N independent observations per sequence."""
    depth = max(observations_by_sequence.values())
    path = tmp_path / "exact.json"
    path.write_text(json.dumps({
        "target": "P20809",
        "per_nanobody_components": {
            str(index): {
                sequence: {"P20809": _metrics(value)}
                for value, (sequence, count) in enumerate(
                    sorted(observations_by_sequence.items()), start=1
                )
                if index < count
            }
            for index in range(depth)
        },
    }))
    return path


def _rank(tmp_path, monkeypatch, exact, **overrides):
    archive = tmp_path / "archive.csv"
    pd.DataFrame([
        {"sequence": f"ARCHIVE{index:02d}", **_metrics(index + 1)}
        for index in range(60)
    ]).to_csv(archive, index=False)
    output = tmp_path / "ranking.json"
    argv = [
        "rank_exact.py",
        "--exact", str(exact),
        "--archive", str(archive),
        "--target", "P20809",
        "--output", str(output),
        "--recent", "50",
        "--field-size", "10",
    ]
    for flag, value in overrides.items():
        argv += [flag, str(value)]
    monkeypatch.setattr(sys, "argv", argv)
    rank_exact.main()
    return json.loads(output.read_text())


def test_a_thin_candidate_loses_to_a_well_observed_one(tmp_path, monkeypatch):
    # THIN has one observation and SOLID has four.  At the floor the thin
    # candidate is not eligible at all, however good its single draw looked.
    exact = _exact_bank(tmp_path, {"SOLID": 4, "THIN": 1})

    payload = _rank(tmp_path, monkeypatch, exact, **{"--minimum-observations": 4})

    assert payload["sequences"] == ["SOLID"]
    assert payload["observation_floor_applied"] == 4


def test_a_thin_epoch_still_publishes_at_a_relaxed_floor(tmp_path, monkeypatch):
    # Nothing reaches the floor.  Publishing nothing is the only outcome that
    # certainly pays nothing, so the floor gives way rather than the epoch.
    exact = _exact_bank(tmp_path, {"THIN": 1, "ALSOTHIN": 1})

    payload = _rank(tmp_path, monkeypatch, exact, **{"--minimum-observations": 4})

    assert payload["candidate_count"] == 2
    assert payload["minimum_observations"] == 4
    assert payload["observation_floor_applied"] == 1


def test_the_floor_gives_way_one_observation_at_a_time(tmp_path, monkeypatch):
    # Three observations is the deepest evidence this epoch produced, so that is
    # where the relaxation stops -- it does not fall straight to one and let a
    # single-draw candidate back into the comparison.
    exact = _exact_bank(tmp_path, {"DEEPEST": 3, "SHALLOW": 1})

    payload = _rank(tmp_path, monkeypatch, exact, **{"--minimum-observations": 8})

    assert payload["sequences"] == ["DEEPEST"]
    assert payload["observation_floor_applied"] == 3


def test_a_molecule_that_breaks_nothing_outranks_a_better_scoring_one():
    """Settled-site breakage is the primary key; the screen orders within a tier.

    Measured over 1,427 live tickets each settled deviation costs ~14 rank
    points, while our screened metrics agree with the validator at r=0.19.  The
    coarse structural signal is the more reliable of the two, so it decides
    which uid gets spent; the screen still decides everything inside a tier.
    """
    from nanobody_opt.rank_exact import settled_deviation_count

    consensus = {0: ("A", 0.95), 1: ("C", 0.95), 2: ("D", 0.40)}
    assert settled_deviation_count("ACD", consensus) == 0
    assert settled_deviation_count("AED", consensus) == 1
    assert settled_deviation_count("EED", consensus) == 2
    # An unsettled position is never breakage, whatever it holds.
    assert settled_deviation_count("ACW", consensus) == 0
    # A short sequence cannot break positions it does not reach.
    assert settled_deviation_count("A", consensus) == 0


def test_the_deviation_tier_is_stable_within_itself():
    """Sorting by tier must not disturb the order the screen already chose.

    Exercises the ranker's own key.  This test used to re-implement the lambda
    on its own fixtures, so it passed while the ranker emitted an order the
    publisher refused and the fleet committed nothing for an epoch.
    """
    from nanobody_opt.rank_exact import win_probability_order_key

    ranked = [
        {"sequence": "BEST_BUT_BROKEN", "settled_deviations": 2,
         "win_probability": 0.9, "rank": 0},
        {"sequence": "CLEAN_A", "settled_deviations": 0,
         "win_probability": 0.3, "rank": 1},
        {"sequence": "CLEAN_B", "settled_deviations": 0,
         "win_probability": 0.2, "rank": 2},
        {"sequence": "ONE_BROKEN", "settled_deviations": 1,
         "win_probability": 0.5, "rank": 3},
    ]
    ranked.sort(key=win_probability_order_key)
    assert [item["sequence"] for item in ranked] == [
        "CLEAN_A", "CLEAN_B", "ONE_BROKEN", "BEST_BUT_BROKEN"
    ]
    # Python's sort is stable, so the screen's own order survives inside tier 0.
    assert [item["rank"] for item in ranked[:2]] == [1, 2]


def test_a_clean_candidate_that_cannot_win_does_not_displace_one_that_can():
    """The deviation rule chooses between molecules that can win.

    Tier-first alone promoted a zero-win-probability molecule over a 0.0148
    one, which is spending a uid on a certainty of nothing -- and it is the
    ordering the publisher then refused outright.
    """
    from nanobody_opt.rank_exact import win_probability_order_key

    ranked = [
        {"sequence": "CLEAN_NO_CHANCE", "settled_deviations": 0,
         "win_probability": 0.0},
        {"sequence": "BROKEN_BUT_LIVE", "settled_deviations": 1,
         "win_probability": 0.0148},
    ]
    ranked.sort(key=win_probability_order_key)
    assert [item["sequence"] for item in ranked] == [
        "BROKEN_BUT_LIVE", "CLEAN_NO_CHANCE"
    ]


def _pose_order_candidate(sequence, score, observations=1, deviations=0):
    scores = [float(score + index) for index in range(observations)]
    return {
        "sequence": sequence,
        "settled_deviations": deviations,
        "per_observation_median_rank_sum": scores,
        "score_p90": max(scores),
        "score_median": float(np.median(scores)),
        "score_worst": max(scores),
        "observation_competitive_rate": 1.0,
        "competitive_rate": 1.0,
    }


def test_live_win_probability_is_the_empirical_fresh_draw_event():
    candidates = [
        _pose_order_candidate("RELIABLE", 2, observations=3),
        {
            **_pose_order_candidate("LOTTERY", 1, observations=1),
            "per_observation_median_rank_sum": [1.0, 50.0, 100.0],
            "score_median": 50.0,
            "score_p90": 100.0,
            "score_worst": 100.0,
        },
    ]

    ranked = rank_exact._order_candidates(
        candidates,
        "win-probability",
        competitive_score_threshold=1.5,
        consensus_ranking_policy="off",
    )

    assert [item["sequence"] for item in ranked] == ["RELIABLE", "LOTTERY"]
    assert ranked[0]["win_probability_method"] == (
        "empirical-independent-fresh-draw"
    )


def test_one_observation_pose_shortcut_preserves_the_exhaustive_selected_set():
    candidates = [
        _pose_order_candidate("A", 44, deviations=1),
        _pose_order_candidate("B", 50),
        _pose_order_candidate("C", 55),
        _pose_order_candidate("D", 60),
    ]
    pose_survivors = {"A", "C", "D"}
    threshold = 75.0
    assert rank_exact._pose_order_is_subset_invariant(
        candidates, "win-probability", threshold
    )

    exhaustive = rank_exact._order_candidates(
        [item for item in candidates if item["sequence"] in pose_survivors],
        "win-probability",
        threshold,
    )[:2]
    preliminary = rank_exact._order_candidates(
        candidates, "win-probability", threshold
    )
    accepted = []
    for item in preliminary:
        if item["sequence"] in pose_survivors:
            accepted.append(item)
        if len(accepted) == 2:
            break
    shortcut = rank_exact._order_candidates(
        accepted, "win-probability", threshold
    )

    assert [item["sequence"] for item in shortcut] == [
        item["sequence"] for item in exhaustive
    ]


def test_mixed_depth_win_probability_keeps_exhaustive_pose_evaluation():
    candidates = [
        _pose_order_candidate("ONE", 45, observations=1),
        _pose_order_candidate("TWO", 46, observations=2),
    ]
    assert not rank_exact._pose_order_is_subset_invariant(
        candidates, "win-probability", 75.0
    )


def test_strict_rank_defers_writer_pose_and_stops_safe_confidence_walk(
    tmp_path, monkeypatch
):
    exact = _exact_bank(tmp_path, {"FIRST": 1, "SECOND": 1, "THIRD": 1})
    config = tmp_path / "config.yaml"
    config.write_text(
        "protein_selection:\n"
        "  nanobody:\n"
        "    target: P20809\n"
        "    structure_chain: A\n"
        "    structure_binding_site: 1..3\n"
    )

    def retained_paths(_paths, _target, *, pose_kind="confidence"):
        return {
            sequence: [tmp_path / f"{pose_kind}-{sequence}.cif"]
            for sequence in ("FIRST", "SECOND", "THIRD")
        }

    calls = []

    def summarize(sequence, paths, **_kwargs):
        calls.append((sequence, paths[0].name))
        return {
            "structure_observation_count": 1,
            "contact_jaccard_median": 1.0,
            "cdr3_rmsd_max_angstrom": 0.0,
            "epitope_occupancy_min": 1.0,
        }

    monkeypatch.setattr(rank_exact, "exact_structure_paths", retained_paths)
    monkeypatch.setattr(rank_exact, "exact_sample_selections", lambda *_: {})
    monkeypatch.setattr(rank_exact, "summarize_sequence_structures", summarize)
    payload = _rank(
        tmp_path,
        monkeypatch,
        exact,
        **{
            "--pose-config": config,
            "--minimum-structure-observations": 1,
            "--selection-order": "win-probability",
            "--limit": 1,
        },
    )

    assert len(payload["sequences"]) == 1
    confidence = [call for call in calls if call[1].startswith("confidence-")]
    writer = [call for call in calls if call[1].startswith("writer-")]
    assert len(confidence) == 1
    assert len(writer) == 1
    assert confidence[0][0] == writer[0][0] == payload["sequences"][0]


def test_the_ranker_order_satisfies_the_publisher_invariant():
    """The two must agree, or every publication cycle dies on the assert."""
    from nanobody_opt import fleet
    from nanobody_opt.rank_exact import _order_candidates

    ranked = [
        {"sequence": "A", "multipass_rank": 300.0, "score_median": 1.0},
        {"sequence": "B", "multipass_rank": 100.0, "score_median": 2.0},
        {"sequence": "C", "multipass_rank": 200.0, "score_median": 3.0},
        {"sequence": "D", "multipass_rank": 900.0, "score_median": 4.0},
        {"sequence": "E", "multipass_rank": 400.0, "score_median": 5.0},
    ]
    ranked = _order_candidates(ranked, "multipass", competitive_score_threshold=0.0)
    fleet._assert_non_decreasing_multipass_rank(ranked)


def test_a_candidate_without_the_field_is_treated_as_unbroken():
    """Missing consensus must not silently demote everything to tier zero-plus."""
    from nanobody_opt.rank_exact import settled_deviation_count

    assert settled_deviation_count("ACDEFG", {}) == 0


def _order(*rows):
    return [{"sequence": name, "multipass_rank": rank}
            for name, rank in rows]


def test_a_multipass_rank_regression_is_refused():
    """The check must still fail on a real regression, or the rewrite to the
    multipass invariant would have turned it into decoration."""
    from nanobody_opt import fleet

    with pytest.raises(fleet.SelectionOrderError, match="multipass_rank"):
        fleet._assert_non_decreasing_multipass_rank(
            _order(("A", 100.0), ("B", 50.0)))


def test_a_monotonic_multipass_file_is_accepted():
    """Best-first is multipass_rank ascending, so non-decreasing is correct."""
    from nanobody_opt import fleet

    fleet._assert_non_decreasing_multipass_rank(
        _order(("A", 217.5), ("B", 218.0), ("C", 900.0)))


def test_the_ranker_and_the_invariant_agree_on_a_randomised_corpus():
    """Property check over the real order key, not a hand-picked fixture.

    The ranker sorts multipass_rank ascending (rank_exact._order_candidates,
    selection_order="multipass"); the fleet invariant asserts non-decreasing.
    The two definitions drifted once before and cost an epoch; a fixture only
    covers the shapes someone thought of.
    """
    import random

    from nanobody_opt import fleet

    generator = random.Random(20260920)
    for _ in range(300):
        rows = [
            {"sequence": f"S{index}",
             "multipass_rank": generator.choice(
                 [100.0, 200.0, 300.0, 400.0, 500.0])}
            for index in range(generator.randint(2, 12))
        ]
        rows.sort(key=lambda item: float(item["multipass_rank"]))
        fleet._assert_non_decreasing_multipass_rank(rows)
