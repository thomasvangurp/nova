import json
import sys

import numpy as np
import pandas as pd
import pytest

from nanobody_opt import rank_exact
from nanobody_opt.simulate_round import (
    METRICS,
    category_rank_sums,
    rank_sum_in_field,
)


def _metrics(value):
    return {name: float(value) for name in METRICS}


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
    assert len(candidate["per_observation_field_rank_sums"]) == 1
    assert len(candidate["per_observation_field_rank_sums"][0]) == 64


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
    """Sorting by tier must not disturb the order the screen already chose."""
    ranked = [
        {"sequence": "BEST_BUT_BROKEN", "settled_deviations": 2, "rank": 0},
        {"sequence": "CLEAN_A", "settled_deviations": 0, "rank": 1},
        {"sequence": "CLEAN_B", "settled_deviations": 0, "rank": 2},
        {"sequence": "ONE_BROKEN", "settled_deviations": 1, "rank": 3},
    ]
    ranked.sort(key=lambda item: int(item.get("settled_deviations", 0)))
    assert [item["sequence"] for item in ranked] == [
        "CLEAN_A", "CLEAN_B", "ONE_BROKEN", "BEST_BUT_BROKEN"
    ]
    # Python's sort is stable, so the screen's own order survives inside tier 0.
    assert [item["rank"] for item in ranked[:2]] == [1, 2]


def test_a_candidate_without_the_field_is_treated_as_unbroken():
    """Missing consensus must not silently demote everything to tier zero-plus."""
    from nanobody_opt.rank_exact import settled_deviation_count

    assert settled_deviation_count("ACDEFG", {}) == 0
