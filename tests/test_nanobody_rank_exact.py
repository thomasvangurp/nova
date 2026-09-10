import json
import sys

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
        assert "no complete exact candidates" in str(error)
    else:
        raise AssertionError("four good and four bad observations were accepted")
