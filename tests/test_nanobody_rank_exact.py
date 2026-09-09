import json
import sys

import pandas as pd

from nanobody_opt import rank_exact
from nanobody_opt.simulate_round import METRICS


def _metrics(value):
    return {name: float(value) for name in METRICS}


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
