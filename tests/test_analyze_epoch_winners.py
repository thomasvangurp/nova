import pandas as pd

from nanobody_opt.analyze_epoch_winners import (
    analyze_epoch_groups,
    parse_epoch_commits,
    partition_recent_epochs,
    rank_epoch,
)
from nanobody_opt.simulate_round import METRICS


def _row(sequence: str, value: float) -> dict:
    row = {"sequence": sequence, "sequence_hash": sequence}
    for metric, direction in METRICS.items():
        row[metric] = value if direction == "max" else -value
    return row


def test_parses_only_requested_target_batches():
    commits = [
        {"id": "a", "date": "now", "title":
         "Added 42 nanobody sequences from epoch 25022 to P20809_nanobodies.csv"},
        {"id": "b", "date": "now", "title":
         "Added 20 nanobody sequences from epoch 25022 to OTHER_nanobodies.csv"},
        {"id": "c", "date": "now", "title":
         "Added 500 molecules from epoch 25022 to P40261_molecules.csv"},
    ]
    assert parse_epoch_commits(commits, "P20809_nanobodies.csv") == [{
        "epoch": 25022, "count": 42, "commit": "a", "date": "now"
    }]


def test_partitions_newest_batches_from_archive_tail():
    frame = pd.DataFrame({
        "sequence": list("ABCDEF"),
        "sequence_hash": list("abcdef"),
    })
    batches = [{"epoch": 2, "count": 2}, {"epoch": 1, "count": 3}]
    groups = partition_recent_epochs(frame, batches)
    assert groups[0][1].sequence.tolist() == ["E", "F"]
    assert groups[1][1].sequence.tolist() == ["B", "C", "D"]


def test_epoch_rank_sum_and_family_summary():
    newer = pd.DataFrame([_row("AAAA", 2), _row("CCCC", 1)])
    older = pd.DataFrame([_row("AAAT", 3), _row("GGGG", 1)])
    ranked = rank_epoch(newer)
    assert ranked.iloc[0].sequence == "AAAA"
    assert ranked.iloc[0].epoch_rank_sum == 10

    result = analyze_epoch_groups([
        ({"epoch": 2, "count": 2, "commit": "x"}, newer),
        ({"epoch": 1, "count": 2, "commit": "y"}, older),
    ])
    assert [item["sequence"] for item in result["winners"]] == ["AAAA", "AAAT"]
    assert result["winner_summary"]["pairwise_identity"]["median"] == 0.75
