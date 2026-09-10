import pandas as pd

from nanobody_opt.generate_donor_grafts import conservative_substitution, select_donors
from nanobody_opt.simulate_round import METRICS


def _row(sequence: str, liability: float, quality: float) -> dict:
    row = {
        "sequence": sequence,
        "liability_score": liability,
        "liability_num_violations": liability / 10,
    }
    for metric, direction in METRICS.items():
        if metric in row:
            continue
        row[metric] = quality if direction == "max" else -quality
    return row


def test_selects_same_length_low_liability_measured_donors():
    frame = pd.DataFrame([
        _row("AAAA", 170, 3),
        _row("CCCC", 175, 2),
        _row("DDDD", 190, 4),
        _row("AAAAA", 165, 5),
    ])
    got = select_donors(frame, teacher_length=4, max_liability_score=180, limit=10)
    assert got.sequence.tolist() == ["AAAA", "CCCC"]


def test_conservative_substitution_classes_are_bounded():
    assert conservative_substitution("W", "Y")
    assert conservative_substitution("M", "L")
    assert conservative_substitution("K", "H")
    assert conservative_substitution("N", "Q")
    assert not conservative_substitution("K", "L")
    assert not conservative_substitution("C", "A")
    assert not conservative_substitution("W", "W")
