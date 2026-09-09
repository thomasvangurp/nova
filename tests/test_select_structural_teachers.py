import pandas as pd
import pytest

from nanobody_opt.select_structural_teachers import (
    STRUCTURAL_METRICS,
    select_teachers,
)


def _row(sequence: str, value: float) -> dict:
    row = {"sequence": sequence}
    for metric, direction in STRUCTURAL_METRICS.items():
        row[metric] = value if direction == "max" else 10 - value
    return row


def test_selects_recent_structural_leaders_and_normalizes_sequences():
    old_best = _row("A" * 110, 10)
    recent_worst = _row("C" * 110, 1)
    recent_best = _row("d" * 110 + " ", 8)
    middle = _row("E" * 110, 4)
    frame = pd.DataFrame([old_best, recent_worst, recent_best, middle])

    got = select_teachers(frame, recent=3, count=2)

    assert got.sequence.tolist() == ["D" * 110, "E" * 110]
    assert got.structural_rank_sum.tolist() == pytest.approx([8.0, 16.0])


def test_rejects_missing_metrics():
    with pytest.raises(ValueError, match="required columns"):
        select_teachers(pd.DataFrame({"sequence": ["A"]}), recent=1, count=1)
