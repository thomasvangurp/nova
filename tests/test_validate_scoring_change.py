"""Tests for tools/validate_scoring_change.py (the scoring-change gate).

Offline tests use a saved leaderboard epoch. The two known-answer cases from
2026-10-09 need the public API and are opt-in: NOVA_NETWORK_TESTS=1.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "scoring_validation"
_spec = importlib.util.spec_from_file_location(
    "validate_scoring_change", ROOT / "tools" / "validate_scoring_change.py")
vsc = importlib.util.module_from_spec(_spec)
sys.modules["validate_scoring_change"] = vsc
_spec.loader.exec_module(vsc)


@pytest.fixture(scope="module")
def epoch_field():
    return vsc.load_field({25613: FIX / "leaderboard_25613.json"})


def test_sum10_reproduces_published_final_score_exactly(epoch_field):
    scores = vsc.f_sum10(epoch_field)
    assert len(epoch_field) > 30
    np.testing.assert_allclose(scores, epoch_field.final_score.to_numpy())


def test_category3_is_not_the_validator_formula(epoch_field):
    scores = vsc.f_category3(epoch_field)
    assert vsc.spearman(scores, epoch_field.final_score.to_numpy()) < 0.99


def test_csv_source_and_higher_is_better(tmp_path, epoch_field):
    csv = tmp_path / "s.csv"
    pd.DataFrame({"sequence": epoch_field.sequence.str.lower(),
                  "v": -epoch_field.final_score}).to_csv(csv, index=False)
    vals = vsc.apply_source(vsc.parse_source(f"csv:{csv}:v@higher"), epoch_field)
    np.testing.assert_allclose(vals.to_numpy(), epoch_field.final_score.to_numpy())


def _synthetic(n_epochs=20, per_epoch=12, noise_inc=0.0, noise_pro=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_epochs):
        truth = rng.normal(size=per_epoch)
        for t in truth:
            rows.append(dict(epoch=e, outcome=t,
                             incumbent=t + noise_inc * rng.normal(),
                             proposed=t + noise_pro * rng.normal()))
    return pd.DataFrame(rows)


def test_clearly_worse_proposal_is_flagged_worse():
    res = vsc.evaluate_slice("all", _synthetic(noise_inc=0.0, noise_pro=1.0), "own", 300, 0, 30)
    assert res.verdict == "WORSE"
    code, title, _ = vsc.overall_verdict([res])
    assert code == vsc.EXIT_WORSE


def test_clearly_better_proposal_is_flagged_better():
    res = vsc.evaluate_slice("all", _synthetic(noise_inc=1.0, noise_pro=0.0), "own", 300, 0, 30)
    code, _, _ = vsc.overall_verdict([res])
    assert code == vsc.EXIT_BETTER


def test_equal_noise_is_no_evidence_not_better():
    res = vsc.evaluate_slice("all", _synthetic(noise_inc=1.0, noise_pro=1.0, seed=3),
                             "own", 300, 0, 30)
    code, title, _ = vsc.overall_verdict([res])
    assert code == vsc.EXIT_NO_EVIDENCE
    assert "Do NOT deploy" in vsc.MEANING[code]


def test_small_sample_is_insufficient():
    res = vsc.evaluate_slice("all", _synthetic(n_epochs=2), "own", 100, 0, 30)
    code, _, _ = vsc.overall_verdict([res])
    assert code == vsc.EXIT_INSUFFICIENT


def test_sign_flip_across_slices_blocks_better():
    better = vsc.evaluate_slice("all", _synthetic(noise_inc=1.0, noise_pro=0.0), "own", 300, 0, 30)
    flipped = vsc.SliceResult("other", 100, 10, 0.5, 0.4, -0.1, (0, 1), (0, 1), (-0.3, 0.1),
                              verdict="NO_DIFFERENCE")
    code, _, _ = vsc.overall_verdict([better, flipped])
    assert code == vsc.EXIT_NO_EVIDENCE


def test_neighbor_outcomes_hamming_and_future():
    lb = pd.DataFrame({
        "sequence": ["AAAA", "AAAB", "AABB", "ABBB", "CCCC"],
        "epoch": [1, 2, 0, 3, 1],
        "published_pct": [0.1, 0.2, 0.4, 0.8, 0.5],
    })
    cand = lb.iloc[[0]].copy()
    assert vsc.neighbor_outcomes(lb, cand, 1, False).iloc[0] == pytest.approx(0.2)
    assert vsc.neighbor_outcomes(lb, cand, 2, False).iloc[0] == pytest.approx(0.3)
    # future-only drops AABB (epoch 0 < 1)
    assert vsc.neighbor_outcomes(lb, cand, 2, True).iloc[0] == pytest.approx(0.2)


network = pytest.mark.skipif(os.environ.get("NOVA_NETWORK_TESTS") != "1",
                             reason="needs the public leaderboard API (NOVA_NETWORK_TESTS=1)")


@network
def test_known_case_formula_20261009():
    assert vsc.main(["--case", "formula-20261009", "--bootstrap", "300"]) == vsc.EXIT_WORSE


@network
def test_known_case_parent20x_20261009():
    assert vsc.main(["--case", "parent20x-20261009", "--bootstrap", "200"]) == vsc.EXIT_NO_EVIDENCE
