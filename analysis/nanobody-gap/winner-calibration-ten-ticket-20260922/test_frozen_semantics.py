#!/usr/bin/env python3
"""Regression tests that freeze joint-ranking and live identity semantics."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiment import (
    METRICS,
    dedupe_field_entries,
    identity_audit,
    joint_field_outcomes,
    live_order_fill,
)
from build_preregistration import complete_metrics


HERE = Path(__file__).resolve().parent


def vector(value: float) -> dict[str, float]:
    return {
        metric: value if direction == "max" else -value
        for metric, direction in METRICS.items()
    }


class FrozenSemanticsTest(unittest.TestCase):
    def test_live_api_target_uses_protein_name_not_numeric_id(self) -> None:
        record = {
            "protein_id": 42,
            "protein_name": "P20809",
            **vector(1),
        }
        self.assertEqual(complete_metrics(record), vector(1))
        record["protein_name"] = "OTHER"
        self.assertIsNone(complete_metrics(record))

    def test_competitor_field_dedupes_exact_sequences(self) -> None:
        sequence = "A" * 100
        row = {"sequence": sequence, "owner_label": "rival_a", "metrics": vector(1)}
        other = {"sequence": "C" * 100, "owner_label": "rival_b", "metrics": vector(0)}
        unique, audit = dedupe_field_entries([row, dict(row), other])
        self.assertEqual(len(unique), 2)
        self.assertEqual(audit["duplicate_sequence_count"], 1)

    def test_inconsistent_duplicate_metrics_are_rejected(self) -> None:
        sequence = "A" * 100
        with self.assertRaises(ValueError):
            dedupe_field_entries([
                {"sequence": sequence, "metrics": vector(1)},
                {"sequence": sequence, "metrics": vector(2)},
            ])

    def test_multiple_own_entries_can_jointly_win(self) -> None:
        result = joint_field_outcomes([vector(2), vector(2)], [vector(1), vector(0)])
        self.assertEqual(result["wins"], [True, True])

    def test_own_entries_compete_for_global_top_ten(self) -> None:
        arm = [vector(value) for value in range(10, 0, -1)]
        rivals = [vector(-1), vector(-2)]
        result = joint_field_outcomes(arm, rivals)
        self.assertEqual(sum(result["top10"]), 10)
        self.assertFalse(any(score <= 0 for score in result["arm_rank_sums"]))

    def test_literal_fleets_use_live_fallback_below_095(self) -> None:
        fleets = json.loads((HERE / "fleets.json").read_text())
        a = fleets["A"]["fleet"]
        b = fleets["B"]["fleet"]
        self.assertEqual(len(live_order_fill(a, 10)), 10)
        audit = identity_audit(b)
        self.assertGreater(audit["pairs_at_or_above_preferred_0_94"], 0)
        self.assertEqual(audit["pairs_at_or_above_validator_0_95"], 0)


if __name__ == "__main__":
    unittest.main()
