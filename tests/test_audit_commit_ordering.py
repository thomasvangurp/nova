"""Stdlib-only tests: no authentication, networking, NumPy, or live services."""
import unittest

from tools.audit_commit_ordering import (
    METRICS, average_ranks, concordance, dense_metric_ranks, joined_rows,
    replay_field, spearman,
)


class CommitOrderingTests(unittest.TestCase):
    def test_average_tie_ranks(self):
        self.assertEqual(average_ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])

    def test_spearman_orientation(self):
        self.assertAlmostEqual(spearman([.1, .2, .3], [-3, -2, -1]), 1)
        self.assertIsNone(spearman([1, 1], [2, 3]))

    def test_pairwise_ties_neutral(self):
        rows = [{"p": 3, "actual_score": 1}, {"p": 2, "actual_score": 2},
                {"p": 1, "actual_score": 2}]
        self.assertAlmostEqual(concordance(rows, "p"), 2.5 / 3)
        self.assertAlmostEqual(concordance(rows, "p", [3, 2, 1]), 0)

    def test_dense_ranks_unique_values_and_direction(self):
        field = [{name: value for name in METRICS} for value in [1, 1, 2, 3]]
        result = dense_metric_ranks(field[2], field)
        self.assertTrue(all(value == 2 for value in result.values()))
        result = dense_metric_ranks(field[0], field)
        self.assertEqual(result["design_iiptm"], 3)
        self.assertEqual(result["liability_score"], 1)

    def test_parity_rejects_disagreement(self):
        row = {"metrics": {name: 1 for name in METRICS},
               "categories": {"confidence_rank": 5, "physical_interaction_rank": 3,
                              "developability_rank": 2}, "actual_score": 10}
        replay_field([row])
        row["actual_score"] = 11
        with self.assertRaises(ValueError):
            replay_field([row])

    def fixture(self):
        prediction = {"event": "selection_rank_sum_below_fixed_threshold",
            "selection_metric_policy": "historical-drivers", "consensus_ranking_policy": "off",
            "probability": .1, "score_median": 90, "threshold": 60,
            "selection_metric_weights": {name: 1 for name in METRICS}}
        submission = {"uid": 1, "hotkey": "public_identity_a", "sequence_sha256": "digest_a",
            "commit_block": 42, "publication_stage": "provisional", "prediction_at_commit": prediction,
            "selection": {"win_probability": .1, "score_p90": 100}, "reference_epochs": [100]}
        history = {"epoch": 101, "submissions": [submission]}
        actual = {"uid": 1, "hotkey": "public_identity_a", "sequence_sha256": "digest_a",
            "actual_score": 60, "actual_rank": 1, "metric_ranks": {name: 6 for name in METRICS},
            "metrics": {}, "categories": {}}
        return history, {"field": [actual]}

    def test_uid_is_not_identity(self):
        history, board = self.fixture()
        board["field"][0]["hotkey"] = "public_identity_b"
        self.assertFalse(joined_rows(history, board)[0]["scored"])

    def test_exact_sequence_required(self):
        history, board = self.fixture()
        board["field"][0]["sequence_sha256"] = "digest_b"
        self.assertFalse(joined_rows(history, board)[0]["scored"])

    def test_changed_uid_does_not_override_exact_identity(self):
        history, board = self.fixture()
        board["field"][0]["uid"] = 99
        row = joined_rows(history, board)[0]
        self.assertTrue(row["scored"])
        self.assertFalse(row["uid_matches"])

    def test_frozen_weights_define_event_not_raw_score(self):
        history, board = self.fixture()
        history["submissions"][0]["prediction_at_commit"]["selection_metric_weights"] = {name: .5 for name in METRICS}
        board["field"][0]["actual_rank"] = 2
        row = joined_rows(history, board)[0]
        self.assertTrue(row["predicted_event_occurred"])
        self.assertEqual(row["observed_selection_score"], 30)
        self.assertFalse(row["winner"])

    def test_event_boundary_is_strict_and_not_win(self):
        history, board = self.fixture()
        row = joined_rows(history, board)[0]
        self.assertTrue(row["winner"])
        self.assertFalse(row["predicted_event_occurred"])

    def test_future_reference_rejected(self):
        history, board = self.fixture()
        history["submissions"][0]["reference_epochs"] = [101]
        with self.assertRaises(ValueError):
            joined_rows(history, board)


if __name__ == "__main__":
    unittest.main()
