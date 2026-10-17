"""Argparse default round-trip for the rewritten CLIs."""
from __future__ import annotations

import unittest

from nanobody_opt_v2 import run_live_producer as rlp
from nanobody_opt_v2 import sequential_exact_race as ser
from nanobody_opt_v2 import select_elite_parents as sep


class TestRunLiveProducerFlags(unittest.TestCase):
    def test_new_defaults(self):
        a = rlp._parser().parse_args([])
        self.assertEqual(a.candidate_limit, 4800)
        self.assertEqual(a.finalists, 150)
        self.assertEqual(a.replication_rounds, 8)
        self.assertEqual(a.replication_rounds_per_stage, 1)
        self.assertAlmostEqual(a.replication_keep_fraction, 0.35)
        self.assertEqual(a.replication_minimum_keep, 20)
        # Parent scaling unchanged.
        self.assertEqual(a.parent_limit, 64)
        self.assertEqual(a.neighbor_parent_limit, 32)
        self.assertEqual(a.basin_parent_limit, 32)
        self.assertEqual(a.proposal_limit, 384)
        # Commitment/handoff reserve: Pi commits at ~325 of 361.
        self.assertEqual(a.reserve_blocks, 36)
        self.assertAlmostEqual(a.budget_gpu_seconds, 4.8)
        self.assertFalse(a.dry_run)

    def test_dry_run_round_trips(self):
        self.assertTrue(rlp._parser().parse_args(["--dry-run"]).dry_run)

    def test_overrides_round_trip(self):
        a = rlp._parser().parse_args(
            ["--candidate-limit", "9999", "--finalists", "7", "--replication-rounds", "3"]
        )
        self.assertEqual(a.candidate_limit, 9999)
        self.assertEqual(a.finalists, 7)
        self.assertEqual(a.replication_rounds, 3)


class TestRaceFlags(unittest.TestCase):
    REQ = [
        "--layer2", "x.json", "--archive", "a.csv", "--config", "c.yaml",
        "--target", "P20809", "--epoch", "1", "--output-dir", "/tmp/o",
    ]

    def test_new_defaults(self):
        b = ser._parser().parse_args(list(self.REQ))
        self.assertEqual(b.candidate_limit, 4800)
        self.assertEqual(b.finalists, 150)
        self.assertEqual(b.replication_rounds, 8)
        self.assertEqual(b.replication_rounds_per_stage, 1)
        self.assertAlmostEqual(b.replication_keep_fraction, 0.35)
        self.assertEqual(b.replication_minimum_keep, 20)
        self.assertAlmostEqual(b.screen_gpu_seconds_per_candidate, 4.8)
        self.assertEqual(b.reserve_blocks, 36)
        self.assertFalse(b.dry_run)

    def test_dry_run_round_trips(self):
        b = ser._parser().parse_args(self.REQ + ["--dry-run"])
        self.assertTrue(b.dry_run)


class TestParentSelectionFlags(unittest.TestCase):
    REQ = ["--target", "P20809", "--before-epoch", "100", "--output", "/tmp/x.json"]

    def test_parent_scaling_unchanged(self):
        c = sep._parser().parse_args(list(self.REQ))
        self.assertEqual(c.limit, 64)
        self.assertEqual(c.neighbor_limit, 32)
        self.assertEqual(c.basin_parent_limit, 32)


if __name__ == "__main__":
    unittest.main()
