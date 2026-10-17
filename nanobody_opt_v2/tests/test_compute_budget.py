"""Budget / reserve math for the progressive-deepening rewrite.

Pure-Python, no GPU, no DB, no network.  Exercises the prune floor, the
staged schedule the flags imply, and the 4800-candidate screen window
against the 72-minute epoch after the commitment/reserve trim.
"""
from __future__ import annotations

import unittest

from nanobody_opt_v2 import compute_budget as cb


class TestPruneWidth(unittest.TestCase):
    def test_floor_never_below_minimum_keep(self):
        # int(150 * 0.35) == 52; int(52 * 0.35) == 18 -> floored at 20.
        self.assertEqual(cb.prune_width(150, 0.35, 20), 52)
        self.assertEqual(cb.prune_width(52, 0.35, 20), 20)  # 18 truncated, floored
        self.assertEqual(cb.prune_width(20, 0.35, 20), 20)  # already at floor
        self.assertEqual(cb.prune_width(3, 0.35, 20), 3)  # never above width

    def test_expected_sequence(self):
        alive, seq = 150, []
        for _ in range(8):
            alive = cb.prune_width(alive, 0.35, 20)
            seq.append(alive)
        self.assertEqual(seq, [52, 20, 20, 20, 20, 20, 20, 20])


class TestStagedSchedule(unittest.TestCase):
    def test_progressive_cuts(self):
        sched = cb.staged_schedule(150, 8, 0.35, 20, rounds_per_stage=1)
        self.assertEqual([w for w, _ in sched], [150, 52, 20, 20, 20, 20, 20, 20])
        self.assertEqual([r for _, r in sched], [1] * 8)

    def test_single_stage_default(self):
        self.assertEqual(cb.staged_schedule(150, 8, 0.35, 20), [(150, 8)])


class TestTimingModel(unittest.TestCase):
    def test_new_gpu_default(self):
        # 4.8 s/candidate/GPU == ~0.60 s/obs aggregate across an 8-GPU pool.
        self.assertAlmostEqual(cb.TimingModel().gpu_seconds_per_candidate, 4.8)

    def test_4800_candidates_fit_screen_window(self):
        block_seconds = 12.0
        epoch_length = 361  # 72 minutes
        commitment_blocks = 36  # Pi commits at ~325 of 361 (361 - 325)
        replication_reserve_seconds = 600  # ~10 min, priced from 8 rounds
        screen_window = (
            epoch_length * block_seconds
            - commitment_blocks * block_seconds
            - replication_reserve_seconds
        )  # ≈ 4332 - 432 - 600 = 3300s ≈ 55 min
        self.assertAlmostEqual(screen_window, 3300, delta=10)
        fold_seconds = 4800 * 0.60  # 2880s ≈ 48 min
        load_seconds = 7 * 45.0  # seven tranche loads ≈ 315s ≈ 5 min
        self.assertLessEqual(fold_seconds + load_seconds, screen_window)


class TestConfirmationPlan(unittest.TestCase):
    def _plan(self):
        model = cb.TimingModel(
            load_seconds=45.0, gpu_seconds_per_candidate=4.8, cpu_guard_seconds=120.0
        )
        return cb.confirmation_plan(
            epoch_length=361, commitment_blocks=36, rounds=8,
            maximum_candidates=150, gpus=8, model=model,
            time_share=0.25, rounds_per_stage=1, keep_fraction=0.35,
            minimum_keep=20,
        )

    def test_screen_window_about_55_minutes(self):
        plan = self._plan()
        screen_window = (361 - 36 - plan["replication_reserve_blocks"]) * 12.0
        # ≈ 55 min, allow a couple of blocks of slack.
        self.assertGreaterEqual(screen_window, 52 * 60)
        self.assertLessEqual(screen_window, 56 * 60)

    def test_reserve_stays_inside_epoch(self):
        plan = self._plan()
        self.assertGreaterEqual(plan["replication_reserve_blocks"], 0)
        self.assertLessEqual(plan["replication_reserve_blocks"], 361 - 36)
        self.assertGreaterEqual(plan["screen_cutoff_block"], 0)

    def test_finalists_is_a_ceiling(self):
        self.assertLessEqual(self._plan()["confirmation_candidates"], 150)

    def test_minimum_keep_matches_schedule(self):
        model = cb.TimingModel(
            load_seconds=45.0, gpu_seconds_per_candidate=4.8, cpu_guard_seconds=120.0
        )
        sched = cb.staged_schedule(150, 8, 0.35, 20, rounds_per_stage=1)
        manual = sum(r * model.round_seconds(w, 8) for w, r in sched)
        self.assertAlmostEqual(
            model.staged_seconds(
                rounds=8, candidates=150, gpus=8,
                rounds_per_stage=1, keep_fraction=0.35, minimum_keep=20,
            ),
            manual,
        )


if __name__ == "__main__":
    unittest.main()
