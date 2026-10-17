"""Replication tournament: the minimum_keep floor and the prune schedule.

Fully mocked -- no GPU, no scorer, no DB, no network.  The scorer invocation
(`_run_screen`), the chain clock (`_chain_allows_work`), GPU contention, and
the re-rank callback are all fakes, so the test exercises only the
successive-elimination pruning logic inside `_run_replication_rounds`.
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest import mock

from nanobody_opt_v2 import compute_budget as cb
from nanobody_opt_v2 import sequential_exact_race as ser


def _fake_run_screen(**kwargs):
    out = kwargs["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "exact_all.json").write_text("{}")


class TestRunReplicationRounds(unittest.TestCase):
    def _run(self, rounds_per_stage, minimum_keep, keep_fraction):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shortlist = [f"s{i:04d}" for i in range(150)]
            prunes = []

            def fake_rerank(alive, keep):
                prunes.append((len(alive), keep))
                return alive[:keep]

            with mock.patch.object(ser, "_run_screen", side_effect=_fake_run_screen), \
                 mock.patch.object(ser, "_chain_allows_work", return_value=True), \
                 mock.patch.object(ser, "_gpu_shared", return_value=False), \
                 mock.patch.object(ser, "_append_timing"), \
                 mock.patch.object(ser._round_parallelism, "plan", return_value=[]):
                completed = ser._run_replication_rounds(
                    shortlist=shortlist,
                    rounds=8,
                    root=root,
                    config=pathlib.Path("config.yaml"),
                    target="P20809",
                    gpus=["0"],
                    scorer=pathlib.Path("scorer.py"),
                    chain_state=None,
                    epoch=1,
                    reserve_blocks=55,
                    predict_batch_size=None,
                    timings=root / "timings.jsonl",
                    rounds_per_stage=rounds_per_stage,
                    keep_fraction=keep_fraction,
                    minimum_keep=minimum_keep,
                    rerank=fake_rerank,
                    finish_model=None,
                    finish_guard_seconds=0.0,
                )
            return completed, prunes

    def test_progressive_schedule_and_floor(self):
        completed, prunes = self._run(rounds_per_stage=1, minimum_keep=20, keep_fraction=0.35)
        # Eight rounds all complete even though the set is pruned between stages.
        self.assertEqual(completed, 8)
        # Pre-prune widths follow 150 -> 52 -> floor(20) ... (truncated int()).
        self.assertEqual([pre for pre, _ in prunes], [150, 52, 20, 20, 20, 20, 20, 20])
        # Every prune respects the 20-survivor floor and mirrors prune_width.
        for pre, keep in prunes:
            self.assertEqual(keep, cb.prune_width(pre, 0.35, 20))
            self.assertGreaterEqual(keep, 20)

    def test_never_prunes_below_floor_at_sharp_width(self):
        # At 52 alive, keep_fraction 0.35 -> int(18.2) == 18; the floor must
        # hold it at 20 (NOT 150 -> 52 -> 18).
        completed, prunes = self._run(rounds_per_stage=1, minimum_keep=20, keep_fraction=0.35)
        at_52 = [keep for pre, keep in prunes if pre == 52]
        self.assertEqual(at_52, [20])

    def test_single_stage_default_prunes_once(self):
        completed, prunes = self._run(rounds_per_stage=0, minimum_keep=20, keep_fraction=0.35)
        self.assertEqual(completed, 8)
        # Flat allocation: one prune after all eight rounds.
        self.assertEqual(prunes, [(150, 52)])


if __name__ == "__main__":
    unittest.main()
