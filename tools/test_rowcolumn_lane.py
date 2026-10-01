"""Dependency-free checks for the coverage + row/column block lane.

rdkit is unavailable locally, so these exercise the lane's own logic --
block anchoring, deterministic sweep traversal, quota selection and the
fixed-batch merge -- without importing the chemistry stack.
"""
import json
import random
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub the chemistry-dependent parent so these checks run without rdkit. The
# stub lands in sys.modules, so any test module that needs the REAL
# adaptive_strategy must evict it first (see test_residual_base_contract).
stub = types.ModuleType("compound_miner.adaptive_strategy")
class _Parent:
    def __init__(self, history, reaction, database, rng, exact_parents=None, *, feedback=None):
        self.database = database
        self.reaction = reaction
        self.rng = rng
        self.roles = database.reactions[reaction].roles
    def propose(self):
        return ("fallback",)
stub.FeedbackProposalPolicy = _Parent
sys.modules.setdefault("compound_miner", types.ModuleType("compound_miner")).__path__ = [str(ROOT / "compound_miner")]
sys.modules["compound_miner.adaptive_strategy"] = stub

import pandas as pd
from compound_miner.rowcolumn_strategy import (ExplorationSweepPolicy, SweepState,
                                               merge_with_quota, quota_records)


class _Reaction:
    roles = (16, 32, 64)


class _Database:
    reactions = {3: _Reaction()}
    role_ids = {16: list(range(100, 110)), 32: list(range(200, 212)), 64: list(range(300, 320))}
    def random_components(self, reaction, rng):
        return (rng.choice(self.role_ids[16]), rng.choice(self.role_ids[32]), rng.choice(self.role_ids[64]))


def _feedback(rows):
    return pd.DataFrame.from_records(rows, columns=["component_a", "component_b", "component_c", "score", "reaction"])


def _ranker(reference, candidates, minimum_similarity=0.5):
    """Deterministic stand-in: 'similarity' falls off with id distance."""
    scored = [(i, 1.0 - min(1.0, abs(i - reference) / 20.0)) for i in candidates if i != reference]
    return [i for i, v in sorted(scored, key=lambda x: (-x[1], x[0])) if v >= minimum_similarity]


def _policy(state, feedback, ranker=_ranker, **kwargs):
    kwargs.setdefault("depth_per_block", 99)
    return ExplorationSweepPolicy(pd.DataFrame(), 3, _Database(), random.Random(1),
                                  feedback=feedback, state=state,
                                  similarity_ranker=ranker, **kwargs)


class BlockAnchoring(unittest.TestCase):
    def test_blocks_are_AB_pairs_from_best_molecules(self):
        # 101/201 has the single best molecule but a poor mean; it must anchor.
        feedback = _feedback([(101, 201, 301, 0.90, 3), (101, 202, 302, 0.01, 3),
                              (102, 203, 303, 0.50, 3), (102, 204, 304, 0.50, 3)])
        reference = {}
        blocks = ExplorationSweepPolicy._blocks(feedback, 2, None, reference)
        self.assertEqual(blocks[0], (101, 201), "blocks must be (A,B), not (A,C)")
        self.assertEqual(reference[(101, 201)], 301, "block carries its anchor's role C")
        self.assertIn((102, 203), blocks)

    def test_no_feedback_disables_sweep(self):
        state = SweepState()
        policy = _policy(state, _feedback([]))
        self.assertEqual(state.blocks, [])
        self.assertIsNone(policy._sweep())


class SweepTraversal(unittest.TestCase):
    def test_sweep_holds_a_and_b_and_varies_c(self):
        state = SweepState()
        policy = _policy(state, _feedback([(101, 201, 301, 0.9, 3)]))
        seen = set()
        for _ in range(6):
            got = policy._sweep()
            state.sweep_components.add(got)
            seen.add(got)
        self.assertTrue(all(x[0] == 101 and x[1] == 201 for x in seen),
                        "roles A and B must be held fixed")
        self.assertEqual(len(seen), 6, "each draw must supply a distinct role C")

    def test_sweep_walks_role_c_in_descending_similarity(self):
        state = SweepState()
        policy = _policy(state, _feedback([(101, 201, 310, 0.9, 3)]))
        drawn = []
        for _ in range(4):
            got = policy._sweep(); state.sweep_components.add(got); drawn.append(got[2])
        distances = [abs(c - 310) for c in drawn]
        self.assertEqual(distances, sorted(distances),
                         "nearest role C reagents must be tried first")

    def test_dissimilar_role_c_is_excluded_not_just_deprioritised(self):
        state = SweepState()
        policy = _policy(state, _feedback([(101, 201, 301, 0.9, 3)]))
        allowed = set(state.c_order[(101, 201)])
        self.assertTrue(allowed, "some role C must clear the threshold")
        self.assertTrue(all(abs(c - 301) <= 10 for c in allowed),
                        "reagents below minimum_similarity must be dropped")

    def test_cursor_persists_across_rebuilt_policies(self):
        state = SweepState()
        feedback = _feedback([(101, 201, 301, 0.9, 3)])
        first = _policy(state, feedback)
        for _ in range(5):
            state.sweep_components.add(first._sweep())
        cursor = state.cursor[(101, 201)]
        second = _policy(state, feedback)
        self.assertEqual(state.cursor[(101, 201)], cursor,
                         "rebuilding the policy must not restart the sweep")
        self.assertNotIn(second._sweep(), state.sweep_components)

    def test_exhausted_grid_reports_rather_than_repeating(self):
        state = SweepState()
        policy = _policy(state, _feedback([(101, 201, 301, 0.9, 3)]))
        for _ in range(len(state.c_order[(101, 201)])):
            state.sweep_components.add(policy._sweep())
        self.assertIsNone(policy._sweep())
        self.assertEqual(state.exhausted, 1)


class CoverageLane(unittest.TestCase):
    def test_coverage_reaches_reagents_absent_from_feedback(self):
        state = SweepState()
        policy = _policy(state, _feedback([(101, 201, 301, 0.9, 3)]),
                         coverage_fraction=1.0, sweep_fraction=0.0)
        drawn = {policy.propose() for _ in range(200)}
        roles_c = {t[2] for t in drawn}
        self.assertGreater(len(roles_c), 1)
        self.assertTrue(roles_c.issubset(set(_Database.role_ids[64])))
        self.assertEqual(state.coverage_proposed, 200)

    def test_lane_fractions_are_validated(self):
        for bad in (dict(coverage_fraction=0.7, sweep_fraction=0.5),
                    dict(coverage_fraction=-0.1), dict(sweep_fraction=1.5)):
            with self.assertRaises(ValueError):
                _policy(SweepState(), _feedback([]), **bad)


class Quota(unittest.TestCase):
    def _archive(self, rows):
        path = Path(tempfile.mkdtemp()) / "a.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE candidates(name TEXT,smiles TEXT,reaction INT,"
                               "components TEXT,inchikey TEXT,heavy_atoms INT,rotatable_bonds INT,"
                               "proxy_mean REAL,proxy_std REAL,score REAL)")
            connection.executemany("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        return path

    def test_quota_prefers_high_proxy_tagged_rows_and_skips_scored(self):
        path = self._archive([
            ("n1", "C", 3, json.dumps([101, 201, 301]), "k1", 10, 1, 0.9, 0.1, None),
            ("n2", "C", 3, json.dumps([101, 202, 301]), "k2", 10, 1, 0.5, 0.1, None),
            ("n3", "C", 3, json.dumps([999, 999, 999]), "k3", 10, 1, 0.99, 0.1, None),
            ("n4", "C", 3, json.dumps([101, 203, 301]), "k4", 10, 1, 0.95, 0.1, 0.5),
        ])
        wanted = {(101, 201, 301), (101, 202, 301), (101, 203, 301)}
        got = quota_records(path, 3, wanted, 5)
        self.assertEqual([r["name"] for r in got], ["n1", "n2"], "untagged and scored rows excluded")

    def test_quota_respects_excluded_inchikeys(self):
        path = self._archive([("n1", "C", 3, json.dumps([101, 201, 301]), "k1", 10, 1, 0.9, 0.1, None)])
        self.assertEqual(quota_records(path, 3, {(101, 201, 301)}, 5, excluded={"k1"}), [])

    def test_empty_inputs_return_nothing(self):
        path = self._archive([("n1", "C", 3, json.dumps([101, 201, 301]), "k1", 10, 1, 0.9, 0.1, None)])
        self.assertEqual(quota_records(path, 3, set(), 5), [])
        self.assertEqual(quota_records(path, 3, {(101, 201, 301)}, 0), [])


class Merge(unittest.TestCase):
    def _rows(self, names):
        return [dict(name=n) for n in names]

    def test_merge_keeps_batch_size_and_drops_weakest(self):
        candidates = self._rows(["a", "b", "c", "d"])
        merged, coverage = merge_with_quota(candidates, self._rows(["x", "y"]))
        self.assertEqual([r["name"] for r in merged], ["a", "b", "x", "y"])
        self.assertEqual(len(merged), len(candidates))
        self.assertEqual(coverage["rows_substituted"], 2)

    def test_merge_is_a_noop_when_rows_already_selected(self):
        candidates = self._rows(["a", "b"])
        merged, coverage = merge_with_quota(candidates, self._rows(["a"]))
        self.assertEqual([r["name"] for r in merged], ["a", "b"])
        self.assertEqual(coverage["rows_substituted"], 0)
        self.assertEqual(coverage["rows_already_selected"], 1)

    def test_merge_never_exceeds_batch_even_if_quota_is_larger(self):
        candidates = self._rows(["a", "b"])
        merged, _ = merge_with_quota(candidates, self._rows(["x", "y", "z", "w"]))
        self.assertEqual(len(merged), 2)
        self.assertEqual([r["name"] for r in merged], ["x", "y"])


class BlockTurnover(unittest.TestCase):
    def test_exhausted_block_is_retired_and_replaced(self):
        state = SweepState()
        feedback = _feedback([(101, 201, 301, 0.9, 3), (102, 202, 302, 0.8, 3)])
        policy = _policy(state, feedback, block_count=1)
        self.assertEqual(state.blocks, [(101, 201)])
        for _ in range(len(state.c_order[(101, 201)])):
            state.sweep_components.add(policy._sweep())
        self.assertIsNone(policy._sweep())
        self.assertIn((101, 201), state.exhausted_blocks)
        # Next round rebuilds the policy: the dead block must not be chosen again.
        second = _policy(state, feedback, block_count=1)
        self.assertEqual(state.blocks, [(102, 202)])
        self.assertIn((101, 201), state.retired_blocks)
        self.assertIsNotNone(second._sweep())

    def test_sweep_skips_exhausted_blocks_within_a_round(self):
        state = SweepState()
        feedback = _feedback([(101, 201, 301, 0.9, 3), (102, 202, 302, 0.8, 3)])
        policy = _policy(state, feedback, block_count=2)
        state.exhausted_blocks.add((101, 201))
        for _ in range(8):
            got = policy._sweep()
            self.assertIsNotNone(got)
            state.sweep_components.add(got)
            self.assertEqual((got[0], got[1]), (102, 202))

    def test_summary_reports_turnover(self):
        state = SweepState()
        state.exhausted_blocks.add((1, 2)); state.retired_blocks.append((3, 4))
        summary = state.summary()
        self.assertEqual(summary["exhausted_blocks"], [[1, 2]])
        self.assertEqual(summary["retired_blocks"], [[3, 4]])


class QuotaOrdering(unittest.TestCase):
    def _archive(self, rows):
        path = Path(tempfile.mkdtemp()) / "a.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE candidates(name TEXT,smiles TEXT,reaction INT,"
                               "components TEXT,inchikey TEXT,heavy_atoms INT,rotatable_bonds INT,"
                               "proxy_mean REAL,proxy_std REAL,score REAL)")
            connection.executemany("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        return path

    def test_exploration_orders_by_uncertainty_not_mean(self):
        # "lo" has the worse predicted mean but far higher disagreement: an
        # exploration lane must prefer it, an exploitation lane must not.
        path = self._archive([
            ("hi", "C", 3, json.dumps([101, 201, 301]), "k1", 10, 1, 0.90, 0.01, None),
            ("lo", "C", 3, json.dumps([101, 202, 301]), "k2", 10, 1, 0.10, 0.90, None),
        ])
        wanted = {(101, 201, 301), (101, 202, 301)}
        by_mean = quota_records(path, 3, wanted, 1, order_by="proxy_mean")
        by_std = quota_records(path, 3, wanted, 1, order_by="proxy_std")
        self.assertEqual([r["name"] for r in by_mean], ["hi"])
        self.assertEqual([r["name"] for r in by_std], ["lo"])

    def test_invalid_order_is_rejected(self):
        path = self._archive([("n", "C", 3, json.dumps([1, 2, 3]), "k", 10, 1, 0.5, 0.1, None)])
        with self.assertRaises(ValueError):
            quota_records(path, 3, {(1, 2, 3)}, 1, order_by="score; DROP TABLE candidates")


class DepthCap(unittest.TestCase):
    def test_depth_cap_limits_candidates_per_block(self):
        state = SweepState()
        _policy(state, _feedback([(101, 201, 301, 0.9, 3)]), depth_per_block=3)
        self.assertEqual(len(state.c_order[(101, 201)]), 3)

    def test_depth_cap_spreads_sweep_across_blocks(self):
        state = SweepState()
        feedback = _feedback([(101, 201, 301, 0.9, 3), (102, 202, 302, 0.8, 3)])
        policy = _policy(state, feedback, block_count=2, depth_per_block=2)
        drawn = []
        for _ in range(4):
            got = policy._sweep()
            if got is None:
                break
            state.sweep_components.add(got); drawn.append((got[0], got[1]))
        self.assertEqual(len(set(drawn)), 2, "both blocks must be used, not one exhausted")

    def test_invalid_depth_rejected(self):
        with self.assertRaises(ValueError):
            _policy(SweepState(), _feedback([]), depth_per_block=0)


if __name__ == "__main__":
    unittest.main()
