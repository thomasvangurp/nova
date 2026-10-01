"""Unit checks only: fixture labels below are not scientific mining evidence."""
import dataclasses
import math
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest

for repository in (Path(__file__).resolve().parents[1], Path('/root/nim19-test/oss-repo')):
    if (repository/'compound_miner').is_dir():
        sys.path.insert(0, str(repository))
        break

import numpy as np
import pandas as pd

from compound_miner.adaptive_strategy import (
    FeedbackProposalPolicy, OnlineResidualProxy, refresh_unscored_archive,
)
from compound_miner.archive import CandidateArchive
from compound_miner.features import ComponentEncoder
from compound_miner.types import Candidate, ScoredCandidate


def fixture_history():
    return pd.DataFrame([dict(reaction=3, component_a=a, component_b=b,
                              component_c=30, score=score)
                         for a, b, score in [(1, 10, .1), (2, 10, .8),
                                             (1, 20, .8), (2, 20, .1)]])


class FrozenFixture:
    def __init__(self):
        self.encoder = ComponentEncoder().fit(fixture_history())
        self.calls = 0

    def predict(self, candidates, chemistry=None):
        self.calls += 1
        return [dataclasses.replace(c, proxy_mean=.01, proxy_std=.02) for c in candidates]


def fixtures(count=36):
    rows = []
    for i in range(count):
        # Mock identities intentionally make these UNIT fixtures, not candidates.
        c = Candidate(f'fixture-{i}', 'CCO', 3, (i, 10, 30), f'unit-{i}', 3, 0)
        score = .02 + (i % 6)/1000
        rows.append(ScoredCandidate(c, .5, .5-3*score, score, 'unit-nim-only', 1.0))
    return rows


class AdaptiveStrategyTests(unittest.TestCase):
    def proxy(self):
        return OnlineResidualProxy(FrozenFixture(), expected_scorer='unit-nim-only',
                                   trees=8, jobs=1, minimum_labels=4)

    def test_cold_start_matches_frozen_and_no_labels_are_added_by_prediction(self):
        proxy = self.proxy()
        values = [s.candidate for s in fixtures(3)]
        self.assertEqual(proxy.predict(values), proxy.frozen.predict(values))
        self.assertEqual(len(proxy.observations), 0)
        self.assertFalse(proxy.fitted)

    def test_observe_updates_separate_head_without_fitting_frozen(self):
        proxy = self.proxy()
        base_encoder = proxy.frozen.encoder
        result = proxy.observe(fixtures())
        self.assertTrue(result['fitted'])
        self.assertEqual(result['unique_labels'], 36)
        self.assertEqual(result['canonical_labels_added'], 0)
        self.assertIs(proxy.frozen.encoder, base_encoder)
        self.assertGreater(proxy.predict([fixtures(1)[0].candidate])[0].proxy_mean, .01)

    def test_insufficient_labels_keep_frozen_ranking(self):
        proxy = self.proxy()
        proxy.observe(fixtures(3))
        self.assertFalse(proxy.fitted)
        values = [s.candidate for s in fixtures(3)]
        self.assertEqual(proxy.predict(values), proxy.frozen.predict(values))

    def test_wrong_recipe_fails_before_mutating_state(self):
        proxy = self.proxy()
        rows = fixtures(5)
        rows[-1] = dataclasses.replace(rows[-1], scorer='canonical')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            proxy.observe(rows)
        self.assertEqual(dict(proxy.observations), {})

    def test_nonfinite_and_incorrect_objective_fail_closed(self):
        for changes in [{'score':math.nan}, {'score':.9}]:
            proxy = self.proxy()
            with self.assertRaises(ValueError):
                proxy.observe([dataclasses.replace(fixtures(1)[0], **changes)])
            self.assertEqual(dict(proxy.observations), {})

    def test_reaction_and_identity_mismatch_rejected(self):
        proxy = self.proxy()
        row = fixtures(1)[0]
        with self.assertRaises(ValueError):
            proxy.observe([dataclasses.replace(row, candidate=dataclasses.replace(row.candidate, reaction=5))])
        proxy.observe([row])
        with self.assertRaisesRegex(ValueError, 'identity'):
            proxy.observe([dataclasses.replace(row, candidate=dataclasses.replace(row.candidate, components=(2,10,30)))])

    def test_repeats_are_averaged_not_counted_as_new_molecules(self):
        proxy = self.proxy()
        row = fixtures(1)[0]
        second = dataclasses.replace(row, score=.03, affinity_pred_value=.5-3*.03)
        proxy.observe([row, second])
        frame = proxy.feedback_frame()
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]['observations'], 2)
        self.assertAlmostEqual(frame.iloc[0]['score'], .025)

    def database(self):
        return SimpleNamespace(reactions={3:SimpleNamespace(roles=(1,2,4))},
                               role_ids={1:(1,2,999),2:(10,20),4:(30,)},
                               random_components=lambda rxn,rng:(999,20,30))

    def test_reaction3_context_enabled_and_valid_role_exploration_reachable(self):
        p = FeedbackProposalPolicy(fixture_history(), 3, self.database(), random.Random(68),
                                   feedback=fixture_history())
        self.assertIsNotNone(p.contextual)
        self.assertIsNotNone(p.current_context)
        proposed = {p._replacement([1,10,30], 0) for _ in range(500)}
        self.assertIn(999, proposed)
        self.assertTrue(proposed <= {1,2,999})

    def test_replacement_ranking_depends_on_retained_context(self):
        p = FeedbackProposalPolicy(fixture_history(), 3, self.database(), random.Random(68),
                                   feedback=fixture_history())
        self.assertGreater(p.current_context.replacement_value((999,10,30), 0, 2),
                           p.current_context.replacement_value((999,10,30), 0, 1))
        self.assertLess(p.current_context.replacement_value((999,20,30), 0, 2),
                        p.current_context.replacement_value((999,20,30), 0, 1))

    def test_context_does_not_mix_feedback_from_other_reactions(self):
        feedback = fixture_history().copy()
        feedback.loc[0, 'reaction'] = 5
        with self.assertRaisesRegex(ValueError, 'feedback_reaction'):
            FeedbackProposalPolicy(fixture_history(), 3, self.database(), random.Random(68),
                                   feedback=feedback)

    def test_archive_refresh_changes_only_unscored_proxy_fields(self):
        with tempfile.TemporaryDirectory(prefix='adaptive-unit-') as tmp:
            archive = CandidateArchive(Path(tmp)/'candidates.sqlite')
            rows = fixtures(5)
            archive.add_candidates([r.candidate for r in rows])
            archive.add_scores(rows[:1])
            with sqlite3.connect(archive.path) as db:
                before = db.execute('SELECT * FROM candidates WHERE name=?', (rows[0].candidate.name,)).fetchone()
            result = refresh_unscored_archive(archive, 3, self.proxy(), batch_size=2)
            self.assertEqual(result['refreshed_unscored'], 4)
            with sqlite3.connect(archive.path) as db:
                after = db.execute('SELECT * FROM candidates WHERE name=?', (rows[0].candidate.name,)).fetchone()
                count = db.execute('SELECT count(*) FROM candidates WHERE score IS NULL AND proxy_mean=.01').fetchone()[0]
            self.assertEqual(before, after)
            self.assertEqual(count, 4)

    def test_bad_archive_refresh_identity_rolls_back(self):
        with tempfile.TemporaryDirectory(prefix='adaptive-unit-') as tmp:
            archive = CandidateArchive(Path(tmp)/'candidates.sqlite')
            archive.add_candidates([r.candidate for r in fixtures(5)])
            class BadProxy:
                def predict(self, candidates):
                    return list(reversed(candidates))
            with self.assertRaisesRegex(ValueError, 'identity'):
                refresh_unscored_archive(archive,3,BadProxy())
            with sqlite3.connect(archive.path) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM candidates WHERE proxy_mean=0').fetchone()[0],5)


if __name__ == '__main__':
    unittest.main()
