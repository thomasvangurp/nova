"""Dependency-free runner contract tests; no GPU, credentials or labels."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class AdaptiveWiringTests(unittest.TestCase):
    def acquire(self):
        tree = ast.parse((ROOT/'tools/nim19_epoch_runner.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='acquire')
        namespace = dict(time=SimpleNamespace(monotonic=lambda: 1),
                         remaining_epoch_seconds=lambda: 10000, FINAL_RESERVE_SECONDS=600,
                         state=lambda **kw: None, save=lambda *args: None)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<acquire>', 'exec'), namespace)
        return namespace['acquire']

    def engine(self, enabled):
        return SimpleNamespace(proxy=SimpleNamespace(refresh_acquisition_archive=enabled),
                               run_round=Mock(return_value=dict(constructed=10, proposed_tuples=20)))

    def test_opt_in_refresh_precedes_acquisition_and_is_recorded(self):
        events=[]
        refresh=Mock(side_effect=lambda *a: events.append('refresh') or dict(refreshed_unscored=10, refresh_seconds=.2))
        load=Mock(side_effect=lambda *a,**kw: events.append('load') or [1,2])
        module=SimpleNamespace(refresh_unscored_archive=refresh)
        with patch.dict(sys.modules, {'compound_miner.adaptive_strategy':module}):
            values, metrics=self.acquire()(self.engine(True), SimpleNamespace(path='unit-archive'), load,
                                            set(), 1, 0, limit=2)
        self.assertEqual(events, ['refresh','load'])
        self.assertEqual(values, [1,2])
        self.assertEqual(metrics['passes'][0]['refreshed_unscored'], 10)

    def test_nonboolean_refresh_flag_cannot_enable_experiment(self):
        for value in [False, None, Mock(), 1, 'true']:
            refresh=Mock(side_effect=AssertionError('unexpected refresh'))
            with patch.dict(sys.modules, {'compound_miner.adaptive_strategy':SimpleNamespace(refresh_unscored_archive=refresh)}):
                self.acquire()(self.engine(value), SimpleNamespace(path='unit-archive'),
                               Mock(return_value=[1,2]), set(), 1, 0, limit=2)
            refresh.assert_not_called()

    def test_strategy_flags_default_off_and_adaptive_resume_fails_closed(self):
        source=(ROOT/'tools/nim19_epoch_runner.py').read_text()
        self.assertIn("conf.get('adaptive_proxy',False)", source)
        self.assertIn("conf.get('contextual_rxn3',False)", source)
        self.assertIn("raise RuntimeError('adaptive_resume_not_supported')", source)
        self.assertLess(source.index('scored,nim_seconds=nim_score'), source.index('feedback.observe(scored)'))
        self.assertIn("save(f'round-{iteration}-feedback.json',feedback_update)", source)

    def test_optional_factory_does_not_replace_baseline_policy(self):
        for root in [ROOT, Path('/Users/tvgdeena/nova-compound-fleet')]:
            tree=ast.parse((root/'compound_miner/search.py').read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SearchEngine')
            init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
            defaults=dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):], init.args.defaults))
            self.assertIsNone(ast.literal_eval(defaults['proposal_policy_factory']))
            source=(root/'compound_miner/search.py').read_text()
            self.assertIn('(self.proposal_policy_factory or ProposalPolicy)', source)


if __name__ == '__main__':
    unittest.main()
