import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import nim19_schedule_comparison as comparison

def final(rounds,batch,score=2.0,elapsed=4000):
    return dict(stage='complete',completed_rounds=rounds,
       rounds=[dict(iteration=i,full_count=batch) for i in range(1,rounds+1)],
       elapsed_seconds=elapsed,within_epoch_budget=elapsed<=4320,
       epoch=25176,reaction=3,full_recipe={'sampling_steps':100},
       pixi_manifest_sha256='manifest',pixi_lock_sha256='lock',
       canonical_sum20=score,matched_rescored_leader_sum20=1.9,
       final_records=[dict(name=str(i)) for i in range(20)],
       upstream_validation={'portfolios':[{'valid_molecules':20,'atompair_entropy':0.26}]})

class ScheduleComparisonTests(unittest.TestCase):
    def test_equal_deadline_not_equal_nominal_score_count_is_primary(self):
        a,b=final(6,384),final(12,192,2.1)
        r=comparison.summarize(a,b)
        self.assertEqual(r['equal_elapsed_budget_seconds'],4320)
        self.assertTrue(r['both_ready_by_equal_deadline'])
        self.assertTrue(r['nominal_equal_2304_score_comparison'])
        self.assertEqual(r['deadline_portfolio_score_winner'],'12x192')
        self.assertAlmostEqual(r['independently_validated_score_delta_12_minus_6'],0.1)

    def test_forged_fit_flag_cannot_admit_overrun(self):
        a,b=final(6,384),final(12,192,2.1,4321)
        b['within_epoch_budget']=True
        r=comparison.summarize(a,b)
        self.assertFalse(r['both_ready_by_equal_deadline'])
        self.assertIsNone(r['deadline_portfolio_score_winner'])

    def test_partial_and_adaptive_runs_never_claim_nominal_2304(self):
        a,b=final(6,384),final(12,192)
        a['rounds'][-1]['full_count']=192
        r=comparison.summarize(a,b)
        self.assertTrue(r['both_ready_by_equal_deadline'])
        self.assertFalse(r['nominal_equal_2304_score_comparison'])
        b['completed_rounds']=11
        self.assertFalse(comparison.summarize(a,b)['both_ready_by_equal_deadline'])

    def test_invalid_entropy_and_runtime_mismatch_are_not_comparable(self):
        a,b=final(6,384),final(12,192)
        b['upstream_validation']['portfolios'][0]['atompair_entropy']=0.249
        self.assertFalse(comparison.summarize(a,b)['both_ready_by_equal_deadline'])
        b=final(12,192);b['pixi_lock_sha256']='other'
        r=comparison.summarize(a,b)
        self.assertFalse(r['matching_runtime_challenge_recipe'])
        self.assertIsNone(r['deadline_portfolio_score_winner'])

    def test_missing_arm_is_explicitly_inconclusive(self):
        r=comparison.summarize(final(6,384),None)
        self.assertFalse(r['both_ready_by_equal_deadline'])
        self.assertIsNone(r['independently_validated_score_delta_12_minus_6'])

    def test_second_arm_is_fresh_and_only_public_inputs_are_shared(self):
        with tempfile.TemporaryDirectory(prefix='nim19-comparison-test-') as tmp:
            base=Path(tmp);baseline=base/'epoch';pair=base/'pair';second=pair/'12x192'
            baseline.mkdir();(baseline/'upstream-validator').mkdir();(base/'client').mkdir()
            (base/'client/P40261.a3m').write_text('public alignment')
            (baseline/'challenge.json').write_text(json.dumps(dict(rounds=6,reaction=3,epoch=25176,maximum_training_epoch=25174)))
            for name in comparison.PUBLIC_INPUTS:(baseline/name).write_text('public scientific input')
            (baseline/'candidates.sqlite').write_text('baseline scores must not be copied')
            for name in comparison.PINNED_SOURCES:
                p=base/'oss-repo'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('public source')
            with patch.multiple(comparison,BASE=base,BASELINE=baseline,PAIR=pair,SECOND=second):
                comparison.prepare_second();comparison.verify_frozen()
                cfg=json.loads((second/'challenge.json').read_text())
                self.assertEqual((cfg['rounds'],cfg['full_candidates_per_round']),(12,192))
                self.assertTrue(cfg['comparison_fixed_batch'])
                self.assertFalse((second/'candidates.sqlite').exists())
                self.assertTrue((second/'training.csv').is_symlink())
                (baseline/'novelty.csv').write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'scientific_input_changed'):comparison.verify_frozen()

    def test_generic_runner_and_unique_scratch_parents(self):
        text=(ROOT/'tools/nim19_epoch_runner.py').read_text();ast.parse(text)
        self.assertIn('range(resumed_rounds+1,round_count+1)',text)
        self.assertIn('rounds_left=round_count+1-iteration',text)
        self.assertIn('allow_partial=not fixed_batch',text)
        self.assertIn("(RUN/'scratch'/output.stem).mkdir(parents=True,exist_ok=True)",text)
        self.assertIn('completed_rounds=round_count',text)

    def test_existing_baseline_is_never_patched_or_restarted(self):
        text=(ROOT/'tools/nim19_schedule_comparison.py').read_text();ast.parse(text)
        self.assertIn("'nim19_comparison_epoch.py'",text)
        self.assertNotIn("BASELINE/'candidates.sqlite').unlink",text)
        self.assertIn("raise RuntimeError('baseline_still_running')",text)
        self.assertIn('fixed_portfolios_no_audit_reranking=True',text)
        self.assertIn('counted_in_trial_72_minute_budget=False',text)

if __name__=='__main__':unittest.main()
