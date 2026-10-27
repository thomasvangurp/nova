"""Offline experiment guards: no labels leak into ranking or failed-run labels."""
import hashlib
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('proxy_experiment', ROOT/'scripts/pr100_proxy_evaluate.py')
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in ('scipy','sklearn')),
                     'optional offline proxy experiment dependencies are not installed')
class ProxyEvaluationTests(unittest.TestCase):
    def test_execution_or_missing_outcomes_are_censored(self):
        for reason in ['not_run','aggrescan_unavailable_or_failed','timeout','backend_error','']:
            self.assertIsNone(proxy.outcome(reason))
        self.assertEqual(proxy.outcome('passed'),1)
        self.assertEqual(proxy.outcome('psh_threshold'),0)
        self.assertEqual(proxy.outcome('developability_red_flags'),0)

    def test_hash_mismatch_is_never_joined(self):
        rows=[{'sequence_hash':'0'*64,'reason':'passed','buckets':['cell'], 'psh':80}]
        with self.assertRaises(ValueError):
            proxy.join_records(rows,{'0'*64:'ACDEFG'},cells_key='buckets',source='test')

    def test_missing_sequence_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            proxy.join_records([{'sequence_hash':'0'*64}],{},cells_key='buckets',source='test')

    def test_join_preserves_censoring_and_exact_identity(self):
        seq='ACDEFG';h=hashlib.sha256(seq.encode()).hexdigest()
        rows=[{'sequence_hash':h,'reason':'aggrescan_unavailable_or_failed','buckets':['cell'],'psh':80}]
        kept,censored=proxy.join_records(rows,{h:seq},cells_key='buckets',source='test')
        self.assertEqual(kept,[])
        self.assertEqual(censored,{'aggrescan_unavailable_or_failed':1})

    def test_cell_budgets_cannot_spend_all_calls_on_easy_cell(self):
        rows=[{'hash':str(i),'cell':'easy' if i<4 else 'hard','y':0} for i in range(8)]
        keep=proxy.cell_budget_keep(rows,np.arange(8),.25)
        self.assertEqual(int(keep[:4].sum()),1)
        self.assertEqual(int(keep[4:].sum()),1)

    def test_ranking_does_not_read_outcome_labels(self):
        rows=[{'hash':str(i),'cell':'cell','y':i%2} for i in range(8)]
        a=proxy.cell_budget_keep(rows,np.zeros(8),.25)
        for r in rows:r['y']=1-r['y']
        b=proxy.cell_budget_keep(rows,np.zeros(8),.25)
        np.testing.assert_array_equal(a,b)

    def test_abstention_keeps_all_members_of_unsupported_cell(self):
        rows=[{'hash':str(i),'cell':'unknown'} for i in range(8)]
        self.assertTrue(proxy.cell_budget_keep(rows,np.zeros(8),.05,abstain_cells=['unknown']).all())

    def test_features_contain_no_measured_outcomes(self):
        row={'sequence':'ACDEFGHIKLMNPQRSTVWY','psh':80,'y':1}
        a=proxy.features([row]);row.update(psh=120,y=0)
        np.testing.assert_array_equal(a,proxy.features([row]))
        self.assertEqual(a.shape,(1,421))

    def test_related_sequences_share_group(self):
        rows=[{'sequence':'ACDEFGHIKLMNPQRSTVWY'}, {'sequence':'ACDEFGHIKLMNPQRSTVWA'},
              {'sequence':'YYYYYYYYYYYYYYYYYYY'}]
        groups=proxy.family_groups(rows)
        self.assertEqual(groups[0],groups[1])
        self.assertNotEqual(groups[0],groups[2])

    def test_five_successes_do_not_establish_ninetyfive_percent_recall(self):
        self.assertLess(proxy.recall_lower_bound(5,5),.60)

    def test_cross_model_score_scales_do_not_produce_pooled_auc(self):
        rows=[{'hash':str(i),'cell':f'fold{i//2}','y':i%2} for i in range(4)]
        report=proxy.evaluate(rows,np.array([.1,.2,.8,.9]),comparable_score_scale=False)
        self.assertIsNone(report['average_precision'])
        self.assertIsNone(report['roc_auc'])
        self.assertEqual(report['budget_table'][0]['retained_passers'],2)

    def test_oof_tracks_disjoint_fold_membership(self):
        from unittest.mock import patch
        rows=[{'hash':str(i),'cell':'cell','y':i%2,'psh':80} for i in range(8)]
        groups=np.array([0,0,1,1,2,2,3,3])
        def fake_scores(name,train_x,train_rows,test_x):
            # First feature encodes each test row's family ID.
            self.assertFalse(set(train_x[:,0]) & set(test_x[:,0]))
            return np.zeros(len(test_x)),{}
        with patch.object(proxy,'fitted_scores',side_effect=fake_scores):
            scores,evidence=proxy.oof('unused',rows,groups[:,None],groups)
        self.assertEqual(len(scores),8)
        self.assertEqual(len(evidence['fold_ids']),8)
        for i in range(0,8,2):
            self.assertEqual(evidence['fold_ids'][i],evidence['fold_ids'][i+1])


if __name__=='__main__':unittest.main()
