import ast
from contextlib import closing
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import nim19_leader_extension as extension

class LeaderExtensionTests(unittest.TestCase):
    def test_read_only_sqlite_clone_and_first_six_receipts_preserved(self):
        with tempfile.TemporaryDirectory(prefix='nim19-extension-test-') as tmp:
            base=Path(tmp);source=base/'epoch';dest=base/'extended'
            source.mkdir();(source/'upstream-validator').mkdir()
            final=dict(stage='complete',completed_rounds=6,elapsed_seconds=2826,
               canonical_sum20=1.43,rounds=[{'full_count':384} for _ in range(6)])
            (source/'final.json').write_text(json.dumps(final))
            (source/'challenge.json').write_text(json.dumps(dict(rounds=6)))
            for n in extension.PUBLIC_INPUTS:(source/n).write_text('public scientific input')
            (source/'rounds.json').write_text(json.dumps({'rounds':final['rounds']}))
            for i in range(1,7):
                for kind in ('search','nim','selected'):(source/f'round-{i}-{kind}.json').write_text('{}')
            with closing(sqlite3.connect(source/'candidates.sqlite')) as db:
                db.execute('CREATE TABLE marker(value TEXT)');db.execute('INSERT INTO marker VALUES (?)',('preserved score',));db.commit()
            before=(source/'candidates.sqlite').read_bytes()
            with patch.multiple(extension,SOURCE=source,EXT=dest):extension.prepare()
            self.assertEqual((source/'candidates.sqlite').read_bytes(),before)
            with closing(sqlite3.connect(dest/'candidates.sqlite')) as db:self.assertEqual(db.execute('SELECT value FROM marker').fetchone()[0],'preserved score')
            cfg=json.loads((dest/'challenge.json').read_text())
            self.assertEqual((cfg['rounds'],cfg['full_candidates_per_round'],cfg['resume_completed_rounds']),(12,384,6))
            self.assertTrue(cfg['offline_leader_diagnostic'])
            self.assertTrue(cfg['comparison_fixed_batch'])
            self.assertFalse((dest/'final.json').exists())
            self.assertTrue((dest/'round-6-nim.json').is_file())

    def replay_function(self):
        tree=ast.parse((ROOT/'tools/nim19_epoch_runner.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay_search_rng')
        saved={};states=[]
        namespace={'time':SimpleNamespace(monotonic=lambda:1),'json':json,'math':math,'LABEL':'label',
           'save':lambda n,p:saved.update({n:p}),'state':lambda **v:states.append(v)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<replay>','exec'),namespace)
        return namespace['replay_search_rng'],saved,states

    def test_rng_replay_preserves_state_without_construction_or_scoring(self):
        replay,saved,states=self.replay_function()
        class Policy:
            def __init__(self,history,reaction,database,rng,exact_parents):self.rng=rng
            def propose(self):return (self.rng.getrandbits(32),)
        scored=[SimpleNamespace(candidate=SimpleNamespace(name=str(i)),score=i/1000) for i in range(384)]
        archive=SimpleNamespace(top=lambda *a,**k:scored)
        engine=SimpleNamespace(history=None,database=None,rng=random.Random(68))
        expected=random.Random(68)
        for _ in range(4):expected.getrandbits(32)
        with tempfile.TemporaryDirectory(prefix='nim19-replay-test-') as tmp:
            p=Path(tmp)
            (p/'round-1-search.json').write_text(json.dumps({'passes':[{'proposed_tuples':4}]}))
            (p/'round-1-nim.json').write_text(json.dumps({'scores':[{'name':str(i),'score':i/1000} for i in range(384)]}))
            with patch.dict(sys.modules,{'compound_miner.search':SimpleNamespace(ProposalPolicy=Policy)}):
                replay(engine,archive,p,1,proposals=2)
        self.assertEqual(engine.rng.getstate(),expected.getstate())
        self.assertTrue(saved['search-rng-state.json']['restored_by_proposal_only_replay'])
        self.assertEqual(states[0]['stage'],'resume_rng_replay')

    def test_unmatched_rng_receipt_fails_closed(self):
        replay,_,_=self.replay_function()
        class Policy:
            def __init__(self,*a,**k):pass
            def propose(self):return (0,)
        with tempfile.TemporaryDirectory(prefix='nim19-replay-failure-') as tmp:
            p=Path(tmp);(p/'round-1-search.json').write_text(json.dumps({'passes':[{'proposed_tuples':4}]}))
            with patch.dict(sys.modules,{'compound_miner.search':SimpleNamespace(ProposalPolicy=Policy)}):
                with self.assertRaisesRegex(RuntimeError,'resume_rng_receipt_mismatch'):
                    replay(SimpleNamespace(history=None,database=None,rng=None),SimpleNamespace(top=lambda *a,**k:[]),p,1,proposals=2)

    def test_diagnostic_not_silently_reported_as_single_epoch_fit(self):
        text=(ROOT/'tools/nim19_epoch_runner.py').read_text();ast.parse(text)
        self.assertIn('7200 if diagnostic else 4320',text)
        self.assertIn('within_single_epoch_72_minutes=',text)
        self.assertIn('started-=previous',text)
        self.assertIn('range(resumed_rounds+1,round_count+1)',text)
        driver=(ROOT/'tools/nim19_leader_extension.py').read_text();ast.parse(driver)
        self.assertIn("'comparison_still_running'",driver)
        self.assertIn('separate_12x192_scores_not_used=True',driver)
        self.assertNotIn('killpg',driver)

if __name__=='__main__':unittest.main()
