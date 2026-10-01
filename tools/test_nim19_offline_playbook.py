import ast
from pathlib import Path
import sys
import tomllib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import nim19_epoch_control as control
import nim19_epoch_calibration as calibration

class OfflinePlaybookTests(unittest.TestCase):
    def test_lease_watch_retries_network_failure_instead_of_exiting(self):
        with patch.object(control,'MANIFEST',SimpleNamespace(read_text=lambda:'{"created_unix":0,"lease_safety_seconds":1}')), \
             patch.object(control.time,'time',return_value=2), \
             patch.object(control.time,'sleep') as sleep, \
             patch.object(control,'stop',side_effect=[RuntimeError('network_unavailable'),0]) as stop, \
             patch('builtins.print'):
            self.assertEqual(control.watch(),0)
        self.assertEqual(stop.call_count,2)
        sleep.assert_called_once_with(30)

    def test_ssh_is_direct_even_with_hostile_user_configuration(self):
        command=control.ssh({'ports':{'22/tcp':[{'HostPort':'2222'}]}})
        self.assertIn('/dev/null',command)
        self.assertIn('ProxyCommand=none',command)
        self.assertIn('ProxyJump=none',command)
        self.assertIn('ServerAliveInterval=10',command)

    def test_epoch_budget_planner_shrinks_before_missing_six_rounds(self):
        tree=ast.parse((ROOT/'tools/nim19_epoch_runner.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='planned_full_count')
        import math
        namespace={'math':math,'FULL_CANDIDATES_PER_ROUND':384,'FINAL_RESERVE_SECONDS':600}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<planner>','exec'),namespace)
        plan=namespace['planned_full_count']
        self.assertEqual(plan(4320,6),384)
        self.assertLess(plan(3600,6),384)
        self.assertLessEqual(plan(4320,6,seconds_per_molecule=2),232)
        self.assertGreaterEqual(plan(3600,6,include_current_search=False),plan(3600,6))
        with self.assertRaisesRegex(RuntimeError,'epoch_budget_insufficient'):plan(650,1)
        text=(ROOT/'tools/nim19_epoch_supervisor.py').read_text()
        self.assertIn("env,BASE/'oss-repo',4320)",text)

    def test_bounded_partial_acquisition_is_explicit(self):
        acquire,saved,states=self.acquisition_function()
        engine=Mock();engine.run_round.return_value={'proposed_tuples':10.,'constructed':5.}
        candidates,summary=acquire(engine,SimpleNamespace(path='archive'),Mock(return_value=list(range(80))),
                                   set(),1,0,limit=384,max_passes=2,allow_partial=True)
        self.assertEqual(len(candidates),80)
        self.assertTrue(summary['partial_acquisition'])
        self.assertEqual(engine.run_round.call_count,2)

    def test_direct_full_strategy_uses_nim_without_reduced_selection(self):
        text=(ROOT/'tools/nim19_epoch_runner.py').read_text();tree=ast.parse(text)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='nim_controls')
        namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<controls>','exec'),namespace)
        reduced=namespace['nim_controls'](True);full=namespace['nim_controls']()
        self.assertEqual([reduced[k] for k in ['recycling_steps','sampling_steps','sampling_steps_affinity','diffusion_samples_affinity']],[1,20,10,1])
        self.assertEqual([full[k] for k in ['recycling_steps','sampling_steps','sampling_steps_affinity','diffusion_samples_affinity']],[3,100,100,3])
        self.assertNotIn('skip_affinity_structure',reduced)
        self.assertIn('FULL_CANDIDATES_PER_ROUND=384',text)
        self.assertIn('limit=planned',text)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        run_text=ast.unparse(run)
        self.assertNotIn('reduced=True',run_text)
        self.assertNotIn('[:768]',run_text)
        self.assertIn('scored, nim_seconds = nim_score(proposed, iteration)',run_text)
        self.assertNotIn('reduced=canonical(',text)
        self.assertIn('archive.add_scores(scored)',text)
        self.assertNotIn('archive.add_scores(reduced)',text)
        self.assertIn("str(RUN/'scratch'/output.stem)",text)
        self.assertIn('reduced_pass_used=False',text)

    def test_reduced_nim_uses_only_supported_controls_and_is_not_promoted(self):
        tree=ast.parse((ROOT/'tools/nim19_reduced_screen_audit.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controls')
        namespace={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<controls>','exec'),namespace)
        controls=namespace['controls']()
        self.assertEqual(controls['recycling_steps'],1)
        self.assertEqual(controls['sampling_steps'],20)
        self.assertEqual(controls['sampling_steps_affinity'],10)
        self.assertEqual(controls['diffusion_samples_affinity'],1)
        self.assertNotIn('recycling_steps_affinity',controls)
        self.assertNotIn('skip_affinity_structure',controls)
        text=(ROOT/'tools/nim19_reduced_screen_audit.py').read_text()
        self.assertIn('policy_accepted=False',text)
        self.assertIn('timings_not_isolated=True',text)

    def acquisition_function(self):
        tree=ast.parse((ROOT/'tools/nim19_epoch_runner.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='acquire')
        saved={};states=[]
        namespace={'time':SimpleNamespace(monotonic=lambda:1.0),
                   'remaining_epoch_seconds':lambda:float('inf'),'FINAL_RESERVE_SECONDS':600,
                   'save':lambda name,payload:saved.update({name:payload}),
                   'state':lambda **fields:states.append(fields)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<acquire>','exec'),namespace)
        return namespace['acquire'],saved,states

    def test_underfilled_acquisition_refills_without_relaxing_gate(self):
        acquire,saved,states=self.acquisition_function()
        engine=Mock();engine.run_round.return_value={'proposed_tuples':500000.,'constructed':250000.}
        load=Mock(side_effect=[list(range(441)),list(range(800)),list(range(1024))])
        seen={'already-scored'}
        candidates,summary=acquire(engine,SimpleNamespace(path='archive'),load,seen,1,0)
        self.assertEqual(len(candidates),1024)
        self.assertEqual(summary['acquisition_passes'],3)
        self.assertEqual(summary['constructed'],750000.)
        self.assertEqual(summary['novelty_threshold'],0.5)
        self.assertTrue(all(c.kwargs['excluded'] is seen for c in load.call_args_list))
        self.assertTrue(all(c.kwargs['exact_shortlist']==0 for c in engine.run_round.call_args_list))
        self.assertEqual(saved['round-1-search.json']['required'],1024)

    def test_refill_is_bounded_and_full_batch_skips_extra_work(self):
        acquire,saved,states=self.acquisition_function()
        engine=Mock();engine.run_round.return_value={'proposed_tuples':10.,'constructed':5.}
        with self.assertRaisesRegex(RuntimeError,'acquisition_refill_exhausted'):
            acquire(engine,SimpleNamespace(path='archive'),Mock(return_value=[]),set(),2,0,max_passes=3)
        self.assertEqual(engine.run_round.call_count,3)
        engine.reset_mock()
        acquire(engine,SimpleNamespace(path='archive'),Mock(return_value=list(range(1024))),set(),1,0)
        self.assertEqual(engine.run_round.call_count,1)

    def test_legacy_baseline_without_model_digest(self):
        original={'source_sha256':'same_main'}
        current={'source_sha256':'same_main','model_source_sha256':calibration.PINNED_BOLTZ2_SHA256}
        self.assertTrue(calibration.source_matches(current,original))
        current['model_source_sha256']='wrong_model'
        self.assertFalse(calibration.source_matches(current,original))
    def test_runtime_boundary_and_matching_compiler(self):
        cfg=tomllib.loads((ROOT/'deploy/nim19_offline/pixi.toml').read_text())
        self.assertEqual(cfg['pypi-dependencies']['torch']['version'],'==2.7.1')
        self.assertEqual(cfg['pypi-dependencies']['torch']['index'],'https://download.pytorch.org/whl/cu128')
        self.assertEqual(cfg['pypi-dependencies']['triton'],'==3.3.1')
        self.assertEqual(cfg['dependencies']['cuda-nvcc'],'12.8.*')
        self.assertNotIn('chain310',cfg['environments'])
        self.assertNotIn('nim',cfg['pypi-dependencies'])

    def test_locked_install_not_pip_overlay(self):
        text=(ROOT/'deploy/nim19_offline/compute.yml').read_text()
        self.assertIn('--locked --environment bz2',text)
        self.assertIn('flock /root/nim19-test/pixi-install.lock',text)
        self.assertNotIn('pip install',text)
        self.assertIn('51335750',text)
        self.assertIn('not live_submission_authorized',text)

    def test_remote_programs_compile_and_boot_is_gated(self):
        programs=[]
        with patch.object(control,'remote',side_effect=lambda code,**kw:programs.append(code) or 0):
            control.progress();control.boot();control.calibration()
        for program in programs:ast.parse(program)
        self.assertIn('unpack-complete',programs[1])
        self.assertIn('pixi-gates.json',programs[2])
        self.assertNotIn('oss-venv',programs[2])

    def test_fast_paths_are_executed_not_only_imported(self):
        text=(ROOT/'tools/nim19_pixi_gates.py').read_text()
        self.assertIn('ext.tanimoto_max(',text)
        self.assertIn('use_kernels=True',text)
        self.assertIn('assert not Path(\'/root/nova.env\').exists()',text)
        self.assertIn('assert not Path(\'/root/.bittensor\').exists()',text)

    def test_strict_parity_failure_is_not_relabelled(self):
        text=(ROOT/'tools/nim19_epoch_calibration.py').read_text()
        self.assertIn("result['spearman']>=0.99",text)
        self.assertIn('eligible_for_offline_matched_rescore',text)
        self.assertIn('passed=passed',text)
        runner=(ROOT/'tools/nim19_epoch_runner.py').read_text()
        self.assertIn('matched_rescored_leader_sum20',runner)
        self.assertIn('completed_rounds=round_count',runner)
        self.assertNotIn("str(BASE/'oss-venv/bin/python')",runner)

if __name__=='__main__':unittest.main()
