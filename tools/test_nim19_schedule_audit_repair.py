import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import nim19_schedule_audit_repair as repair
import nim19_epoch_control as control

class AuditRepairTests(unittest.TestCase):
    def test_heavy_audit_and_curves_execute_inside_locked_pixi(self):
        text=(ROOT/'tools/nim19_schedule_audit_repair.py').read_text();tree=ast.parse(text)
        supervisor=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='supervise')
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        self.assertIn("'/root/search_python.sh'",ast.unparse(supervisor))
        self.assertNotIn('matched_audit(',ast.unparse(supervisor))
        self.assertIn('m.matched_audit(first, second)',ast.unparse(worker))
        self.assertIn('m.convergence(directory, final)',ast.unparse(worker))
        self.assertIn('primary_trial_scores_and_timing_unchanged=True',text)
        self.assertIn("'comparison-before-validation-repair.json'",text)

    def test_monitor_does_not_stop_host_before_queued_audit_is_terminal(self):
        text=(ROOT/'tools/nim19_leader_extension.py').read_text();ast.parse(text)
        self.assertIn("current.get('posthoc-audit-stage') in [None,'complete','failed']",text)
        self.assertIn('audit-repair-status.json',text)

if __name__=='__main__':unittest.main()
