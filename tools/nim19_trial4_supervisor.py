#!/usr/bin/env python3
"""Bounded supervisor for the coverage + row/column block trial.

Targets only the new run directory and the new runner. It never touches the
baseline epoch paths, the NIM server, production, or any other trial, and it
relies on the runner's own private process group for deadline cleanup.
"""
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=Path('/root/nim19-test');RUN=BASE/'trial4-20260919'
sys.path.insert(0,str(HERE))
from nim19_trial4_runner import environment,save,run_bounded,RUN as RUNNER_RUN

def main():
    if RUNNER_RUN!=RUN:raise RuntimeError('supervisor_target_mismatch')
    conf=json.loads((RUN/'challenge.json').read_text())
    budget=conf['budget_seconds']
    if type(budget) is not int or not 600<=budget<=4320:raise RuntimeError('supervisor_budget_contract')
    if conf.get('rounds')!=6 or conf.get('reaction')!=3:raise RuntimeError('supervisor_budget_contract')
    wall=budget+300
    started=time.monotonic()
    save('supervisor.json',dict(stage='running',budget_seconds=budget,wall_seconds=wall,
      off_chain=True,live_submission_authorized=False))
    result=run_bounded(['/root/search_python.sh',str(HERE/'nim19_trial4_runner.py')],
                       environment(),BASE/'oss-repo',wall)
    if result.returncode:raise RuntimeError('strategy_trial_failed')
    final=json.loads((RUN/'final.json').read_text())
    if final.get('stage')!='complete' or final.get('completed_rounds')!=6:raise RuntimeError('strategy_trial_failed')
    save('supervisor.json',dict(stage='complete',completed_rounds=final['completed_rounds'],
      elapsed_seconds=time.monotonic()-started,off_chain=True,live_submission_authorized=False))

if __name__=='__main__':
    try:main()
    except Exception as e:
        # Never persist child logs or resolved credentials.
        reason=str(e) if str(e) in {'supervisor_target_mismatch','supervisor_budget_contract',
          'strategy_trial_failed','epoch_deadline_exceeded','process_group_guard'} else None
        save('supervisor.json',dict(stage='failed',error_class=type(e).__name__,reason=reason,
          off_chain=True,live_submission_authorized=False))
        sys.exit(1)
