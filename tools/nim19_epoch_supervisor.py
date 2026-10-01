#!/usr/bin/env python3
"""Advance the offline test automatically after locked compute gates pass."""
import json
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/root/nim19-test');RUN=BASE/'epoch'
sys.path.insert(0,str(BASE/'client'))
from nim19_epoch_runner import environment,save,run_bounded

def main():
    deadline=time.monotonic()+1500
    save('supervisor.json',dict(stage='waiting_for_pixi_gates',off_chain=True))
    while time.monotonic()<deadline:
        paths=[RUN/'pixi-gates.json',RUN/'pixi-preflight.json',RUN/'nim-throughput-canary.json']
        if all(p.exists() for p in paths):
            gates,preflight,canary=[json.loads(p.read_text()) for p in paths]
            if (gates.get('passed') and preflight.get('stage')=='complete' and preflight.get('failed')==0
                and canary.get('passed') and canary.get('all_eight_gpus_observed_working')):break
        time.sleep(3)
    else:raise RuntimeError('compute_gate_deadline')
    env=environment()
    save('supervisor.json',dict(stage='pixi_calibration',off_chain=True))
    calibration=['/root/search_python.sh',str(BASE/'client/nim19_epoch_calibration.py')]
    if '--assess-existing-calibration' in sys.argv:calibration.append('--assess-existing')
    result=subprocess.run(calibration,
        cwd=BASE/'oss-repo',env=env,capture_output=True,timeout=1200)
    if result.returncode:raise RuntimeError('pixi_calibration_failed')
    save('supervisor.json',dict(stage='six_sequential_rounds',off_chain=True))
    result=run_bounded(['/root/search_python.sh',str(BASE/'client/nim19_epoch_runner.py')],env,BASE/'oss-repo',4320)
    if result.returncode:raise RuntimeError('six_round_test_failed')
    final=json.loads((RUN/'final.json').read_text())
    assert final.get('stage')=='complete' and final.get('completed_rounds')==6
    save('supervisor.json',dict(stage='complete',completed_rounds=6,off_chain=True))

if __name__=='__main__':
    try:main()
    except Exception as e:
        # Never persist child logs or resolved credentials.
        reason=str(e) if str(e) in {'compute_gate_deadline','pixi_calibration_failed','six_round_test_failed','epoch_deadline_exceeded'} else None
        save('supervisor.json',dict(stage='failed',error_class=type(e).__name__,reason=reason,off_chain=True))
        sys.exit(1)
