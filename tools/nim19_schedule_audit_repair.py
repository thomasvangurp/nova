#!/usr/bin/env python3
"""Run the extra matched audit and retrospective curves inside locked Pixi.

Waits for the longer diagnostic to finish; never overlaps its GPU inference.
Primary arm scores, counts and timing are unchanged by this post-hoc repair.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

BASE=Path('/root/nim19-test')
PAIR=BASE/'schedule-comparison'
EXT=BASE/'leader-12x384'
CLIENT=BASE/'extension-client'
AUDIT_BUDGET=1200

def write(path,value):
    t=path.with_suffix('.partial');t.write_text(json.dumps(value,allow_nan=False));t.replace(path)

def status(stage,**fields):
    write(PAIR/'audit-repair-status.json',dict(stage=stage,off_chain=True,
          live_submission_authorized=False,posthoc_not_epoch_ready=True,**fields))

def comparison():
    spec=importlib.util.spec_from_file_location('nim19_audit_repair_comparison',BASE/'comparison-client/nim19_schedule_comparison.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def worker():
    # This operation is launched only through /root/search_python.sh.
    m=comparison();m.verify_frozen()
    first=m.read_final(BASE/'epoch');second=m.read_final(PAIR/'12x192')
    if first is None or second is None:raise RuntimeError('completed_arms_missing')
    report=json.loads((PAIR/'comparison.json').read_text())
    write(PAIR/'comparison-before-validation-repair.json',report)
    previous=report.get('matched_audit_error_class')
    audit=m.matched_audit(first,second)
    report.update(matched_fixed_final_audit=audit,matched_audit_complete=True,
       matched_audit_error_class=None,previous_matched_audit_error_class=previous,
       matched_audit_winner='12x192' if audit['delta_12_minus_6']>0 else '6x384' if audit['delta_12_minus_6']<0 else 'tie',
       validation_repair_runtime='locked Pixi bz2 via search_python.sh',
       primary_trial_scores_and_timing_unchanged=True)
    write(PAIR/'comparison.json',report)
    curves={}
    for label,directory,final in [('6x384',BASE/'epoch',first),('12x192',PAIR/'12x192',second)]:
        curves[label]=m.convergence(directory,final)
    report['previous_retrospective_convergence']=report.get('retrospective_convergence',{})
    report['retrospective_convergence']=curves
    checkpoints=[]
    for cutoff in (900,1800,2700,3600,4320):
        row=dict(equal_search_elapsed_seconds=cutoff,retrospective=True,
                 submission_ready_by_cutoff_not_established=True)
        for label in ('6x384','12x192'):
            ready=[x for x in curves[label] if x['search_elapsed_seconds']<=cutoff]
            row[label]=ready[-1] if ready else None
        checkpoints.append(row)
    report['retrospective_equal_search_time_checkpoints']=checkpoints
    report['validation_repair_complete']=True;m.verify_frozen()
    write(PAIR/'comparison.json',report)
    status('complete',matched_audit_complete=True,retrospective_curves_complete=True)
    return 0

def supervise():
    protocol=json.loads((PAIR/'audit-repair-protocol.json').read_text())
    status('waiting_for_extension')
    deadline=protocol['shutdown_deadline_unix']-AUDIT_BUDGET-180
    terminal=False
    while time.time()<deadline:
        p=EXT/'extension-status.json'
        if p.is_file() and json.loads(p.read_text()).get('stage') in ['complete','failed']:
            terminal=True;break
        time.sleep(15)
    if not terminal:raise RuntimeError('extension_still_running')
    m=comparison();m.verify_frozen();r=m.runner()
    status('running_locked_pixi')
    result=r.run_bounded(['/root/search_python.sh',str(CLIENT/'nim19_schedule_audit_repair.py'),'worker'],
                         r.environment(),r.REPO,AUDIT_BUDGET)
    if result.returncode:raise RuntimeError('audit_repair_failed')
    return 0

def start():
    import nim19_epoch_control as c
    host=c.instance();saved=json.loads(c.MANIFEST.read_text())
    source=Path(__file__).resolve();target=CLIENT/source.name
    result=subprocess.run(['rsync','-rt','--checksum','--no-owner','--no-group','--no-perms',
       '-e',shlex.join(c.ssh(host)),str(source),'root@'+host['public_ipaddr']+':'+str(target)],
       stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
    if result.returncode:raise RuntimeError('audit_repair_transfer_failed')
    code=r'''
import json,sys,hashlib,subprocess
from pathlib import Path
d=json.load(sys.stdin);p=Path('/root/nim19-test/extension-client/nim19_schedule_audit_repair.py');root=Path('/root/nim19-test/schedule-comparison')
assert hashlib.sha256(p.read_bytes()).hexdigest()==d['source_sha256']
if (root/'audit-repair-protocol.json').exists():raise RuntimeError('audit_repair_already_queued')
(root/'audit-repair-protocol.json').write_text(json.dumps(d))
(root/'audit-repair-status.json').write_text(json.dumps({'stage':'waiting_for_extension','off_chain':True}))
env=dict(HOME='/root/nim19-test/oss-home',PATH='/usr/local/bin:/usr/bin:/bin',LANG='C.UTF-8')
q=subprocess.Popen(['python3',str(p),'supervise'],env=env,stdin=subprocess.DEVNULL,
 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(event='started',summary={'posthoc_validation_repair_queued':True,'pid':q.pid,'locked_pixi':True})),flush=True)
'''
    protocol=dict(shutdown_deadline_unix=saved['created_unix']+saved['lease_safety_seconds'],
       source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),audit_budget_seconds=AUDIT_BUDGET,
       preserve_primary_comparison=True,wait_for_12x384=True,off_chain=True,live_submission_authorized=False)
    return c.remote(code,protocol,timeout=30)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['start','supervise','worker'])
    op=parser.parse_args().operation
    try:return globals()[op]()
    except Exception as e:
        if op in ['supervise','worker']:
            allowed={'upstream_validator_rejected_portfolio','upstream_source_hash_mismatch',
              'upstream_quoted_score_mismatch','epoch_deadline_exceeded','canonical_scoring_failed',
              'canonical_count','completed_arms_missing','extension_still_running','audit_repair_failed'}
            try:status('failed',error_class=type(e).__name__,reason=str(e) if str(e) in allowed else None)
            except Exception:pass
        else:print(json.dumps(dict(event='operation_failed',error_class=type(e).__name__)),flush=True)
        return 1

if __name__=='__main__':sys.exit(main())
