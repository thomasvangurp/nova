#!/usr/bin/env python3
"""Resume the preserved 6x384 archive through round 12, offline only.

This longer leader-competitiveness diagnostic is NOT a 72-minute epoch trial.
No score from the separate 12x192 arm is inserted into this continuation.
"""
import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time

BASE=Path('/root/nim19-test')
SOURCE=BASE/'epoch'
PAIR=BASE/'schedule-comparison'
EXT=BASE/'leader-12x384'
CLIENT=BASE/'extension-client'
ACTIVE_BUDGET=7200
WAIT_LIMIT=1800
PUBLIC_INPUTS=('leaderboard.json','training.csv','novelty.csv','proxy.joblib',
               'pixi-gates.json','pixi-preflight.json','pixi-calibration.json')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix('.partial');t.write_text(json.dumps(value,allow_nan=False));t.replace(path)

def status(stage,**fields):
    write(EXT/'extension-status.json',dict(stage=stage,off_chain=True,
          live_submission_authorized=False,not_a_single_epoch_fit_test=True,**fields))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def runner():return load('nim19_extension_epoch',CLIENT/'nim19_extension_epoch.py')
def comparison():return load('nim19_extension_comparison',BASE/'comparison-client/nim19_schedule_comparison.py')

def prepare():
    if (EXT/'challenge.json').exists():raise RuntimeError('extension_already_prepared')
    final=json.loads((SOURCE/'final.json').read_text())
    if final.get('stage')!='complete' or final.get('completed_rounds')!=6 or any(r['full_count']!=384 for r in final['rounds']):
        raise RuntimeError('baseline_incomplete')
    EXT.mkdir(exist_ok=True)
    conf=json.loads((SOURCE/'challenge.json').read_text())
    conf.update(rounds=12,full_candidates_per_round=384,comparison_fixed_batch=True,
       offline_leader_diagnostic=True,resume_completed_rounds=6,budget_seconds=ACTIVE_BUDGET)
    write(EXT/'challenge.json',conf)
    for name in PUBLIC_INPUTS:
        source=SOURCE/name
        if not source.is_file():raise RuntimeError('frozen_input_missing')
        (EXT/name).symlink_to(source)
    (EXT/'upstream-validator').symlink_to(SOURCE/'upstream-validator',target_is_directory=True)
    # SQLite's read-only backup preserves rows/order/labels without modifying A.
    with closing(sqlite3.connect('file:'+str(SOURCE/'candidates.sqlite')+'?mode=ro',uri=True)) as source:
        with closing(sqlite3.connect(EXT/'candidates.sqlite')) as dest:source.backup(dest)
    for iteration in range(1,7):
        for kind in ('search','nim','selected'):
            filename=f'round-{iteration}-{kind}.json'
            shutil.copyfile(SOURCE/filename,EXT/filename)
    shutil.copyfile(SOURCE/'rounds.json',EXT/'rounds.json')
    write(EXT/'resume-receipt.json',dict(source_completed_rounds=6,reused_full_nim_scores=2304,
       baseline_elapsed_seconds=final['elapsed_seconds'],baseline_canonical_sum20=final['canonical_sum20'],
       baseline_final_sha256=hashlib.sha256((SOURCE/'final.json').read_bytes()).hexdigest(),
       proposal_rng_restored_by_receipt_replay=True,separate_12x192_scores_not_used=True,
       active_budget_seconds=ACTIVE_BUDGET,not_a_single_epoch_fit_test=True))
    return final

def trial():
    r=runner();r.RUN=EXT
    try:r.run()
    except Exception as e:
        allowed={'epoch_deadline_exceeded','epoch_budget_insufficient','acquisition_refill_exhausted',
          'nim_scoring_failed','canonical_scoring_failed','resume_acceptance_missing',
          'resume_rng_receipt_mismatch','resume_score_receipt_mismatch'}
        r.state(stage='failed',error_class=type(e).__name__,reason=str(e) if str(e) in allowed else None)
        return 1
    return 0

def supervise():
    cmp=comparison();cmp.verify_frozen()
    final=prepare();status('waiting_for_comparison',resume_completed_rounds=6)
    protocol=json.loads((EXT/'extension-protocol.json').read_text())
    remaining=ACTIVE_BUDGET-final['elapsed_seconds']
    if remaining<=0:raise RuntimeError('extension_budget_insufficient')
    deadline=min(time.time()+WAIT_LIMIT,protocol['shutdown_deadline_unix']-remaining-180)
    terminal=False
    while time.time()<deadline:
        path=PAIR/'status.json'
        if path.is_file() and json.loads(path.read_text()).get('stage') in ['complete','failed']:
            terminal=True;break
        time.sleep(15)
    if not terminal:raise RuntimeError('comparison_still_running')
    cmp.verify_frozen();status('continuing_rounds_7_to_12',resume_completed_rounds=6)
    r=runner()
    result=r.run_bounded(['/root/search_python.sh',str(CLIENT/'nim19_leader_extension.py'),'trial'],
             r.environment(),r.REPO,math.ceil(remaining)+5)
    if result.returncode:raise RuntimeError('extension_trial_failed')
    complete=json.loads((EXT/'final.json').read_text())
    if complete.get('stage')!='complete' or complete.get('completed_rounds')!=12:
        raise RuntimeError('extension_incomplete')
    cmp.verify_frozen()
    status('complete',completed_rounds=12,canonical_sum20=complete['canonical_sum20'],
       matched_rescored_leader_sum20=complete['matched_rescored_leader_sum20'],
       beats_matched_rescored_leader=complete['beats_matched_rescored_leader'],
       elapsed_seconds=complete['elapsed_seconds'],within_single_epoch_72_minutes=complete['within_single_epoch_72_minutes'])
    return 0

def local_control():
    import nim19_epoch_control as c
    return c

def start():
    c=local_control();host=c.instance();manifest=json.loads(c.MANIFEST.read_text())
    shutdown=manifest['created_unix']+manifest['lease_safety_seconds']
    # The controller never transfers/restarts production, models or NIM runtime.
    if c.remote("from pathlib import Path;Path('/root/nim19-test/extension-client').mkdir(exist_ok=True)"):
        return 1
    root=Path(__file__).resolve().parent
    transfers=[(root/'nim19_epoch_runner.py','nim19_extension_epoch.py'),
               (root/'nim19_leader_extension.py','nim19_leader_extension.py')]
    for source,name in transfers:
        result=subprocess.run(['rsync','-rt','--checksum','--no-owner','--no-group','--no-perms',
             '-e',shlex.join(c.ssh(host)),str(source),'root@'+host['public_ipaddr']+':'+str(CLIENT/name)],
             stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        if result.returncode:raise RuntimeError('extension_source_transfer_failed')
    pins={name:hashlib.sha256(source.read_bytes()).hexdigest() for source,name in transfers}
    code=r'''
import json,sys,hashlib,subprocess
from pathlib import Path
d=json.load(sys.stdin);client=Path('/root/nim19-test/extension-client');root=Path('/root/nim19-test/leader-12x384')
if root.exists():raise RuntimeError('extension_already_exists')
assert all(hashlib.sha256((client/n).read_bytes()).hexdigest()==h for n,h in d['source_sha256'].items())
baseline=json.loads(Path('/root/nim19-test/epoch/final.json').read_text())
remaining=7200-baseline['elapsed_seconds']
assert d['shutdown_deadline_unix']-__import__('time').time()>=1800+remaining+180
root.mkdir();(root/'extension-protocol.json').write_text(json.dumps(d))
env=dict(HOME='/root/nim19-test/oss-home',PATH='/usr/local/bin:/usr/bin:/bin',LANG='C.UTF-8')
p=subprocess.Popen(['python3',str(client/'nim19_leader_extension.py'),'supervise'],env=env,
 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(event='started',summary={'leader_extension_queued':True,'extension_pid':p.pid,
 'reused_scores':2304,'new_rounds':6,'batch_size':384})),flush=True)
'''
    protocol=dict(shutdown_deadline_unix=shutdown,source_sha256=pins,
       requested_total_rounds=12,full_molecules_per_round=384,reused_completed_rounds=6,
       active_budget_seconds=ACTIVE_BUDGET,wait_for_comparison_seconds=WAIT_LIMIT,
       no_simultaneous_gpu_trials=True,not_a_single_epoch_fit_test=True,
       off_chain=True,live_submission_authorized=False)
    return c.remote(code,protocol,timeout=30)

def collect():
    c=local_control()
    import nim19_schedule_comparison as cmp
    cmp.collect()
    host=c.instance();dest=c.DEST/'leader-12x384';dest.mkdir(exist_ok=True)
    result=subprocess.run(['rsync','-rlt','--checksum','--no-owner','--no-group','--no-perms',
       '-e',shlex.join(c.ssh(host)),'--include=/*.json','--include=/candidates.sqlite','--exclude=*',
       'root@'+host['public_ipaddr']+':'+str(EXT)+'/',str(dest)+'/'],
       stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
    if result.returncode:raise RuntimeError('extension_collection_failed')
    print(json.dumps(dict(event='source',summary={'leader_extension_collected':True})),flush=True)
    return 0

def monitor():
    c=local_control();saved=json.loads(c.MANIFEST.read_text())
    deadline=saved['created_unix']+saved['lease_safety_seconds']-120;previous=None
    code=r'''
import json,math
from pathlib import Path
root=Path('/root/nim19-test/leader-12x384');result={}
for name in ['extension-status','progress']:
 p=root/(name+'.json')
 if not p.is_file():continue
 d=json.loads(p.read_text());v={}
 allowed={'waiting_for_comparison','continuing_rounds_7_to_12','resume_rng_replay','warmup','search','full_nim','round_complete','final_canonical','complete','failed'}
 if d.get('stage') in allowed:v['stage']=d['stage']
 for k in ['iteration','completed','expected','completed_rounds','phase_seconds','elapsed_seconds','canonical_sum20','matched_rescored_leader_sum20']:
  if type(d.get(k)) in [int,float] and math.isfinite(d[k]):v[k]=d[k]
 result[name]=v
p=root/'rounds.json'
if p.is_file():
 rows=json.loads(p.read_text()).get('rounds',[])
 if rows:result['last_round']={k:rows[-1][k] for k in ['iteration','full_count','nim_quoted_sum20','entropy','search_seconds','nim_seconds','cumulative_seconds']
  if type(rows[-1].get(k)) in [int,float] and math.isfinite(rows[-1][k])}
p=Path('/root/nim19-test/schedule-comparison/audit-repair-status.json')
if p.is_file():
 d=json.loads(p.read_text())
 if d.get('stage') in {'waiting_for_extension','running_locked_pixi','complete','failed'}:
  result['posthoc-audit-stage']=d['stage']
print(json.dumps(dict(event='source',summary=result)),flush=True)
'''
    while time.time()<deadline:
        try:
            host=c.instance()
            if host.get('cur_state')=='stopped':return 0
            result=subprocess.run([*c.ssh(host),'root@'+host['public_ipaddr'],'python3 -c '+shlex.quote(code)],
                  input=b'{}',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
        except Exception as e:
            print(json.dumps(dict(event='source',summary={'extension_monitor_retry':True,'error_class':type(e).__name__})),flush=True)
            time.sleep(15);continue
        current={}
        for line in result.stdout.splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if event.get('event')=='source':current=event.get('summary',{})
        signature=json.dumps(current,sort_keys=True)
        if signature!=previous:
            print(json.dumps(dict(event='source',summary=current)),flush=True);previous=signature
        if (current.get('extension-status',{}).get('stage') in ['complete','failed']
            and current.get('posthoc-audit-stage') in [None,'complete','failed']):
            try:collect()
            finally:c.stop()
            return 0
        time.sleep(15)
    try:collect()
    finally:c.stop()
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['start','monitor','collect','supervise','trial'])
    operation=parser.parse_args().operation
    try:return globals()[operation]()
    except Exception as e:
        if operation in ['supervise','trial']:
            try:status('failed',error_class=type(e).__name__)
            except Exception:pass
        else:print(json.dumps(dict(event='operation_failed',error_class=type(e).__name__)),flush=True)
        return 1

if __name__=='__main__':sys.exit(main())
