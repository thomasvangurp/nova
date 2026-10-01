#!/usr/bin/env python3
"""Sequential, isolated 6x384 versus 12x192 offline comparison.

The existing six-round process is never patched or restarted. The second arm
has a fresh archive. Only the fixed final portfolios enter the extra matched
audit; audit results cannot retroactively create an epoch-ready submission.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

BASE=Path('/root/nim19-test')
BASELINE=BASE/'epoch'
PAIR=BASE/'schedule-comparison'
SECOND=PAIR/'12x192'
CLIENT=BASE/'comparison-client'
LIMIT=4320
AUDIT_LIMIT=1200
STAGES={'waiting_for_6x384','running_12x192','matched_final_audit','complete','failed'}
PUBLIC_INPUTS=('leaderboard.json','training.csv','novelty.csv','proxy.joblib',
               'pixi-gates.json','pixi-preflight.json','pixi-calibration.json')
PINNED_SOURCES=('compound_miner/search.py','compound_miner/proxy.py',
                'compound_miner/archive.py','compound_miner/selection.py',
                'tools/score_boltz_batch.py','external_tools/boltz/src/boltz/main.py',
                'external_tools/boltz/src/boltz/model/models/boltz2.py')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(value,allow_nan=False));temporary.replace(path)

def status(stage,**fields):
    write(PAIR/'status.json',dict(stage=stage,off_chain=True,
          live_submission_authorized=False,**fields))

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def runner():
    spec=importlib.util.spec_from_file_location('nim19_comparison_epoch',CLIENT/'nim19_comparison_epoch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def arm_summary(final,rounds,batch):
    """Never trust a boolean fit flag or silently call a partial arm nominal."""
    if not isinstance(final,dict):return dict(completed=False,ready_by_72_minutes=False,nominal_schedule_completed=False)
    rows=final.get('rounds',[])
    elapsed=final.get('elapsed_seconds')
    proof=final.get('upstream_validation',{}).get('portfolios',[])
    records=final.get('final_records',[])
    completed=(final.get('stage')=='complete' and final.get('completed_rounds')==rounds
               and len(rows)==rounds and [r.get('iteration') for r in rows]==list(range(1,rounds+1)))
    valid=(len(proof)==1 and proof[0].get('valid_molecules')==20
           and proof[0].get('atompair_entropy',0)>=0.25 and len(records)==20
           and len({r.get('name') for r in records})==20)
    finite=(type(elapsed) in (int,float) and math.isfinite(elapsed)
            and type(final.get('canonical_sum20')) in (int,float) and math.isfinite(final['canonical_sum20']))
    ready=(completed and valid and finite and 0<=elapsed<=LIMIT
           and final.get('within_epoch_budget') is True)
    actual=sum(r.get('full_count',0) for r in rows)
    return dict(completed=completed,ready_by_72_minutes=ready,
        nominal_schedule_completed=completed and all(r.get('full_count')==batch for r in rows),
        requested_rounds=rounds,requested_batch=batch,actual_full_scores=actual,
        elapsed_seconds=elapsed if finite else None,
        canonical_sum20=final.get('canonical_sum20') if finite else None,
        chemistry_novelty_entropy_valid=valid,
        matched_rescored_leader_sum20=final.get('matched_rescored_leader_sum20'))

def summarize(first,second):
    a,b=arm_summary(first,6,384),arm_summary(second,12,192)
    both=a['ready_by_72_minutes'] and b['ready_by_72_minutes']
    matched=False
    if isinstance(first,dict) and isinstance(second,dict):
        keys=('epoch','reaction','full_recipe','pixi_manifest_sha256','pixi_lock_sha256')
        matched=all(k in first and k in second and first[k]==second[k] for k in keys)
    scores_comparable=both and matched
    delta=b['canonical_sum20']-a['canonical_sum20'] if scores_comparable else None
    winner=None if delta is None else ('12x192' if delta>0 else '6x384' if delta<0 else 'tie')
    return dict(equal_elapsed_budget_seconds=LIMIT,arms={'6x384':a,'12x192':b},
        matching_runtime_challenge_recipe=matched,both_ready_by_equal_deadline=both,
        nominal_equal_2304_score_comparison=a['nominal_schedule_completed'] and b['nominal_schedule_completed'],
        independently_validated_score_delta_12_minus_6=delta,deadline_portfolio_score_winner=winner,
        stochastic_noise_not_eliminated=True,single_epoch_not_general_optimality_proof=True,
        off_chain=True,live_submission_authorized=False,live_win_claim=False)

def read_final(directory):
    path=directory/'final.json'
    return json.loads(path.read_text()) if path.is_file() else None

def prepare_second():
    if SECOND.exists():raise RuntimeError('second_arm_already_exists')
    SECOND.mkdir(parents=True)
    conf=json.loads((BASELINE/'challenge.json').read_text())
    if conf.get('rounds')!=6 or conf.get('reaction')!=3:raise RuntimeError('baseline_contract')
    conf.update(rounds=12,full_candidates_per_round=192,comparison_fixed_batch=True,budget_seconds=LIMIT)
    write(SECOND/'challenge.json',conf)
    for name in PUBLIC_INPUTS:
        source=BASELINE/name
        if not source.is_file():raise RuntimeError('frozen_input_missing')
        (SECOND/name).symlink_to(source)
    source=BASELINE/'upstream-validator'
    if not source.is_dir():raise RuntimeError('frozen_input_missing')
    (SECOND/'upstream-validator').symlink_to(source,target_is_directory=True)
    pins={name:digest(BASE/'oss-repo'/name) for name in PINNED_SOURCES}
    inputs={name:digest(BASELINE/name) for name in (*PUBLIC_INPUTS,'challenge.json')}
    write(PAIR/'frozen-inputs.json',dict(source_sha256=pins,input_sha256=inputs,
          msa_sha256=digest(BASE/'client/P40261.a3m'),
          epoch=conf['epoch'],maximum_training_epoch=conf['maximum_training_epoch'],
          fresh_archive=True,no_baseline_scores_reused=True))

def verify_frozen():
    pins=json.loads((PAIR/'frozen-inputs.json').read_text())
    if any(digest(BASE/'oss-repo'/name)!=h for name,h in pins['source_sha256'].items()):
        raise RuntimeError('scientific_source_changed')
    if any(digest(BASELINE/name)!=h for name,h in pins['input_sha256'].items()):
        raise RuntimeError('scientific_input_changed')
    if digest(BASE/'client/P40261.a3m')!=pins['msa_sha256']:raise RuntimeError('scientific_input_changed')

def trial():
    r=runner();r.RUN=SECOND
    try:r.run()
    except Exception as e:
        allowed={'epoch_deadline_exceeded','epoch_budget_insufficient','acquisition_refill_exhausted',
                 'nim_scoring_failed','canonical_scoring_failed','canonical_count'}
        r.state(stage='failed',error_class=type(e).__name__,reason=str(e) if str(e) in allowed else None)
        return 1
    return 0

def matched_audit(first,second):
    """Score precommitted final sets together; do not select better sets here."""
    r=runner();r.RUN=PAIR;r.EPOCH_DEADLINE=time.monotonic()+AUDIT_LIMIT
    from compound_miner.database import ReactionDatabase
    from check_compound_with_upstream_validator import UpstreamCompoundValidator
    board=json.loads((BASELINE/'leaderboard.json').read_text())
    database=ReactionDatabase()
    validator=UpstreamCompoundValidator(BASELINE/'upstream-validator',database.path,
                       BASELINE/'novelty.csv',target='P40261',reaction=3)
    portfolios=[dict(uid=0,records=first['final_records']),dict(uid=1,records=second['final_records'])]
    leader=next(row for row in board['rows'] if row['uid']==board['winner_uid'])
    leaders=[]
    for record in leader['molecules']:
        candidate=database.candidate(3,tuple(int(x) for x in record['name'].split(':')[2:]))
        if candidate is None:raise RuntimeError('leader_identity')
        leaders.append(dict(candidate.as_dict(),score=record.get('final_score',0)))
    portfolios.append(dict(uid=2,records=leaders))
    union={record['name']:record for p in portfolios for record in p['records']}
    # One stable union gives every shared molecule exactly the same sampled score.
    records=[union[name] for name in sorted(union)]
    start=time.monotonic();scored=r.canonical(records,PAIR/'matched-fixed-finals.json')
    by_name={record['name']:record for record in scored['scores']}
    fixed=[dict(uid=p['uid'],records=[by_name[x['name']] for x in p['records']]) for p in portfolios]
    proof=validator.validate(fixed)
    scores={row['uid']:row['quoted_score'] for row in proof['portfolios']}
    result=dict(score_6x384=scores[0],score_12x192=scores[1],score_public_leader=scores[2],
        delta_12_minus_6=scores[1]-scores[0],unique_molecules=len(records),seed=68,
        elapsed_seconds=time.monotonic()-start,upstream_validation=proof,
        fixed_portfolios_no_audit_reranking=True,shared_molecules_same_prediction=True,
        counted_in_trial_72_minute_budget=False,retrospective_audit_not_submission_ready=True,
        single_repeat_stochastic_uncertainty_remains=True)
    write(PAIR/'matched-audit.json',result)
    return result

def convergence(directory,final):
    """Independent score curves are retrospective, not ready-by-cutoff claims."""
    if final is None:return []
    score_path=directory/'final-canonical-scores.json'
    if not score_path.is_file():return []
    r=runner()
    from compound_miner.database import ReactionDatabase
    from check_compound_with_upstream_validator import UpstreamCompoundValidator
    database=ReactionDatabase()
    validator=UpstreamCompoundValidator(BASELINE/'upstream-validator',database.path,
                       BASELINE/'novelty.csv',target='P40261',reaction=3)
    scores=json.loads(score_path.read_text())['scores'];by_name={x['name']:x for x in scores}
    rows=[]
    for metric in final['rounds']:
        selection=json.loads((directory/('round-'+str(metric['iteration'])+'-selected.json')).read_text())['records']
        proof=validator.validate([dict(uid=0,records=[by_name[x['name']] for x in selection])])
        rows.append(dict(iteration=metric['iteration'],search_elapsed_seconds=metric['cumulative_seconds'],
            independently_rescored_fixed_sum20=proof['portfolios'][0]['quoted_score'],
            retrospective=True,submission_ready_at_checkpoint=False))
    return rows

def supervise():
    protocol=json.loads((PAIR/'protocol.json').read_text())
    status('waiting_for_6x384',equal_elapsed_budget_seconds=LIMIT)
    prepare_second()
    # mkdir only: the current process expects this unique scratch parent to exist.
    (BASELINE/'scratch/final-canonical-scores').mkdir(parents=True,exist_ok=True)
    deadline=min(protocol['shutdown_deadline_unix']-LIMIT-AUDIT_LIMIT-180,time.time()+LIMIT)
    first=None;baseline_failed=False
    while time.time()<deadline:
        first=read_final(BASELINE)
        if first is not None:break
        stopped=False
        for name in ('progress.json','supervisor.json'):
            path=BASELINE/name
            if path.is_file() and json.loads(path.read_text()).get('stage')=='failed':stopped=True
        if stopped:baseline_failed=True;break
        time.sleep(15)
    if first is None and not baseline_failed:raise RuntimeError('baseline_still_running')
    verify_frozen()
    status('running_12x192',baseline_complete=first is not None,equal_elapsed_budget_seconds=LIMIT)
    r=runner()
    process_error=None;exit_code=None
    try:
        result=r.run_bounded(['/root/search_python.sh',str(CLIENT/'nim19_schedule_comparison.py'),'trial'],
                            r.environment(),r.REPO,LIMIT)
        exit_code=result.returncode
    except RuntimeError as e:
        if str(e)!='epoch_deadline_exceeded':raise
        process_error='epoch_deadline_exceeded'
    second=read_final(SECOND)
    report=summarize(first,second)
    report['second_process_exit_code']=exit_code
    report['second_process_failure']=process_error
    write(PAIR/'deadline-comparison.json',report)
    if report['both_ready_by_equal_deadline'] and report['matching_runtime_challenge_recipe']:
        verify_frozen();status('matched_final_audit',equal_elapsed_budget_seconds=LIMIT)
        try:
            audit=matched_audit(first,second);report['matched_fixed_final_audit']=audit
            report['matched_audit_winner']='12x192' if audit['delta_12_minus_6']>0 else '6x384' if audit['delta_12_minus_6']<0 else 'tie'
        except Exception as e:
            report['matched_audit_error_class']=type(e).__name__
            report['matched_audit_complete']=False
    report['retrospective_convergence']={}
    for label,directory,final in [('6x384',BASELINE,first),('12x192',SECOND,second)]:
        try:report['retrospective_convergence'][label]=convergence(directory,final)
        except Exception as e:report['retrospective_convergence'][label]=dict(error_class=type(e).__name__)
    cutoffs=[]
    for cutoff in (900,1800,2700,3600,LIMIT):
        row=dict(equal_search_elapsed_seconds=cutoff,retrospective=True,
                 submission_ready_by_cutoff_not_established=True)
        for label in ('6x384','12x192'):
            curve=report['retrospective_convergence'].get(label,[])
            eligible=[x for x in curve if x['search_elapsed_seconds']<=cutoff] if isinstance(curve,list) else []
            row[label]=eligible[-1] if eligible else None
        cutoffs.append(row)
    report['retrospective_equal_search_time_checkpoints']=cutoffs
    write(PAIR/'comparison.json',report)
    status('complete',both_ready_by_equal_deadline=report['both_ready_by_equal_deadline'],
           deadline_portfolio_score_winner=report['deadline_portfolio_score_winner'],
           nominal_equal_2304_score_comparison=report['nominal_equal_2304_score_comparison'])
    return 0

def local_control():
    import nim19_epoch_control as control
    return control

def start():
    c=local_control();host=c.instance();manifest=json.loads(c.MANIFEST.read_text())
    shutdown=manifest['created_unix']+manifest['lease_safety_seconds']
    if shutdown-time.time()<2*LIMIT+AUDIT_LIMIT+240:raise RuntimeError('comparison_lease_too_short')
    if c.remote("from pathlib import Path;Path('/root/nim19-test/comparison-client').mkdir(exist_ok=True)"):
        return 1
    root=Path(__file__).resolve().parent
    transfers=[(root/'nim19_epoch_runner.py','nim19_comparison_epoch.py'),
               (root/'nim19_schedule_comparison.py','nim19_schedule_comparison.py')]
    for source,name in transfers:
        result=subprocess.run(['rsync','-rt','--checksum','--no-owner','--no-group','--no-perms',
                  '-e',shlex.join(c.ssh(host)),str(source),
                  'root@'+host['public_ipaddr']+':'+str(CLIENT/name)],
                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        if result.returncode:raise RuntimeError('comparison_source_transfer_failed')
    expected={name:digest(source) for source,name in transfers}
    code=r'''
import json,sys,hashlib,subprocess
from pathlib import Path
d=json.load(sys.stdin);client=Path('/root/nim19-test/comparison-client');pair=Path('/root/nim19-test/schedule-comparison')
if pair.exists():raise RuntimeError('comparison_already_exists')
assert all(hashlib.sha256((client/n).read_bytes()).hexdigest()==h for n,h in d['source_sha256'].items())
pair.mkdir();(pair/'protocol.json').write_text(json.dumps(d))
env=dict(HOME='/root/nim19-test/oss-home',PATH='/usr/local/bin:/usr/bin:/bin',LANG='C.UTF-8')
p=subprocess.Popen(['python3',str(client/'nim19_schedule_comparison.py'),'supervise'],env=env,
 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(event='started',summary={'schedule_comparison_queued':True,'comparison_pid':p.pid})),flush=True)
'''
    protocol=dict(schema_version=1,shutdown_deadline_unix=shutdown,source_sha256=expected,
       arms={'6x384':{'rounds':6,'maximum_batch':384,'existing_process_unchanged':True},
             '12x192':{'rounds':12,'fixed_batch':192,'fresh_archive':True}},
       each_trial_budget_seconds=LIMIT,sequential_same_eight_gpu_host=True,
       primary_comparison='independently_validated_final_portfolio_ready_by_4320_seconds',
       audit_budget_seconds=AUDIT_LIMIT,posthoc_audit_cannot_improve_primary_portfolio=True,
       off_chain=True,live_submission_authorized=False)
    return c.remote(code,protocol,timeout=30)

def collect():
    c=local_control();c.collect();host=c.instance();dest=c.DEST/'schedule-comparison';dest.mkdir(exist_ok=True)
    result=subprocess.run(['rsync','-rlt','--checksum','--no-owner','--no-group','--no-perms',
       '-e',shlex.join(c.ssh(host)),'--include=/*.json','--include=/12x192/',
       '--include=/12x192/*.json','--include=/12x192/candidates.sqlite','--exclude=*',
       'root@'+host['public_ipaddr']+':'+str(PAIR)+'/',str(dest)+'/'],
       stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
    if result.returncode:raise RuntimeError('comparison_collection_failed')
    print(json.dumps(dict(event='source',summary={'comparison_artifacts_collected':True})),flush=True)
    return 0

def monitor():
    c=local_control();saved=json.loads(c.MANIFEST.read_text())
    deadline=saved['created_unix']+saved['lease_safety_seconds']-120;previous=None
    code=r'''
import json,math
from pathlib import Path
result={}
for label,root in [('comparison',Path('/root/nim19-test/schedule-comparison')),
                   ('6x384',Path('/root/nim19-test/epoch')),
                   ('12x192',Path('/root/nim19-test/schedule-comparison/12x192'))]:
 p=root/('status.json' if label=='comparison' else 'progress.json')
 if not p.is_file():continue
 d=json.loads(p.read_text());v={}
 for k in ['iteration','completed','expected','completed_rounds','phase_seconds','elapsed_seconds']:
  if type(d.get(k)) in [int,float] and math.isfinite(d[k]) and d[k]>=0:v[k]=d[k]
 allowed={'waiting_for_6x384','running_12x192','matched_final_audit','complete','failed','warmup','search','full_nim','round_complete','final_canonical'}
 if d.get('stage') in allowed:v['stage']=d['stage']
 rp=root/'rounds.json'
 rounds=json.loads(rp.read_text()).get('rounds',[]) if rp.is_file() else d.get('rounds',[])
 if isinstance(rounds,list) and rounds:
  v['last_round']={k:rounds[-1][k] for k in ['iteration','full_count','nim_quoted_sum20','entropy','search_seconds','nim_seconds','cumulative_seconds']
   if type(rounds[-1].get(k)) in [int,float] and math.isfinite(rounds[-1][k])}
 result[label]=v
print(json.dumps(dict(event='source',summary=result)),flush=True)
'''
    while time.time()<deadline:
        host=c.instance()
        if host.get('cur_state')=='stopped':return 0
        try:
            result=subprocess.run([*c.ssh(host),'root@'+host['public_ipaddr'],'python3 -c '+shlex.quote(code)],
                      input=b'{}',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
        except subprocess.TimeoutExpired:
            print(json.dumps(dict(event='source',summary={'comparison_ssh_retry':True})),flush=True)
            time.sleep(15);continue
        current={}
        for line in result.stdout.splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if event.get('event')=='source':current=event.get('summary',{})
        signature=json.dumps(current,sort_keys=True)
        if signature!=previous:
            print(json.dumps(dict(event='source',summary=current)),flush=True);previous=signature
        if current.get('comparison',{}).get('stage') in ['complete','failed']:
            try:collect()
            finally:c.stop()
            return 0
        time.sleep(15)
    try:collect()
    finally:c.stop()
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['start','monitor','collect','supervise','trial'])
    op=p.parse_args().operation
    try:return globals()[op]()
    except Exception as e:
        if op in ('supervise','trial'):
            try:status('failed',error_class=type(e).__name__)
            except Exception:pass
        else:print(json.dumps(dict(event='operation_failed',error_class=type(e).__name__)),flush=True)
        return 1

if __name__=='__main__':sys.exit(main())
