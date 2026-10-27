"""Read Pi submission receipts and pool cardinality without exposing raw logs."""
import json,subprocess,sys
wanted=int(sys.argv[1]) if len(sys.argv)>1 else 25967
remote='''import json,pathlib
wanted='''+repr(wanted)+''';epoch=None;events=[];success=set()
with pathlib.Path('/opt/nova-submit/loop.log').open() as stream:
 for line in stream:
  start=line.find('{')
  if start<0:continue
  try:d=json.loads(line[start:])
  except ValueError:continue
  if not isinstance(d,dict):continue
  if isinstance(d.get('epoch'),int):epoch=d['epoch']
  if epoch!=wanted or d.get('event') not in ['plan','commit','done','crew_refreshed','committed']:continue
  event={'event':d['event'],'observed_epoch':epoch}
  event.update({k:v for k,v in d.items() if k in ['epoch','uid','rc','crew','committing','committed','failed','before','after','with_sequence','with_molecules','skipped_empty'] and isinstance(v,(int,float,bool,type(None)))})
  if d['event']=='commit' and d.get('rc')==0 and isinstance(d.get('uid'),int):success.add(d['uid'])
  events.append(event)
p=pathlib.Path('/opt/nova-submit/nanobody_result.json');artifact=json.loads(p.read_text());crew=json.loads(pathlib.Path('/opt/nova-submit/crew.json').read_text())
print(json.dumps({'requested_epoch':wanted,'registered_crew_snapshot_count':len(crew),'publication':{'epoch':artifact.get('epoch'),'sequence_pool_count':len(artifact.get('sequence_pool',[])),'submission_count':artifact.get('submission_count'),'updated_at_unix':artifact.get('updated_at_unix')},'successful_commit_event_unique_uids':sorted(success),'events':events}))
'''
p=subprocess.run(['ssh','-T','-oBatchMode=yes','-oConnectTimeout=10','thomasg@100.64.243.65','python3','-'],input=remote,text=True,capture_output=True,timeout=40)
try:result=json.loads(p.stdout) if p.returncode==0 else None
except ValueError:result=None
print(json.dumps({'pi_exit':p.returncode,'result':result}))
