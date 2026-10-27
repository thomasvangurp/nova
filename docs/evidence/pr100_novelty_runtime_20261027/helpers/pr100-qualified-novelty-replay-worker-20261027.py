"""Run inside a fresh, bounded qualifier directory; emit only hashes/counts."""
import contextlib,hashlib,io,json,pathlib,subprocess,time
from nanobody_opt import pr100_novelty_runtime as runtime
from nanobody_opt import rules
base=pathlib.Path(__file__).parent
settings=json.loads((base/'qualification.json').read_text());rows=json.loads((base/'input.json').read_text())['candidates']
result={'status':'running','runtime_sha256':settings['runtime']['identity_sha256'],'input_count':len(rows),'input_sha256':rules.digest(rows),'snapshots':{}}
def save():
 temp=base/'result.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(base/'result.json')
try:
 save()
 for label in ('old','new'):
  captured={};start=time.monotonic()
  def run(*args,**kwargs):
   p=subprocess.run(*args,**kwargs)
   if p.returncode==0:captured['reply']=json.loads(p.stdout)
   return p
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
   kept,count=runtime.check(rows,base/(label+'-archive.csv'),runtime=settings['runtime'],run=run)
  reply=captured['reply'];result['snapshots'][label]={
   'archive_sha256':reply['archive_sha256'],'archive_rows':count,
   'archive_cdr_available':reply['archive_cdr_available'],'kept_count':len(kept),
   'elapsed_seconds':time.monotonic()-start,'verdicts':reply['verdicts']}
  save()
 result['status']='complete';save()
except Exception as e:
 result.update(status='failed',error_type=type(e).__name__);save();raise SystemExit(1)
print(json.dumps({'status':'complete','input_count':len(rows),'kept':{k:v['kept_count'] for k,v in result['snapshots'].items()}}))
