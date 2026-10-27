"""Exact pinned lifecycle method AST; fake discovery only, no hardware/science."""
import ast,asyncio,hashlib,json,pathlib,types
SOURCE=pathlib.Path(__file__).resolve().parents[3]/'NOVA-nanobody-filter/metanano/utils/gpu_scheduler.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='13048ae69dcc85ae1048fff83d0e63ddd67590354478671af5aa4515dd77181a'
METHODS={'__init__','initialize','shutdown'}
def make_class():
 tree=ast.parse(SOURCE.read_text());original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPUScheduler');body=[n for n in original.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in METHODS]
 assert {n.name for n in body}==METHODS
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='GPUScheduler',bases=[],keywords=[],body=body,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(module)
 namespace={'asyncio':asyncio,'logger':types.SimpleNamespace(info=lambda *a,**k:None),'GPUConfig':lambda **k:types.SimpleNamespace(**k)};exec(compile(module,str(SOURCE),'exec'),namespace);return namespace['GPUScheduler']
def config():return types.SimpleNamespace(queue_max_size=64,enabled=True,gpus=[],auto_detect=True,default_max_concurrent_per_gpu=1)
async def empty_detect(self):await asyncio.sleep(0);return []
def cross_loop():
 cls=make_class();cls._detect_gpus=empty_detect;s=cls(config())
 async def arm():await asyncio.gather(s.initialize(),s.initialize());await s.shutdown()
 with asyncio.Runner() as runner:runner.run(arm())
 caught=None
 try:
  with asyncio.Runner() as runner:runner.run(arm())
 except RuntimeError as error:
  caught='different_event_loop' if 'different event loop' in str(error) else 'unexpected_runtime_error'
 assert caught=='different_event_loop';return {'first_runner':'completed','second_runner':caught}
def health_race():
 cls=make_class();tasks=[]
 async def detect(self):await asyncio.sleep(0);return [0]
 async def register(self,value):await asyncio.sleep(0);self._gpus[value.index]=object()
 async def health(self):
  try:await asyncio.sleep(3600)
  except asyncio.CancelledError:raise
 cls._detect_gpus=detect;cls._register_gpu=register;cls._health_check_loop=health;s=cls(config())
 async def arm():
  await asyncio.gather(*(s.initialize() for _ in range(4)))
  tasks.extend(t for t in asyncio.all_tasks() if t is not asyncio.current_task() and getattr(t.get_coro(),'__qualname__','').endswith('health'))
  await s.shutdown();return {'health_tasks_created':len(tasks),'pending_after_scheduler_shutdown':sum(not t.done() for t in tasks)}
 with asyncio.Runner() as runner:value=runner.run(arm())
 value['pending_after_runner_close']=sum(not t.done() for t in tasks)
 assert value=={'health_tasks_created':4,'pending_after_scheduler_shutdown':3,'pending_after_runner_close':0};return value
if __name__=='__main__':
 result={'schema':'pr100-pinned-scheduler-inert-reproduction-v1','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'unmodified_method_ast':sorted(METHODS),'cross_runner':cross_loop(),'concurrent_initialization':health_race(),'native_calls':0,'gpu_calls':0,'hardware_probes':0,'limitation':'Lifecycle-only reproduction; fake asynchronous discovery/register/health routines, original initialization/shutdown bodies; production phases use one Runner in separate worker interpreters.'}
 p=pathlib.Path(__file__).with_suffix('.json');p.write_text(json.dumps(result,indent=2));print(json.dumps(result))
