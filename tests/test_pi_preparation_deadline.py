import ast, hashlib, subprocess, sys, time
from pathlib import Path
import pytest
from scripts import patch_pi_preparation_deadline as patcher
ROOT=Path(__file__).resolve().parents[1]
def helper():
    ns={'time':time,'subprocess':subprocess};exec(compile(patcher.helper_source(),str(ROOT/'tools/pi_epoch_loop.py'),'exec'),ns);return ns['run_preparation']
def test_expired_never_spawns():
    assert helper()(['must-not-execute'],cutoff_at=time.time()+1,timeout=100).returncode==124
@pytest.mark.parametrize('cutoff,timeout',[(float('inf'),1),(float('nan'),1),(123,float('inf'))])
def test_nonfinite_refused(cutoff,timeout):
    with pytest.raises(ValueError,match='finite'):helper()(['must-not-execute'],cutoff_at=cutoff,timeout=timeout)
def test_absolute_cutoff_overrides_timeout():
    t=time.monotonic();r=helper()([sys.executable,'-c','import time;time.sleep(30)'],cutoff_at=time.time()+2.2,timeout=20)
    assert r.returncode==124 and time.monotonic()-t<2
def test_nonzero_status_preserved():
    r=helper()([sys.executable,'-c','import sys;print("refused");sys.exit(3)'],cutoff_at=time.time()+10,timeout=2)
    assert r.returncode==3 and r.stdout=='refused\n'
def test_grandchild_killed_and_next_epoch_starts(tmp_path):
    marker=tmp_path/'child.pid';late=tmp_path/'old-epoch-write'
    child='import time,pathlib;time.sleep(8);pathlib.Path('+repr(str(late))+').touch()'
    parent='import subprocess,sys,time,pathlib;p=subprocess.Popen([sys.executable,"-c",'+repr(child)+']);pathlib.Path('+repr(str(marker))+').write_text(str(p.pid));time.sleep(30)'
    t=time.monotonic();r=helper()([sys.executable,'-c',parent],cutoff_at=time.time()+20,timeout=1)
    assert r.returncode==124 and time.monotonic()-t<4 and marker.exists()
    probe=subprocess.run(['ps','-o','stat=','-p',marker.read_text()],capture_output=True,text=True,timeout=2)
    assert not probe.stdout.strip() or probe.stdout.strip().startswith('Z')
    assert not late.exists()
    assert helper()([sys.executable,'-c','pass'],cutoff_at=time.time()+10,timeout=2).returncode==0
def raw_fixture():return b'# prefix\n'+(ROOT/'tests/fixtures/pr100/molecule_gate/pi_handle_epoch.py').read_bytes()+b'\n# suffix\n'
def test_three_pi_calls_preserve_scientific_handler_ast():
    raw=raw_fixture();fixed=patcher.patch_pi_preparation(raw,expected_sha256=hashlib.sha256(raw).hexdigest())
    assert fixed.startswith(b'# prefix\n') and fixed.endswith(b'\n# suffix\n')
    old=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='handle_epoch')
    new=next(n for n in ast.parse(fixed).body if isinstance(n,ast.FunctionDef) and n.name=='handle_epoch')
    calls=sorted([n for n in ast.walk(new) if isinstance(n,ast.Call) and ast.unparse(n.func)=='run_preparation'],key=lambda n:n.lineno)
    assert len(calls)==3
    assert ast.unparse(next(k.value for k in calls[0].keywords if k.arg=='cutoff_at'))==ast.unparse(ast.parse(patcher.PRIMARY_CUTOFF,mode='eval').body)
    assert all(ast.unparse(next(k.value for k in n.keywords if k.arg=='cutoff_at'))=='collect_at' for n in calls[1:])
    for n in calls:
        n.func=ast.Attribute(value=ast.Name(id='subprocess',ctx=ast.Load()),attr='run',ctx=ast.Load());n.keywords=[k for k in n.keywords if k.arg!='cutoff_at']
    assert ast.dump(old)==ast.dump(new)
def test_wrong_source_drift_duplicate_refused():
    raw=raw_fixture();h=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError,match='source changed'):patcher.patch_pi_preparation(raw)
    fixed=patcher.patch_pi_preparation(raw,expected_sha256=h)
    with pytest.raises(ValueError,match='already patched'):patcher.patch_pi_preparation(fixed,expected_sha256=hashlib.sha256(fixed).hexdigest())
    bad=raw.replace(b'timeout=1200',b'timeout=1201',1)
    with pytest.raises(ValueError,match='handler changed'):patcher.patch_pi_preparation(bad,expected_sha256=hashlib.sha256(bad).hexdigest())
def test_local_integration_and_required_ci():
    s=(ROOT/'tools/pi_epoch_loop.py').read_text();calls=[n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Call) and ast.unparse(n.func) in {'subprocess.run','run_preparation'} and 'tools/make_fleet.py' in ast.get_source_segment(s,n)]
    assert len(calls)==2
    assert all(ast.unparse(n.func)=='run_preparation' and any(k.arg=='cutoff_at' for k in n.keywords) for n in calls)
    assert 'tests/test_pi_preparation_deadline.py' in (ROOT/'scripts/shipping_checks.py').read_text()
