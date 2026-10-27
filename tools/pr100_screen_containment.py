"""Inert Linux host receipt: dead coordinator must leave no detached child."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nanobody_opt.pr100_native_screen import _guard

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    forbidden=('torch','openmm','ImmuneBuilder','theraprofnano')
    if any(name.split('.')[0] in forbidden for name in sys.modules):
        raise RuntimeError('native_imports_forbidden')
    def launch(_command,**kwargs):
        code=('import subprocess,sys,time,os\n'
              'subprocess.Popen([sys.executable,"-S","-c","import time; time.sleep(300)"],start_new_session=True)\n'
              'time.sleep(.15)\nos._exit(7)\n')
        return subprocess.Popen([sys.executable,'-S','-c',code],**kwargs)
    started=time.monotonic()
    _guard({'directory':str(args.output),'timeout_seconds':10},launch=launch)
    receipt=json.loads((args.output/'guard.json').read_text())
    receipt.update(native_calls=0,elapsed_seconds=time.monotonic()-started)
    receipt['passed']=(receipt.get('coordinator_exit_code')==7 and
        receipt['cleanup']['remaining_descendants']==0 and receipt['cleanup']['fixed_point_reached'] and
        receipt['cleanup']['killed_descendants']>=1)
    (args.output/'containment.json').write_text(json.dumps(receipt))
    print(json.dumps(receipt))
    return 0 if receipt['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
