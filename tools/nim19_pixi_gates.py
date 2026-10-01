#!/usr/bin/env python3
"""Allowlisted compute gates; capture existing helpers without disclosing logs."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

BASE = Path('/root/nim19-test')
RUN = BASE/'epoch'
sys.path.insert(0,str(BASE/'oss-repo'))
CURRENT_GATE='credential_environment_absent'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def save(name, value):
    path=RUN/name; tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value)); tmp.replace(path)

def main():
    global CURRENT_GATE
    started=time.monotonic()
    # The legacy preflight sources this path if present. Never read or source
    # credential files: reject such a host before invoking it.
    assert not Path('/root/nova.env').exists()
    assert not Path('/root/.bittensor').exists()
    if '--preflight' in sys.argv:
        CURRENT_GATE='original_compute_preflight'
        env=dict(os.environ, NOVA_MODE='compute')
        result=subprocess.run(['bash', str(BASE/'client/preflight.sh')], env=env,
            capture_output=True, timeout=240)
        found=re.search(rb'PREFLIGHT: (\d+) passed, (\d+) failed, (\d+) warnings', result.stdout)
        assert result.returncode==0 and found is not None and int(found[2])==0
        # No raw log, exception, inherited environment or auth field is emitted.
        save('pixi-preflight.json', dict(stage='complete',passed=int(found[1]),
            failed=int(found[2]),warnings=int(found[3]),elapsed_s=time.monotonic()-started))
        return
    CURRENT_GATE='scoring_stack_imports'
    import numpy as np
    import torch
    from compound_miner import gpu_novelty
    from cuequivariance_ops_torch import triangle_multiplicative_update
    from boltz.model.layers.triangular_mult import TriangleMultiplicationOutgoing
    CURRENT_GATE='eight_gpu_cuda_reference_runtime'
    assert torch.cuda.device_count()==8 and torch.version.cuda=='12.8'
    assert np.__version__=='1.26.4'
    for index in range(8):
        a=torch.ones((32,32),device=f'cuda:{index}')
        assert (a@a).sum().item()==32768
    import boltz.main
    CURRENT_GATE='vendored_source_path'
    assert Path(boltz.main.__file__).resolve()==BASE/'oss-repo/external_tools/boltz/src/boltz/main.py'
    CURRENT_GATE='msa_count_and_sha256'
    msas=list((BASE/'oss-repo/data/msa_files').glob('*.a3m'))
    assert len(msas)==21
    expected=json.loads((RUN/'msa-checksums.json').read_text())
    assert {p.name:digest(p) for p in msas}==expected
    CURRENT_GATE='checkpoint_ccd_sha256'
    expected_assets=json.loads((RUN/'asset-checksums.json').read_text())
    assert all(digest(BASE/'models'/name)==sha for name,sha in expected_assets.items())
    # Compile AND execute the novelty extension; import alone is insufficient.
    CURRENT_GATE='cuda_novelty_compile_and_execute'
    ext=gpu_novelty._cuda_ext()
    q=torch.tensor([[1,3],[0,0]],dtype=torch.int32,device='cuda')
    r=torch.tensor([[1,1],[0,0]],dtype=torch.int32,device='cuda')
    qc=torch.tensor([3,0],dtype=torch.int32,device='cuda')
    rc=torch.tensor([2,0],dtype=torch.int32,device='cuda')
    observed=ext.tanimoto_max(q,r,qc,rc).cpu().numpy()
    assert np.allclose(observed,[2/3,1],atol=1e-6)
    # Exercise the fused Boltz entrypoint too: an import does not prove the
    # package contains a usable kernel for this GPU architecture.
    CURRENT_GATE='fused_boltz_kernel_execute'
    layer=TriangleMultiplicationOutgoing(dim=128).cuda().eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        layer.p_out.weight.normal_(std=0.01)
        fused=layer(torch.randn(1,32,32,128,device='cuda'),
                    torch.ones(1,32,32,device='cuda'),use_kernels=True)
        assert fused.shape==(1,32,32,128) and torch.isfinite(fused).all().item()
    CURRENT_GATE='hostile_pythonpath_isolation'
    hostile=dict(os.environ,PYTHONPATH='/root/.pythonpath_decoy:/root/.pixi/envs/chain310/lib/python3.10/site-packages')
    result=subprocess.run(['/root/search_python.sh','-c','import numpy; assert numpy.__version__ == "1.26.4"'],
        env=hostile,capture_output=True,timeout=60)
    assert result.returncode==0
    CURRENT_GATE='validator_gate_audit'
    result=subprocess.run(['/root/search_python.sh','/root/tools/gate_audit.py'],
        capture_output=True,timeout=60)
    assert result.returncode==0 and b'GATES NOT FOUND IN OUR CODE: none' in result.stdout
    save('pixi-gates.json',dict(stage='complete',passed=True,gpus=8,torch_version=torch.__version__,
        cuda_version=torch.version.cuda,msa_count=21,asset_sha256_verified=True,
        cuda_novelty_executed=True,fused_kernel_imports=True,fused_kernel_executed=True,hostile_pythonpath_isolated=True,
        validator_gate_audit=True,manifest_sha256=digest(Path('/root/pixi.toml')),
        lock_sha256=digest(Path('/root/pixi.lock')),elapsed_s=time.monotonic()-started))

if __name__=='__main__':
    try: main()
    except Exception as e:
        known_modules={'compound_miner','boltz','torch','numpy','cuequivariance_ops_torch','cuequivariance_torch'}
        module=getattr(e,'name',None)
        save('pixi-gates-failure.json',dict(stage='failed',error_class=type(e).__name__,gate=CURRENT_GATE,
            missing_module=module if module in known_modules else None))
        sys.exit(1)
