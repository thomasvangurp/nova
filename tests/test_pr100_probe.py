"""The technical probe must cancel descendants and verify retained evidence."""
import hashlib
import json
import os
import sys
import time

import pytest

from scripts.pr100_gpu_probe import bounded, validate_result
from tools.slave_runtime.slave_multipass_writeback import unpack
from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL
from nanobody_opt.simulate_round import METRICS


def test_probe_timeout_stops_the_worker_descendant(tmp_path):
    marker = tmp_path/'descendant-finished'
    child = 'import time,pathlib;time.sleep(1);pathlib.Path('+repr(str(marker))+').write_text("done")'
    parent = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(child)+']);time.sleep(20)'
    result = bounded([sys.executable, '-c', parent], dict(os.environ), .2)
    assert result == {'exit': None, 'timed_out': True}
    time.sleep(1.1)
    assert not marker.exists()


def test_probe_diagnostics_do_not_persist_backend_output():
    result = bounded([sys.executable, '-c',
        'import sys;sys.stderr.write("private backend detail\\nValueError: fixture");sys.exit(1)'],
        dict(os.environ), 5)
    assert result['error_classes'] == ['ValueError']
    assert 'private backend detail' not in json.dumps(result)


@pytest.mark.parametrize('fault', [None, 'wrong_target', 'nan_metric', 'missing_replica',
                                 'wrong_sample_count', 'corrupt_structure', 'escaped_structure'])
def test_probe_rejects_bad_metric_or_structure_evidence(tmp_path, fault):
    seq='EXAMPLE'; key=hashlib.sha256(seq.encode()).hexdigest()+'0'; target='Q9NZQ7'
    p=tmp_path/'sample.cif';p.write_text('synthetic structure fixture')
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    record={'path':str(p), 'confidence_path':str(p), 'writer_path':str(p),
            'confidence_sha256':digest, 'writer_sha256':digest,
            'sample_selection':{'sample_count':5}}
    metrics={m:1. for m in METRICS}
    data={'target':target, 'n':1, 'scorer_protocol':SCORER_PROTOCOL,
          'per_nanobody_components':{'0':{key:{target:metrics}}}, 'structures':{key:{target:record}}}
    if fault=='wrong_target': data['target']='P20809'
    if fault=='nan_metric': metrics[next(iter(METRICS))]=float('nan')
    if fault=='missing_replica': data['per_nanobody_components']['0']={}
    if fault=='wrong_sample_count': record['sample_selection']['sample_count']=1
    if fault=='corrupt_structure': p.write_text('modified')
    if fault=='escaped_structure': record['confidence_path']=str(tmp_path.parent/'unreadable.cif')
    output=tmp_path/'exact_all.json';output.write_text(json.dumps(data))
    if fault:
        with pytest.raises(ValueError): validate_result(output,[seq],1,target,unpack)
    else:
        assert validate_result(output,[seq],1,target,unpack)['valid']
