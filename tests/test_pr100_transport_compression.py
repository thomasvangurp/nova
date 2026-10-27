"""Large archive requests keep identical contents across the actual SSH receiver."""
import gzip
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from nanobody_opt import pr100_live_controller as controller


@pytest.mark.parametrize('compressed', [False, True])
def test_actual_receiver_accepts_plain_and_compressed_requests(tmp_path, compressed):
    missing = tmp_path / 'never-launched'
    request = {'operation': 'poll', 'directory': str(missing)}
    payload = json.dumps(request).encode()
    if compressed:
        payload = gzip.compress(payload, compresslevel=1, mtime=0)
    result = subprocess.run([sys.executable, '-c', controller.TRANSPORT], input=payload,
                            capture_output=True, timeout=10)
    assert result.returncode == 0
    assert json.loads(result.stdout)['receipt']['reason'] == 'not_launched'
    assert not missing.exists()


def test_large_request_round_trips_all_fields_without_altering_identity(monkeypatch):
    request = {'operation': 'start', 'directory': '/inert',
               'files': {'archive.csv': 'repeated-public-archive-data-' * 10000},
               'spec': {'identity': 'unchanged', 'deadline_unix': 12345}}
    def run(_command, *, input, **_kwargs):
        assert input.startswith(b'\x1f\x8b')
        assert len(input) < len(json.dumps(request)) / 10
        assert json.loads(gzip.decompress(input)) == request
        return SimpleNamespace(returncode=0, stdout=b'{"running":true}', stderr=b'')
    monkeypatch.setattr(controller.subprocess, 'run', run)
    result = controller.remote({'hosts': {'master': {'port': 22, 'address': 'inert.invalid'}}},
                               'master', request)
    assert result == {'running': True}
