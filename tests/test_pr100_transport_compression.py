"""Large archive requests keep identical contents across the actual SSH receiver."""
import gzip
import base64
import json
import lzma
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


def test_actual_receiver_compresses_complete_native_artifacts_losslessly(tmp_path):
    directory = tmp_path / 'completed'; directory.mkdir()
    receipt = {'reason': 'finished', 'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}}
    (directory / 'supervisor-receipt.json').write_text(json.dumps(receipt))
    native = json.dumps({'public_scientific_evidence': ['unchanged evidence bytes'] * 10000}).encode()
    (directory / 'result.json').write_bytes(native)
    (directory / 'native-confirmations.json').write_bytes(native)
    request = {'operation': 'poll', 'directory': str(directory), 'clock': {'epoch': 26021},
               'compress_response': True}
    result = subprocess.run([sys.executable, '-c', controller.TRANSPORT], input=json.dumps(request).encode(),
                            capture_output=True, timeout=10)
    assert result.returncode == 0 and result.stdout.startswith(controller.REPLY_PREFIX)
    assert len(result.stdout) < len(native) / 10
    value = controller.decode_transport_reply(result.stdout)
    assert value['receipt'] == receipt and value['running'] is False
    assert all(base64.b64decode(raw) == native for raw in value['files'].values())


@pytest.mark.parametrize('kind', ['too_large', 'truncated', 'trailing', 'invalid_base64'])
def test_compressed_reply_refuses_unbounded_or_incomplete_stream(monkeypatch, kind):
    monkeypatch.setattr(controller, 'MAX_REPLY_BYTES', 1024)
    data = json.dumps({'evidence': 'a' * (2000 if kind == 'too_large' else 20)}).encode()
    packed = lzma.compress(data, preset=3)
    if kind == 'truncated': packed = packed[:-4]
    if kind == 'trailing': packed += b'not part of the result'
    encoded = base64.b64encode(packed)
    if kind == 'invalid_base64': encoded = b'@' + encoded
    with pytest.raises(ValueError):
        controller.decode_transport_reply(controller.REPLY_PREFIX + encoded)


def test_compressed_reply_accepts_banner_but_keeps_json_compatibility():
    value = {'running': False, 'files': {}}
    raw = json.dumps(value).encode()
    assert controller.decode_transport_reply(b'login banner\n' + raw) == value
    assert controller.decode_transport_reply(b'login banner\n' + controller.REPLY_PREFIX
        + base64.b64encode(lzma.compress(raw, preset=3))) == value
    assert controller.decode_transport_reply(controller.GZIP_REPLY_PREFIX
        + base64.b64encode(gzip.compress(raw))) == value


def test_xz_reply_cannot_allocate_unbounded_decoder_dictionary(monkeypatch):
    packed = lzma.compress(b'{"running":false}', preset=6)
    monkeypatch.setattr(controller, 'MAX_REPLY_DECODER_MEMORY', 1024 * 1024)
    with pytest.raises(lzma.LZMAError):
        controller.decode_transport_reply(controller.REPLY_PREFIX + base64.b64encode(packed))


def test_compressed_response_requires_explicit_config_opt_in(monkeypatch):
    seen = []
    def run(_command, *, input, **_kwargs):
        seen.append(json.loads(input))
        return SimpleNamespace(returncode=0, stdout=b'{"running":true}', stderr=b'')
    monkeypatch.setattr(controller.subprocess, 'run', run)
    base = {'hosts': {'master': {'port': 22, 'address': 'inert.invalid'}}}
    request = {'operation': 'poll', 'directory': '/inert'}
    controller.remote(base, 'master', request)
    controller.remote({**base, 'transport_compress_results': True}, 'master', request)
    assert 'compress_response' not in seen[0] and seen[1]['compress_response'] is True
    assert 'compress_response' not in request
