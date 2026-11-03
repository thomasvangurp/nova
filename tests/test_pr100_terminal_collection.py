"""Bounded transfer of completed science, without extending a worker's life."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from nanobody_opt import pr100_live_controller as c
from tests.test_pr100_continuation_lifecycle import Chain, config, invoke, terminal


def artifact_reply(blobs):
    receipt = terminal()['receipt']
    return {'running': False, 'receipt': receipt, 'files': {},
            'receipt_sha256': hashlib.sha256(json.dumps(receipt).encode()).hexdigest(),
            'artifact_manifest': {name: {'size_bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest()} for name, raw in blobs.items()}}


def test_expired_worker_probes_metadata_then_collects_large_files_in_existing_reserve(tmp_path, monkeypatch):
    cfg = {**config(tmp_path), '_bounded_phase_collection': True,
           'native_collection_reserve_seconds': 90}
    chain = Chain()
    with pytest.raises(RuntimeError):
        invoke(tmp_path, cfg, chain, lambda *args: (_ for _ in ()).throw(RuntimeError('interrupted')))
    intent = tmp_path / 'attempt/native-request.json'
    saved = json.loads(intent.read_text())
    saved['created_at_unix'] -= saved['request']['spec']['wall_seconds'] + 140
    intent.write_text(json.dumps(saved))
    blobs = {'result.json': b'{"status":"complete","candidates":[]}',
             'native-confirmations.json': b'{"records":[]}'}
    metadata = artifact_reply(blobs)
    now = [time.time()]
    monkeypatch.setattr(c.time, 'time', lambda: now[0])
    calls = []
    def transport(_config, _host, request):
        calls.append(request)
        assert request['operation'] == 'poll'
        assert 'spec' not in request and 'files' not in request
        if request['artifact_mode'] == 'manifest':
            assert request['transport_deadline_unix'] <= now[0] + 20
            return metadata
        # Actual slave reply preparation measured37sec. Model40sec per file,
        # shared within the configured90sec; a repeated20sec probe would fail.
        assert 40 < request['transport_deadline_unix'] - now[0] <= 90
        now[0] += 40
        name, = request['artifact_names']
        return {**metadata, 'files': {name: base64.b64encode(blobs[name]).decode()}}
    assert invoke(tmp_path, cfg, chain, transport) == blobs
    assert [r['artifact_mode'] for r in calls] == ['manifest', 'file', 'file']
    assert calls[1]['transport_deadline_unix'] == calls[2]['transport_deadline_unix']
    for name, raw in blobs.items():
        assert (tmp_path / 'attempt/native' / name).read_bytes() == raw


def test_terminal_collection_stops_before_crossing_epoch(tmp_path):
    chain = Chain()
    blobs = {'result.json': b'one', 'native-confirmations.json': b'two'}
    metadata = artifact_reply(blobs)
    request = {'directory': '/owned', 'clock': c.clock_state(chain)}
    calls = []
    def transport(_cfg, _host, selected):
        calls.append(selected)
        chain.block += 361
        name, = selected['artifact_names']
        return {**metadata, 'files': {name: base64.b64encode(blobs[name]).decode()}}
    with pytest.raises(TimeoutError, match='cutoff'):
        c.collect_terminal_artifacts({}, 'slave', request, metadata, time.time()+200,
                                     chain, 90, transport)
    assert len(calls) == 1


def test_terminal_collection_never_renews_its_reserve(tmp_path, monkeypatch):
    chain = Chain()
    blobs = {'result.json': b'original'}
    metadata = artifact_reply(blobs)
    now = [time.time()]
    monkeypatch.setattr(c.time, 'time', lambda: now[0])
    request = {'directory': '/owned', 'clock': c.clock_state(chain)}
    def transport(*_args):
        now[0] += 91
        return {**metadata, 'files': {'result.json': base64.b64encode(blobs['result.json']).decode()}}
    with pytest.raises(TimeoutError, match='cutoff'):
        c.collect_terminal_artifacts({}, 'slave', request, metadata, now[0]+200,
                                     chain, 90, transport)


@pytest.mark.parametrize('change', ['bytes', 'receipt', 'manifest', 'selection'])
def test_terminal_collection_rejects_changed_original_binding(change):
    chain = Chain()
    blobs = {'result.json': b'original'}
    metadata = artifact_reply(blobs)
    request = {'directory': '/owned', 'clock': c.clock_state(chain)}
    def transport(*_args):
        reply = {**metadata, 'files': {'result.json': base64.b64encode(blobs['result.json']).decode()}}
        if change == 'bytes':
            reply['files'] = {'result.json': base64.b64encode(b'changed').decode()}
        elif change == 'receipt':
            reply['receipt_sha256'] = 'f' * 64
        elif change == 'manifest':
            reply['artifact_manifest'] = {}
        else:
            reply['files']['native-confirmations.json'] = ''
        return reply
    with pytest.raises(ValueError, match='(binding|bytes)'):
        c.collect_terminal_artifacts({}, 'slave', request, metadata, time.time()+200,
                                     chain, 90, transport)


def protocol(tmp_path, **selection):
    request = {'operation': 'poll', 'directory': str(tmp_path),
               'clock': {'epoch': 1, 'block': 361, 'epoch_length': 361,
                         'updated_at_unix': time.time()}, **selection}
    return subprocess.run([sys.executable, '-c', c.TRANSPORT], input=json.dumps(request),
                          text=True, capture_output=True, timeout=10,
                          env={k: os.environ[k] for k in ('PATH', 'HOME', 'LANG') if k in os.environ})


def test_terminal_protocol_metadata_is_body_free_then_returns_exact_selected_file(tmp_path):
    receipt = json.dumps(terminal()['receipt']).encode()
    (tmp_path / 'supervisor-receipt.json').write_bytes(receipt)
    blobs = {'result.json': b'{"scientific":"original"}',
             'native-confirmations.json': b'{"records":[1,2,3]}'}
    for name, raw in blobs.items():
        (tmp_path / name).write_bytes(raw)
    first = protocol(tmp_path, artifact_mode='manifest')
    assert first.returncode == 0
    metadata = json.loads(first.stdout)
    assert metadata['files'] == {}
    assert metadata['receipt_sha256'] == hashlib.sha256(receipt).hexdigest()
    assert metadata['artifact_manifest'] == artifact_reply(blobs)['artifact_manifest']
    for name, raw in blobs.items():
        part = protocol(tmp_path, artifact_mode='file', artifact_names=[name])
        assert part.returncode == 0
        data = json.loads(part.stdout)
        assert data['artifact_manifest'] == metadata['artifact_manifest']
        assert set(data['files']) == {name}
        assert base64.b64decode(data['files'][name], validate=True) == raw


@pytest.mark.parametrize('names', [['../outside.json'], ['/tmp/outside.json'],
                                  ['result.json', 'native-confirmations.json'], []])
def test_terminal_protocol_refuses_arbitrary_paths_or_multiple_files(tmp_path, names):
    (tmp_path / 'supervisor-receipt.json').write_text(json.dumps(terminal()['receipt']))
    (tmp_path / 'result.json').write_text('{}')
    assert protocol(tmp_path, artifact_mode='file', artifact_names=names).returncode != 0


def test_terminal_collection_keeps_reply_bounds():
    assert c.MAX_REPLY_BYTES == 256 * 1024 * 1024
    assert c.MAX_COMPRESSED_REPLY_BYTES == 64 * 1024 * 1024
    assert c.MAX_REPLY_DECODER_MEMORY == 64 * 1024 * 1024
