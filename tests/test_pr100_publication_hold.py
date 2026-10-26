import hashlib
import importlib.util
import time
from pathlib import Path

import pytest

from scripts.pr100_publication_hold import patch_consumer, patch_signer, MARKER

ROOT=Path(__file__).resolve().parents[1]


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_hold_preserves_legacy_allocation_and_molecule_plan(tmp_path, capsys):
    fixture=ROOT/'tests/fixtures/pr100/pi_consumer_plan.py'
    source=fixture.read_bytes()
    path=tmp_path/'held_consumer.py'
    path.write_bytes(patch_consumer(source,expected_sha256=hashlib.sha256(source).hexdigest()))
    old=load(fixture,'_old_hold_test')
    held=load(path,'_held_test')
    artifact={'epoch':25954,'updated_at_unix':time.time(),'sequence_pool':['A'*120]}
    crew=[(7,'public-key')]
    assert held.sequences_by_uid(artifact,25954,crew=crew)==old.sequences_by_uid(artifact,25954,crew=crew)
    for epoch in [25955,25956,26000]:
        assert held.sequences_by_uid(None,epoch,allow_stale=True,crew=crew)=={}
        molecules=','.join('mol'+str(i) for i in range(20))
        plan=held.plan(crew,{7:molecules},{})
        assert len(plan)==1 and plan[0]['plaintext']==molecules+'|~'
    assert 'pr100_qualification_hold' in capsys.readouterr().out
    assert MARKER.encode() in path.read_bytes()


def test_hold_refuses_source_drift_and_double_patch():
    source=b'def sequences_by_uid(artifact, epoch):\n    return {1: "fixture"}\n'
    sha=hashlib.sha256(source).hexdigest()
    with pytest.raises(ValueError,match='consumer changed'):
        patch_consumer(source+b'\n',expected_sha256=sha)
    held=patch_consumer(source,expected_sha256=sha)
    with pytest.raises(ValueError,match='already present'):
        patch_consumer(held,expected_sha256=hashlib.sha256(held).hexdigest())


def test_actual_signer_null_half_preserves_all_molecule_validation(tmp_path):
    fixture=ROOT/'tests/fixtures/pr100/pi_signer_validate.py'
    source=fixture.read_bytes()
    path=tmp_path/'held_signer.py'
    path.write_bytes(patch_signer(source,expected_sha256=hashlib.sha256(source).hexdigest()))
    old=load(fixture,'_old_actual_signer');held=load(path,'_held_actual_signer')
    molecules=','.join(f'rxn:1:{i}:2' for i in range(20))
    for text in [molecules+'|ACDEFGHIKLMNPQRSTVWY','~|ACDEFGHIKLMNPQRSTVWY']:
        assert held.validate_plaintext(text)==old.validate_plaintext(text)==text
    assert held.validate_plaintext(molecules+'|~')==molecules+'|~'
    invalid=['~|~',molecules+'|',molecules+'|X',molecules+' |~',
             molecules.replace('rxn:1:0:2','rxn:2:0:2')+'|~',
             molecules.replace('rxn:1:0:2','rxn:1:1:2')+'|~',
             ','.join(f'rxn:45:{i}:2' for i in range(20))+'|~',
             ','.join('mol'+str(i) for i in range(20))+'|~']
    for text in invalid:
        with pytest.raises(ValueError):held.validate_plaintext(text)
    assert source.split(b'    molecule_ids =',1)[1]==path.read_bytes().split(b'    molecule_ids =',1)[1]
