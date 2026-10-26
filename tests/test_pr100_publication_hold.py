import hashlib
import importlib.util
import ast
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.pr100_publication_hold import patch_consumer, patch_signer, publication_routes, verified_restore, MARKER

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
    source=(ROOT/'tests/fixtures/pr100/pi_consumer_plan.py').read_bytes()
    sha=hashlib.sha256(source).hexdigest()
    with pytest.raises(ValueError,match='consumer changed'):
        patch_consumer(source+b'\n',expected_sha256=sha)
    held=patch_consumer(source,expected_sha256=sha)
    with pytest.raises(ValueError,match='already present'):
        patch_consumer(held,expected_sha256=hashlib.sha256(held).hexdigest())


@pytest.mark.parametrize('before,after,message',[
    ('def plan(', 'def renamed_plan(', 'planning function'),
    ('if not seq:', 'if seq is None:', 'planning restriction'),
    ('plaintext=f"{mols or NULL}|{seq}",', 'plaintext=f"{mols}|{seq}",', 'replacement'),
    ('has_sequence=True)', 'has_sequence=bool(seq))', 'replacement'),
    ('import json', 'import json as renamed_json', 'import json'),
    ('sequences = sequences_by_uid(', 'sequences = different_source(', 'publication route'),
    ('plan(crew, molecules, sequences, value)', 'plan(crew, molecules, other_sequences, value)', 'publication route'),
    ('rows = plan(crew, molecules, sequences, value)', 'sequences = {}\n    rows = plan(crew, molecules, sequences, value)', 'publication route'),
])
def test_hold_refuses_partial_repair_and_unguarded_routes(before,after,message):
    source=(ROOT/'tests/fixtures/pr100/pi_consumer_plan.py').read_text()
    assert before in source
    changed=source.replace(before,after).encode()
    with pytest.raises(ValueError,match=message):
        patch_consumer(changed,expected_sha256=hashlib.sha256(changed).hexdigest())


def test_hold_routes_and_verified_restore_rehearsal():
    source=(ROOT/'tests/fixtures/pr100/pi_consumer_plan.py').read_bytes()
    original_sha=hashlib.sha256(source).hexdigest()
    held=patch_consumer(source,expected_sha256=original_sha);held_sha=hashlib.sha256(held).hexdigest()
    assert publication_routes(held)['sequence_producer']=='sequences_by_uid'
    assert verified_restore(held,source,expected_current_sha256=held_sha,
                            expected_backup_sha256=original_sha)==source
    with pytest.raises(ValueError,match='rollback'):
        verified_restore(held+b'\n',source,expected_current_sha256=held_sha,
                         expected_backup_sha256=original_sha)


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


@pytest.mark.parametrize('failure', [FileNotFoundError, ValueError])
def test_hold_does_not_read_or_fetch_unusable_nanobody_artifacts(tmp_path, monkeypatch, failure):
    fixture=ROOT/'tests/fixtures/pr100/pi_consumer_plan.py'
    source=fixture.read_bytes();path=tmp_path/'held.py'
    path.write_bytes(patch_consumer(source,expected_sha256=hashlib.sha256(source).hexdigest()))
    held=load(path,'_held_io_test');reads=[];pulls=[]
    def read(path):
        reads.append(path.name)
        if path.name=='crew.json':return [[7,'public-key']]
        raise failure('unusable nanobody artifact')
    monkeypatch.setattr(held,'_load',read,raising=False)
    monkeypatch.setattr(held,'pull_producer',lambda *args:pulls.append(True),raising=False)
    args=SimpleNamespace(epoch=25955,pull=True,pull_only=False,vastai='unused',
        producer_label='unused',producer_path='unused',nanobody='nanobody.json',
        crew='crew.json',allow_stale_nanobody=True)
    assert held.main(args)=={}
    assert reads==['crew.json'] and pulls==[]
    args.pull_only=True
    assert held.main(args)==0
    assert reads==['crew.json'] and pulls==[]
    args.pull_only=False;args.epoch=25954
    with pytest.raises(failure):held.main(args)
    assert reads==['crew.json','crew.json','nanobody.json'] and pulls==[True]


def test_molecule_only_wire_round_trips_through_exact_pinned_validator_parser():
    source=ROOT/'tests/fixtures/pr100/validator/neurons/validator/commitments.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='c8daed3d9ef2fd347045f7bd28ea16e00ab83f39932efa51363e5e72b4e57e80'
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)
           and n.name in ('normalize_list','parse_decrypted_submission')]
    namespace={'NULL_TOKEN':'~','SEPARATOR':',',
        'bt':SimpleNamespace(logging=SimpleNamespace(warning=lambda *a:None))}
    module=ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
    molecules=[f'rxn:1:{i}:2' for i in range(20)]
    assert namespace['parse_decrypted_submission'](7,','.join(molecules)+'|~',20,1)==(molecules,[])
