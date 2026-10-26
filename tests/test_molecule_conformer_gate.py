"""Exact-source/control tests. Fake RDKit boundaries do not qualify chemistry."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/pr100/molecule_gate'


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


gate=load(ROOT/'tools/molecule_gate.py','_test_molecule_gate')
patcher=load(ROOT/'scripts/patch_pi_molecule_gate.py','_test_pi_molecule_patch')


class Molecule:
    def GetAtoms(self):return [SimpleNamespace(GetSymbol=lambda:'C')]
    def GetNumHeavyAtoms(self):return 1


class FakeAllChem:
    def __init__(self,results=(0,),*,parse=True,rank=0):
        self.results=iter(results);self.parse=parse;self.rank=rank
        self.embeds=[];self.parameters=[];self.hydrogenated=Molecule()
    def MolFromSmiles(self,_):return Molecule() if self.parse else None
    def AddHs(self,_):return self.hydrogenated
    def CanonicalRankAtoms(self,_):return [self.rank]
    def ETKDGv3(self):
        value=SimpleNamespace(useRandomCoords=False);self.parameters.append(value);return value
    def EmbedMolecule(self,molecule,parameters):
        self.embeds.append((molecule,parameters,parameters.useRandomCoords))
        result=next(self.results)
        if isinstance(result,Exception):raise result
        return result


def compiled_upstream(fake,monkeypatch):
    monkeypatch.setattr(gate,'rdkit_identity',lambda:(fake,fake,{'fixture_only':True}))
    return gate.load_helpers(FIXTURE/'upstream_molecules.py')['boltz']


def install_fake_rdkit(monkeypatch,tmp_path,fake,version='2024.09.4'):
    modules={name:ModuleType(name) for name in ('rdkit','rdkit.Chem','rdkit.rdBase','rdkit.Chem.rdDistGeom')}
    for name,module in modules.items():
        path=tmp_path/(name.replace('.','_')+'.py');path.write_text('# explicit runtime fixture\n')
        module.__file__=str(path);monkeypatch.setitem(sys.modules,name,module)
    fake_path=tmp_path/'AllChem.py';fake_path.write_text('# fake RDKit boundary\n');fake.__file__=str(fake_path)
    modules['rdkit'].Chem=modules['rdkit.Chem'];modules['rdkit'].rdBase=modules['rdkit.rdBase']
    modules['rdkit.rdBase'].rdkitVersion=version
    modules['rdkit.Chem'].AllChem=fake
    modules['rdkit.Chem'].rdDistGeom=modules['rdkit.Chem.rdDistGeom']
    modules['rdkit.Chem'].MolFromSmiles=fake.MolFromSmiles
    monkeypatch.setattr(gate.importlib.metadata,'version',lambda name:'2024.9.4')
    return modules


def test_upstream_blob_and_bundled_function_are_exact():
    raw=(FIXTURE/'upstream_molecules.py').read_bytes()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==gate.UPSTREAM_BLOB_SHA1
    assert hashlib.sha256(raw).hexdigest()==gate.UPSTREAM_FILE_SHA256
    assert set(gate.source_functions(raw))==set(gate.FUNCTION_SHA256)
    assert set(gate.source_functions((ROOT/'utils/molecules.py').read_bytes()))==set(gate.FUNCTION_SHA256)


@pytest.mark.parametrize('results,ok,random_flags',[
    ([0],True,[False]),([-1,0],True,[False,True]),([-1,-1],False,[False,True]),
    ([RuntimeError('controlled native exception')],False,[False]),
])
def test_exact_two_attempt_behavior_and_pi_replacement_match(results,ok,random_flags,monkeypatch,tmp_path):
    original=FakeAllChem(results)
    expected=compiled_upstream(original,monkeypatch)('fixture')
    assert expected[0] is ok
    assert [row[2] for row in original.embeds]==random_flags
    assert len(original.parameters)==1
    assert all(row[0] is original.hydrogenated and row[1] is original.parameters[0] for row in original.embeds)
    assert not hasattr(original.parameters[0],'randomSeed')
    duplicate=FakeAllChem(results);install_fake_rdkit(monkeypatch,tmp_path,duplicate)
    namespace={};exec(compile(patcher.replacement_function(),'<exact-pi-helper>','exec'),namespace)
    assert namespace['boltz_safe']('fixture')==expected
    assert [row[2] for row in duplicate.embeds]==random_flags
    assert len(duplicate.embeds)<=2


@pytest.mark.parametrize('parse,rank',[(False,0),(True,10000)])
def test_parse_and_atom_name_refusal_precede_conformer_work(parse,rank,monkeypatch):
    fake=FakeAllChem(parse=parse,rank=rank)
    assert compiled_upstream(fake,monkeypatch)('fixture')[0] is False
    assert fake.embeds==[] and fake.parameters==[]


def test_source_drift_is_rejected_before_importing_rdkit(monkeypatch,tmp_path):
    source=(FIXTURE/'upstream_molecules.py').read_text().replace('params.useRandomCoords = True','params.useRandomCoords = False')
    path=tmp_path/'molecules.py';path.write_text(source)
    monkeypatch.setattr(gate,'rdkit_identity',lambda:pytest.fail('runtime reached before source verification'))
    with pytest.raises(gate.MoleculeGateUnavailable,match='source_mismatch'):
        gate.load_helpers(path)


def test_runtime_identity_requires_validator_rdkit_version(monkeypatch,tmp_path):
    install_fake_rdkit(monkeypatch,tmp_path,FakeAllChem())
    _,_,identity=gate.rdkit_identity()
    assert identity['rdkit_version']=='2024.09.4'
    assert len(identity['module_file_sha256'])==5
    assert identity['complete_native_dependency_closure'] is False
    sys.modules['rdkit.rdBase'].rdkitVersion='2025.09.1'
    with pytest.raises(gate.MoleculeGateUnavailable,match='version_mismatch'):
        gate.rdkit_identity()


def test_bank_helper_ignores_stale_utils_and_does_not_execute_module_setup(monkeypatch,tmp_path):
    bank=load(ROOT/'tools/bank_release.py','_test_bound_bank')
    tree=tmp_path/'validator';(tree/'neurons/validator').mkdir(parents=True)
    (tree/'neurons/validator/molecule_validity.py').write_text('# source presence fixture\n')
    (tree/'utils').mkdir();(tree/'config').mkdir()
    (tree/'utils/molecules.py').write_text('raise AssertionError("module startup must not execute")\n'+(FIXTURE/'upstream_molecules.py').read_text())
    (tree/'config/config.yaml').write_text('molecule_requirements: {}\n')
    fake=FakeAllChem([-1,-1]);install_fake_rdkit(monkeypatch,tmp_path,fake)
    monkeypatch.setitem(sys.modules,'utils',SimpleNamespace(is_boltz_safe_smiles=lambda _: (True,None)))
    helpers=bank.validator_helpers(tree)
    assert helpers['boltz']('fixture')==(False,'RDKit ETKDG failed to generate a conformer')
    identity=helpers['molecule_gate_identity']
    assert identity['function_sha256']==gate.FUNCTION_SHA256 and identity['runtime_qualified'] is False
    assert identity['source_path']==str((tree/'utils/molecules.py').resolve())


def test_pi_patcher_changes_only_the_bound_ast_span_and_refuses_drift():
    old=(FIXTURE/'pi_boltz_safe.py').read_bytes().rstrip(b'\n')
    prefix='# preceding café audit text\nUNCHANGED = 1\n\n'.encode()
    suffix=b'\n\n# exact trailing audit history\nOTHER = 2\n'
    source=prefix+old+suffix
    changed=patcher.patch_pi_boltz_safe(source,expected_sha256=hashlib.sha256(source).hexdigest())
    assert changed==prefix+patcher.replacement_function()+suffix
    assert changed.startswith(prefix) and changed.endswith(suffix)
    with pytest.raises(ValueError,match='source changed'):
        patcher.patch_pi_boltz_safe(source)  # CLI's audited installed-file pin.
    with pytest.raises(ValueError,match='already patched'):
        patcher.patch_pi_boltz_safe(changed,expected_sha256=hashlib.sha256(changed).hexdigest())
    drift=source.replace(b'AllChem.AddHs(mol)',b'AllChem.AddHs(mol, addCoords=True)')
    with pytest.raises(ValueError,match='function changed'):
        patcher.patch_pi_boltz_safe(drift,expected_sha256=hashlib.sha256(drift).hexdigest())


def test_post_activation_release_cannot_bypass_missing_gate_with_no_archive(monkeypatch,tmp_path,capsys):
    bank=load(ROOT/'tools/bank_release.py','_test_modern_bank')
    path=tmp_path/'bank.sqlite';connection=sqlite3.connect(path)
    connection.execute('CREATE TABLE candidates(name,smiles,inchikey,heavy_atoms,rotatable_bonds,components,affinity_probability_binary,affinity_pred_value,reaction)')
    connection.execute('CREATE TABLE rescored(name,mean_score,passes)');connection.close()
    monkeypatch.setattr(bank,'validator_helpers',lambda:None)
    output=tmp_path/'release.json'
    monkeypatch.setattr(sys,'argv',['bank_release.py','--bank',str(path),'--epoch','25955','--no-archive','--output',str(output)])
    assert bank.main()==1 and not output.exists()
    assert 'molecule_gate_unavailable' in capsys.readouterr().err


def test_molecule_gate_suite_is_mandatory_in_shipping_ci():
    from scripts.shipping_checks import TEST_FILES
    assert 'tests/test_molecule_conformer_gate.py' in TEST_FILES
