"""Observer contract tests using inert scientific doubles, not parity evidence."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import random

import numpy as np
import pytest

from nanobody_opt.pr100_context_observer import Observer
from nanobody_opt import pr100_context_observer, pr100_validator_reference
import hashlib


def scientific_doubles(successes=(True,)):
    calls = []
    residue = NS(name='ALA', id='1', insertionCode='', chain=NS(id='H'))
    atoms = [NS(index=i, name=name, residue=residue,
                element=NS(symbol=element, atomic_number=number))
             for i, (name, element, number) in enumerate((('CA', 'C', 6), ('H', 'H', 1)))]
    topology = NS(atoms=lambda: iter(atoms))

    class Positions:
        def __init__(self, value=None): self.value = np.array(value if value is not None else [[1, 2, 3], [4, 5, 6]], dtype=float)
        def value_in_unit(self, unit): return self.value

    class Platform:
        def __init__(self, name): self.name = name
        def getName(self): return self.name
        def getPropertyNames(self): return ['Threads'] if self.name == 'CPU' else []
        def getPropertyValue(self, context, name): return '1'

    class Context:
        def __init__(self, platform='CPU', fail=False):
            if fail: raise ArithmeticError('scientific_failure')
            self.platform = Platform(platform)
        def getPlatform(self): return self.platform
        def getState(self, getPositions=False): return NS(getPositions=lambda: Positions())

    alias_before_install = Context

    class PDBFile:
        def __init__(self, path): self.topology, self.positions = topology, Positions()
        @staticmethod
        def writeFile(topology, positions, stream, keepIds=False):
            assert keepIds
            stream.write('INERT PDB DOUBLE\n')

    class Fixer:
        def __init__(self): self.topology, self.positions = topology, Positions()
        def addMissingAtoms(self, seed=None):
            calls.append(('missing_atoms', seed)); alias_before_install('Reference')

    class Modeller:
        def __init__(self): self.topology, self.positions = topology, Positions()
        def addHydrogens(self, forcefield, pH=7.0, platform=None):
            calls.append(('hydrogens', pH, platform)); alias_before_install('CUDA')

    native = NS()
    def minimize_energy(topology, positions, k1=2.5, k2=2.5, n_threads=-1):
        Modeller().addHydrogens('inert_forcefield')
        calls.append(('minimize', k1, k2, n_threads))
        return NS(topology=topology, context=alias_before_install('CPU'))
    def strained_sidechain_bonds_fixer(strained_residues, topology, positions, n_threads=-1):
        alias_before_install('CPU'); return topology, positions
    outcomes = iter(successes)
    def refine(input_file, output_file, check_for_strained_bonds=True, tries=3, n=6, n_threads=-1):
        calls.append(('refine', check_for_strained_bonds, tries, n, n_threads))
        fixer = Fixer(); fixer.addMissingAtoms()
        native.minimize_energy(topology, fixer.positions, n_threads=n_threads)
        native.strained_sidechain_bonds_fixer([], topology, fixer.positions, n_threads=n_threads)
        Path(output_file).write_text('INERT FINAL PDB\n')
        return next(outcomes)
    native.minimize_energy, native.strained_sidechain_bonds_fixer, native.refine = minimize_energy, strained_sidechain_bonds_fixer, refine

    class Tensor:
        def __init__(self, value): self.value = np.asarray(value, dtype=np.float32)
        def detach(self): return self
        def cpu(self): return self
        def contiguous(self): return self
        def numpy(self): return self.value
        def mean(self, dimension): return Tensor(self.value.mean(dimension))
        def tolist(self): return self.value.tolist()

    builder = NS(refine=refine)
    class Nanobody:
        def __init__(self):
            self.atoms = [Tensor([[i, 1, 2], [3, 4, 5]]) for i in range(4)]
            self.error_estimates = Tensor([[2, 2], [1, 1], [3, 3], [4, 4]])
            self.ranking = [1, 0, 2, 3]
        def save_single_unrefined(self, filename, index=0): Path(filename).write_text('INERT UNREFINED ' + str(index))
        def save(self, filename=None, check_for_strained_bonds=True, n_threads=-1):
            for index in self.ranking:
                for _ in range(2):
                    self.save_single_unrefined(filename, index)
                    if builder.refine(filename, filename, check_for_strained_bonds=check_for_strained_bonds, n_threads=n_threads): return
    class NanoBodyBuilder2:
        device = 'cuda'
        models = {f'nanobody_model_{i}': object() for i in range(1, 5)}
        def predict(self, sequences): return Nanobody()
    builder.Nanobody, builder.NanoBodyBuilder2 = Nanobody, NanoBodyBuilder2
    mm = NS(Context=Context, unit=NS(nanometer=object()), app=NS(Modeller=Modeller, PDBFile=PDBFile))
    return NS(mm=mm, fixer=NS(PDBFixer=Fixer), builder=builder, native=native,
              torch=NS(cuda=NS(is_available=lambda: True), get_num_threads=lambda: 1),
              topology=topology, Positions=Positions, calls=calls, alias=alias_before_install)


def read_events(path): return [json.loads(line) for line in path.read_text().splitlines()]


def test_observer_alias_context_coverage_preserves_parameters_and_rng(tmp_path):
    fake = scientific_doubles(); trace = tmp_path / 'trace.jsonl'; observer = Observer(trace)
    before_random = random.getstate()
    originals = (fake.mm.Context.__init__, fake.native.refine)
    observer.install(fake.mm, fake.fixer, fake.builder, fake.native, fake.torch)
    try:
        result = fake.builder.NanoBodyBuilder2().predict({'H': 'AAA'})
        result.save(str(tmp_path / 'result.pdb'), n_threads=1)
        observer.finish(); observer.finish()
    finally: observer.restore()
    events = read_events(trace)
    contexts = [r for r in events if r['event'] == 'context_created']
    assert {r['role'] for r in contexts} == {'missing_atoms', 'hydrogens', 'minimize_energy', 'sidechain_repair'}
    assert [r['platform'] for r in contexts] == ['Reference', 'CUDA', 'CPU', 'CPU']
    assert events[-1]['contexts'] == 4 and events[-1]['hooks_intact']
    assert sum(r['event'] == 'observer_finished' for r in events) == 1
    assert ('refine', True, 3, 6, 1) in fake.calls
    assert ('missing_atoms', None) in fake.calls
    assert ('hydrogens', 7.0, None) in fake.calls
    assert random.getstate() == before_random
    assert fake.mm.Context.__init__ is originals[0] and fake.native.refine is originals[1]
    prediction = next(r for r in events if r['event'] == 'prediction')
    assert prediction['ranking'] == [1, 0, 2, 3]
    assert prediction['confidence_mean_squared_disagreement'] == [2, 1, 3, 4]
    for r in prediction['coordinates']:
        assert np.array_equal(np.load(tmp_path / 'prediction_coordinates' / r['coordinate_file']), result.atoms[r['model_index']].value)
    validated = pr100_validator_reference.validate_trace(events,
        sequence_hash=hashlib.sha256(b'AAA').hexdigest(),
        expected_observer_sha256=pr100_context_observer.digest_file(pr100_context_observer.__file__),
        expected_cuda_available=True)
    assert validated['all_contexts_observed']
    stages = {r['stage'] for r in events if r['event'] == 'geometry_snapshot'}
    assert {'missing_atoms_before', 'missing_atoms_after', 'hydrogens_before', 'hydrogens_after', 'selected_unrefined', 'final_model'} <= stages


def test_observer_records_real_model_fallback_without_selecting_a_model(tmp_path):
    fake = scientific_doubles((False, False, True)); trace = tmp_path / 'trace.jsonl'; observer = Observer(trace)
    observer.install(fake.mm, fake.fixer, fake.builder, fake.native, fake.torch)
    try:
        fake.builder.NanoBodyBuilder2().predict({'H': 'AAA'}).save(str(tmp_path / 'model.pdb'), n_threads=1)
        observer.finish()
    finally: observer.restore()
    events = read_events(trace)
    assert [r['model_index'] for r in events if r['event'] == 'refine_started'] == [1, 1, 0]
    assert [r['success'] for r in events if r['event'] == 'refine_finished'] == [False, False, True]
    final = next(r for r in events if r['event'] == 'model_saved')
    assert final['model_index'] == 0 and final['refinement_success'] is True
    validated = pr100_validator_reference.validate_trace(events,
        sequence_hash=hashlib.sha256(b'AAA').hexdigest(),
        expected_observer_sha256=pr100_context_observer.digest_file(pr100_context_observer.__file__),
        expected_cuda_available=True)
    assert validated['final_refinement_success']


def test_observer_coordinate_hashes_distinguish_heavy_and_hydrogen_changes(tmp_path):
    fake = scientific_doubles(); observer = Observer(tmp_path / 'trace.jsonl'); observer.mm = fake.mm
    observer.geometry(fake.topology, fake.Positions(), stage='a')
    observer.geometry(fake.topology, fake.Positions([[1, 2, 3], [7, 8, 9]]), stage='b')
    observer.geometry(fake.topology, fake.Positions([[2, 2, 3], [7, 8, 9]]), stage='c')
    a, b, c = read_events(tmp_path / 'trace.jsonl')
    assert a['heavy_coordinate_sha256'] == b['heavy_coordinate_sha256']
    assert a['coordinate_sha256'] != b['coordinate_sha256']
    assert b['heavy_coordinate_sha256'] != c['heavy_coordinate_sha256']
    assert a['atom_count'] == 2 and a['heavy_atom_count'] == 1


def test_observer_records_context_failure_and_keeps_exception(tmp_path):
    fake = scientific_doubles(); trace = tmp_path / 'trace.jsonl'; observer = Observer(trace)
    observer.install(fake.mm, fake.fixer, fake.builder, fake.native, fake.torch)
    try:
        with pytest.raises(ArithmeticError):
            with observer.phase('missing_atoms'): fake.alias(fail=True)
        observer.finish()
    finally: observer.restore()
    events = read_events(trace)
    assert next(r for r in events if r['event'] == 'context_failed')['error_type'] == 'ArithmeticError'
    assert events[-1]['failed_contexts'] == 1 and events[-1]['contexts'] == 0


def test_observer_detects_hook_replacement_and_does_not_overwrite_it(tmp_path):
    fake = scientific_doubles(); trace = tmp_path / 'trace.jsonl'; observer = Observer(trace)
    observer.install(fake.mm, fake.fixer, fake.builder, fake.native, fake.torch)
    replacement = lambda *args, **kwargs: None
    fake.native.minimize_energy = replacement
    observer.finish(); observer.restore()
    assert read_events(trace)[-1]['hooks_intact'] is False
    assert fake.native.minimize_energy is replacement


def test_observer_rejects_nonfinite_coordinates_and_symlink_trace(tmp_path):
    fake = scientific_doubles(); observer = Observer(tmp_path / 'trace.jsonl'); observer.mm = fake.mm
    with pytest.raises(ValueError, match='unobservable_coordinates'):
        observer.geometry(fake.topology, fake.Positions([[float('nan'), 2, 3], [4, 5, 6]]), stage='invalid')
    target = tmp_path / 'target'; target.touch(); link = tmp_path / 'link'; link.symlink_to(target)
    with pytest.raises(ValueError): Observer(link)
