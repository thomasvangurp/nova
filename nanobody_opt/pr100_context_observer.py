"""Opt-in observation of the unmodified TNP/NBB2 call chain in isolated probes.

This does not route devices, seed RNGs, change precision or alter return values.
Its trace is scientific evidence, not permission to publish a candidate.
"""
from __future__ import annotations

import atexit
import contextlib
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import threading
import time

SCHEMA = 'pr100-context-observer-v1'
SCOPES = ('openmm.Context.__init__', 'pdbfixer.PDBFixer.addMissingAtoms',
          'openmm.app.Modeller.addHydrogens', 'ImmuneBuilder.refine.minimize_energy',
          'ImmuneBuilder.refine.strained_sidechain_bonds_fixer',
          'ImmuneBuilder.NanoBodyBuilder2.predict',
          'ImmuneBuilder.Nanobody.save_single_unrefined', 'ImmuneBuilder.Nanobody.save')
_installed = None


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Observer:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_absolute() or self.path.is_symlink():
            raise ValueError('trace_path_must_be_absolute_regular_file')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.local = threading.local()
        self.lock = threading.RLock()
        self.counter = 0
        self.contexts = 0
        self.failed_contexts = 0
        self.phase_counts = {}
        self.hooks = []
        self.last_model = None
        self.last_success = None
        self.finished = False
        self.geometry_index = 0

    def geometry(self, topology, positions, *, stage, phase_id=None):
        """Persist exact coordinate arrays plus a convenient PDB, without mutation.

        The hashes bind ordered atom identities and float64 nanometer values.
        PDB bytes are separately hashed and are not mistaken for exact arrays.
        """
        import numpy as np
        atoms = list(topology.atoms())
        array = np.asarray(positions.value_in_unit(self.mm.unit.nanometer), dtype='<f8')
        if array.shape != (len(atoms), 3) or not np.isfinite(array).all():
            raise ValueError('unobservable_coordinates')
        ids = [[a.index, a.name, a.residue.name, str(a.residue.id),
                str(getattr(a.residue, 'insertionCode', '')), str(a.residue.chain.id),
                a.element.symbol if a.element is not None else None] for a in atoms]
        heavy = [i for i, a in enumerate(atoms) if a.element is not None and a.element.atomic_number > 1]
        def digest(indices):
            header = json.dumps({'units': 'nanometer', 'dtype': '<f8',
                                 'atoms': [ids[i] for i in indices]}, sort_keys=True).encode()
            return hashlib.sha256(header + np.ascontiguousarray(array[indices]).tobytes()).hexdigest()
        full_sha, heavy_sha = digest(list(range(len(atoms)))), digest(heavy)
        self.geometry_index += 1
        directory = self.path.parent / 'coordinates'; directory.mkdir(exist_ok=True)
        stem = f'{os.getpid()}-{self.geometry_index}-{stage}'
        arrays = directory / (stem + '.npz')
        with arrays.open('xb') as stream:
            np.savez_compressed(stream, positions_nm=array, heavy_indices=np.asarray(heavy, dtype='<i8'))
        pdb = directory / (stem + '.pdb')
        with pdb.open('x') as stream:
            self.mm.app.PDBFile.writeFile(topology, positions, stream, keepIds=True)
        self.emit('geometry_snapshot', stage=stage, phase_id=phase_id,
                  atom_count=len(atoms), heavy_atom_count=len(heavy),
                  unknown_element_count=sum(a.element is None for a in atoms),
                  coordinate_sha256=full_sha, heavy_coordinate_sha256=heavy_sha,
                  array_file=arrays.name, array_file_sha256=digest_file(arrays),
                  pdb_file=pdb.name, pdb_sha256=digest_file(pdb), atom_ids=ids)

    def pdb_geometry(self, path, *, stage):
        model = self.mm.app.PDBFile(str(path))
        self.geometry(model.topology, model.positions, stage=stage)

    def emit(self, event, **fields):
        import fcntl
        with self.lock:
            self.counter += 1
            row = dict(schema=SCHEMA, event=event, pid=os.getpid(),
                       event_index=self.counter, monotonic=time.monotonic(), **fields)
            data = (json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode()
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                with os.fdopen(fd, 'ab', closefd=False) as stream:
                    stream.write(data)
                    stream.flush()
            finally:
                os.close(fd)
            return self.counter

    def hook(self, owner, name, replacement):
        old = getattr(owner, name)
        self.hooks.append((owner, name, old, replacement))
        setattr(owner, name, replacement)
        return old

    @contextlib.contextmanager
    def phase(self, role):
        stack = getattr(self.local, 'stack', [])
        parent = stack[-1][1] if stack else None
        index = self.emit('phase_started', role=role, parent_phase_id=parent)
        self.phase_counts[index] = 0
        self.local.stack = [*stack, (role, index)]
        error = None
        try:
            yield index
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self.emit('phase_finished', role=role, phase_id=index,
                      context_count=self.phase_counts[index], error_type=error)
            self.local.stack = stack

    def phase_hook(self, owner, name, role):
        original = getattr(owner, name)
        signature = inspect.signature(original)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            bound = signature.bind(*args, **kwargs); bound.apply_defaults()
            target = args[0] if args and hasattr(args[0], 'topology') and hasattr(args[0], 'positions') else None
            topology = target.topology if target is not None else bound.arguments.get('topology')
            positions = target.positions if target is not None else bound.arguments.get('positions')
            with self.phase(role) as phase_id:
                public = {k: v for k, v in bound.arguments.items()
                          if k in ('seed', 'pH', 'n_threads', 'k1', 'k2')
                          and (v is None or type(v) in (int, float, bool))}
                platform = bound.arguments.get('platform')
                if 'platform' in bound.arguments:
                    public['requested_platform'] = None if platform is None else platform.getName()
                self.emit('phase_parameters', phase_id=phase_id, role=role, parameters=public)
                if topology is not None and positions is not None:
                    self.geometry(topology, positions, stage=role + '_before', phase_id=phase_id)
                result = original(*args, **kwargs)
                if target is not None:
                    topology, positions = target.topology, target.positions
                elif hasattr(result, 'topology') and hasattr(result, 'context'):
                    topology, positions = result.topology, result.context.getState(getPositions=True).getPositions()
                elif isinstance(result, tuple) and len(result) == 2 and hasattr(result[0], 'atoms'):
                    topology, positions = result
                else:
                    topology = positions = None
                if topology is not None and positions is not None:
                    self.geometry(topology, positions, stage=role + '_after', phase_id=phase_id)
                return result
        self.hook(owner, name, observed)

    def install(self, mm, fixer, builder, native, torch):
        self.mm = mm
        original_context_init = mm.Context.__init__
        @functools.wraps(original_context_init)
        def context_init(context, *args, **kwargs):
            stack = getattr(self.local, 'stack', [])
            role, phase_id = stack[-1] if stack else ('unclassified', None)
            try:
                original_context_init(context, *args, **kwargs)
            except BaseException as exc:
                self.failed_contexts += 1
                self.emit('context_failed', role=role, phase_id=phase_id,
                          error_type=type(exc).__name__)
                raise
            platform = context.getPlatform()
            props = {}
            for name in platform.getPropertyNames():
                if name in ('Threads', 'Precision', 'DeviceIndex', 'DeterministicForces'):
                    props[name] = platform.getPropertyValue(context, name)
            self.contexts += 1
            if phase_id is not None:
                self.phase_counts[phase_id] += 1
            self.emit('context_created', role=role, phase_id=phase_id,
                      platform=platform.getName(), properties=props)
        self.hook(mm.Context, '__init__', context_init)
        self.phase_hook(fixer.PDBFixer, 'addMissingAtoms', 'missing_atoms')
        self.phase_hook(mm.app.Modeller, 'addHydrogens', 'hydrogens')
        self.phase_hook(native, 'minimize_energy', 'minimize_energy')
        self.phase_hook(native, 'strained_sidechain_bonds_fixer', 'sidechain_repair')

        original_predict = builder.NanoBodyBuilder2.predict
        @functools.wraps(original_predict)
        def predict(model, sequences):
            available = bool(torch.cuda.is_available())
            result = original_predict(model, sequences)
            coords = []
            predictions_dir = self.path.parent / 'prediction_coordinates'
            predictions_dir.mkdir(exist_ok=True)
            for index, atoms in enumerate(result.atoms):
                array = atoms.detach().cpu().contiguous().numpy()
                header = json.dumps({'dtype': str(array.dtype), 'shape': list(array.shape)},
                                    sort_keys=True).encode()
                path = predictions_dir / f'{os.getpid()}-model-{index}.npy'
                import numpy as np
                with path.open('xb') as stream:
                    np.save(stream, array, allow_pickle=False)
                coords.append(dict(model_index=index,
                                   coordinate_sha256=hashlib.sha256(header + array.tobytes()).hexdigest(),
                                   coordinate_file=path.name, coordinate_file_sha256=digest_file(path)))
            errors = result.error_estimates.mean(-1).detach().cpu().tolist()
            self.emit('prediction', sequence_sha256=hashlib.sha256(sequences['H'].encode()).hexdigest(),
                      cuda_available=available, device=str(model.device),
                      torch_threads=int(torch.get_num_threads()),
                      model_ids=[int(k.rsplit('_', 1)[1]) for k in model.models],
                      ranking=[int(i) for i in result.ranking],
                      confidence_mean_squared_disagreement=[float(v) for v in errors], coordinates=coords)
            return result
        self.hook(builder.NanoBodyBuilder2, 'predict', predict)

        original_save_single = builder.Nanobody.save_single_unrefined
        @functools.wraps(original_save_single)
        def save_single(model, filename, index=0):
            result = original_save_single(model, filename, index=index)
            self.last_model = int(index)
            data = Path(filename).read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            prepared = self.path.parent / 'prepared'
            prepared.mkdir(exist_ok=True)
            destination = prepared / f'{os.getpid()}-{self.counter + 1}-{sha}.pdb'
            with destination.open('xb') as stream:
                stream.write(data)
            self.emit('unrefined_saved', model_index=int(index), pdb_sha256=sha,
                      prepared_file=destination.name)
            self.pdb_geometry(filename, stage='selected_unrefined')
            return result
        self.hook(builder.Nanobody, 'save_single_unrefined', save_single)

        original_refine = native.refine
        signature = inspect.signature(original_refine)
        @functools.wraps(original_refine)
        def refine(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            public = {k: bound.arguments[k] for k in ('check_for_strained_bonds', 'tries', 'n', 'n_threads')}
            self.emit('refine_started', model_index=self.last_model, **public)
            with self.phase('refine'):
                result = original_refine(*args, **kwargs)
            self.last_success = bool(result)
            self.emit('refine_finished', model_index=self.last_model, success=bool(result))
            return result
        self.hook(native, 'refine', refine)
        self.hook(builder, 'refine', refine)

        original_save = builder.Nanobody.save
        @functools.wraps(original_save)
        def save(model, filename=None, *args, **kwargs):
            result = original_save(model, filename, *args, **kwargs)
            self.emit('model_saved', model_index=self.last_model,
                      refinement_success=self.last_success,
                      pdb_sha256=digest_file(filename or 'NanoBodyBuilder2_output.pdb'))
            self.pdb_geometry(filename or 'NanoBodyBuilder2_output.pdb', stage='final_model')
            return result
        self.hook(builder.Nanobody, 'save', save)
        self.emit('observer_started', scopes=list(SCOPES),
                  observer_sha256=digest_file(__file__), scientific_overrides=False)

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.emit('observer_finished', contexts=self.contexts,
                  failed_contexts=self.failed_contexts,
                  hooks_intact=all(getattr(o, n) is v for o, n, _, v in self.hooks))

    def restore(self):
        for owner, name, original, replacement in reversed(self.hooks):
            if getattr(owner, name) is replacement:
                setattr(owner, name, original)


def install(path=None):
    global _installed
    path = path or os.environ.get('NOVA_PR100_CONTEXT_TRACE')
    if not path or _installed is not None:
        return _installed
    observer = Observer(path)
    try:
        mm = importlib.import_module('openmm')
        importlib.import_module('openmm.app')
        observer.install(mm, importlib.import_module('pdbfixer'),
                         importlib.import_module('ImmuneBuilder.NanoBodyBuilder2'),
                         importlib.import_module('ImmuneBuilder.refine'),
                         importlib.import_module('torch'))
    except BaseException as exc:
        observer.emit('observer_error', error_type=type(exc).__name__)
        observer.restore()
        raise RuntimeError('reference_observer_initialization_failed') from None
    _installed = observer
    atexit.register(observer.finish)
    return observer
