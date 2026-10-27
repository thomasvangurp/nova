"""Experimental, scoped acceleration of pinned TNP residue distances.

Scientific annotation/parser/PSA/salt-bridge code is left intact. This private-
process context changes only Residue.distance, whose inputs must remain immutable
for one CreateAnnotation invocation. It is not installed by any live entrypoint.
"""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import threading

from nanobody_opt.pr100_psh_profile import PINNED_HASHES

_LOCK = threading.Lock()
_SENTINEL = 99999999


def _source_hash(module):
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def _distance_table(residues, backend, max_atoms):
    """Original all-atom distance, calculated together with explicit x+y+z order.

    sqrt(min(squared distances)) equals min(sqrt(each squared distance)) for
    finite nonnegative inputs. The source sentinel is retained. Unsupported
    coordinates fail back to the original implementation in the context.
    CUDA is an unqualified optional experiment; CPU parity does not qualify it.
    """
    import numpy as np
    counts = [len(res.atoms) for res in residues]
    total = sum(counts)
    if not counts or not all(counts) or total > max_atoms:
        return None
    coords = np.asarray([atom.coords for res in residues for atom in res.atoms], dtype=np.float64)
    if coords.shape != (total, 3) or not np.isfinite(coords).all() or np.max(np.abs(coords), initial=0) > 1e150:
        return None
    if backend == 'numpy':
        dx = coords[:, None, 0] - coords[None, :, 0]
        squared = dx * dx
        del dx
        dy = coords[:, None, 1] - coords[None, :, 1]
        squared += dy * dy
        del dy
        dz = coords[:, None, 2] - coords[None, :, 2]
        squared += dz * dz
        del dz
    elif backend == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('cuda_not_available')
        with torch.no_grad():
            x = torch.tensor(coords, device='cuda', dtype=torch.float64)
            dx = x[:, None, 0] - x[None, :, 0]
            squared_gpu = dx * dx
            del dx
            dy = x[:, None, 1] - x[None, :, 1]
            squared_gpu += dy * dy
            del dy
            dz = x[:, None, 2] - x[None, :, 2]
            squared_gpu += dz * dz
            del dz
            squared = squared_gpu.cpu().numpy()
    else:
        raise ValueError('unsupported_distance_backend')
    starts = np.cumsum([0, *counts[:-1]], dtype=np.int64)
    minimum = np.minimum.reduceat(np.minimum.reduceat(squared, starts, axis=0), starts, axis=1)
    # Keep the original scalar sqrt implementation rather than substituting a
    # vector library sqrt. No matrix multiplication, TF32 or float32 is used.
    return [[min(_SENTINEL, math.sqrt(float(value))) for value in row] for row in minimum]


@contextmanager
def accelerated_distances(surface, geometry, *, backend='reference_cache', max_atoms=2048,
                          require_pinned=True):
    """Restore originals on every exit, with one cache/table per fixed structure.

    Caller owns a private single-threaded process. A process lock rejects nested
    or competing installations; it cannot make other uncooperative threads safe.
    'reference_cache' computes each unordered pair with the original method.
    'numpy'/'cuda' precompute all-atom distances from the original parsed PDB.
    """
    if backend not in ('reference_cache', 'numpy', 'cuda'):
        raise ValueError('unsupported_distance_backend')
    if not isinstance(max_atoms, int) or isinstance(max_atoms, bool) or max_atoms < 1:
        raise ValueError('invalid_atom_limit')
    if require_pinned and ({'surface': _source_hash(surface), 'geometry': _source_hash(geometry)} != PINNED_HASHES):
        raise ValueError('pinned_source_mismatch')
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('distance_accelerator_already_installed')
    try:
        original_distance = geometry.Residue.distance
        original_init = geometry.PDBchain.__init__
    except BaseException:
        _LOCK.release()
        raise
    pairs, objects, locations = {}, {}, {}
    stats = {'backend': backend, 'calls': 0, 'cache_hits': 0, 'reference_calls': 0,
             'tables_built': 0, 'table_fallbacks': 0, 'table_hits': 0, 'max_atoms': max_atoms}

    def parsed(chain, *args, **kwargs):
        original_init(chain, *args, **kwargs)
        residues = list(chain.residues.values())
        table = _distance_table(residues, backend, max_atoms)
        if table is None:
            stats['table_fallbacks'] += 1
            return
        stats['tables_built'] += 1
        for index, residue in enumerate(residues):
            objects[id(residue)] = residue
            locations[id(residue)] = (table, index)

    def distance(self, left, right):
        stats['calls'] += 1
        li, ri = id(left), id(right)
        objects[li], objects[ri] = left, right
        a, b = locations.get(li), locations.get(ri)
        if a is not None and b is not None and a[0] is b[0]:
            stats['table_hits'] += 1
            return a[0][a[1]][b[1]]
        key = (li, ri) if li <= ri else (ri, li)
        if key in pairs:
            stats['cache_hits'] += 1
            return pairs[key]
        result = original_distance(self, left, right)
        stats['reference_calls'] += 1
        pairs[key] = result
        return result

    try:
        geometry.Residue.distance = distance
        if backend != 'reference_cache':
            geometry.PDBchain.__init__ = parsed
        yield stats
    finally:
        geometry.Residue.distance = original_distance
        geometry.PDBchain.__init__ = original_init
        _LOCK.release()
