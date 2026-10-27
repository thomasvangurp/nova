"""Benchmark pinned TNP annotation on fixed PDBs; no scientific code is changed.

Run in a private process with an external deadline. Instrumentation temporarily
wraps module/class attributes and must not share a process with live work.
"""
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import tempfile
import time

PINNED_REVISION = '29dcac72f1380e8538e8870f45a699d3c6156162'
PINNED_HASHES = {
    'surface': '71c4fcd4fbe3aa2892598a54dfb1d208265a31f30f455efc8a9c85ea10af7c46',
    'geometry': 'e626d60701861b137b8c76c01cd10e69c3d317fb46eedf44e388d0df92c9f8bd',
}
PREFIX = 'theraprofnano.Hydrophobicity_and_Charge_Profiler.'
METRIC_KEYS = ('Patch_Hydrophob_CDR', 'Patch_Pos_Charge_CDR', 'Patch_Neg_Charge_CDR')


@contextmanager
def quiet_native_output():
    """Silence both Python and inherited subprocess stdout/stderr."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with open(os.devnull, 'w') as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        for fd, previous in zip((1, 2), saved):
            os.dup2(previous, fd)
            os.close(previous)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SourceIdentityMismatch(ValueError):
    def __init__(self, observed):
        super().__init__('pinned_source_mismatch')
        self.observed = observed


def load_pinned_modules(timer=time.perf_counter):
    started = timer()
    surface = importlib.import_module(PREFIX + 'Hydrophobicity_and_Charge_Assigner')
    geometry = importlib.import_module(PREFIX + 'Common.PDBUtils')
    elapsed = timer() - started
    hashes = {'surface': file_hash(surface.__file__), 'geometry': file_hash(geometry.__file__)}
    if hashes != PINNED_HASHES:
        raise SourceIdentityMismatch(hashes)
    return surface, geometry, hashes, elapsed


@contextmanager
def stage_timings(surface, geometry, *, timer=time.perf_counter):
    """Time disjoint expensive calls, restoring originals even after an error."""
    timings, patches, pairs, objects = {}, [], set(), {}

    def instrument(owner, attribute, label, track_pairs=False):
        original = getattr(owner, attribute)
        entry = timings.setdefault(label, {'calls': 0, 'wall_seconds': 0.0})

        def wrapped(*args, **kwargs):
            entry['calls'] += 1
            if track_pairs:
                left, right = args[1:3]
                objects[id(left)], objects[id(right)] = left, right
                pairs.add(tuple(sorted((id(left), id(right)))))
            started = timer()
            try:
                return original(*args, **kwargs)
            finally:
                entry['wall_seconds'] += timer() - started

        patches.append((owner, attribute, original))
        setattr(owner, attribute, wrapped)

    try:
        for attribute, label in (
            ('runPSA', 'surface_executable'),
            ('parsePSA', 'surface_parse_relative'),
            ('parsePSAArea', 'surface_parse_area'),
        ):
            instrument(surface, attribute, label)
        instrument(geometry.PDBchain, '__init__', 'pdb_parse')
        instrument(geometry.Residue, 'distance', 'all_atom_distances', True)
        instrument(geometry.Residue, 'sb_distance', 'salt_bridge_distances')
        yield timings
    finally:
        timings['distance_pair_counts'] = {
            'unordered_unique': len(pairs), 'residue_objects': len(objects),
            'calls': timings.get('all_atom_distances', {}).get('calls', 0),
        }
        for owner, attribute, original in reversed(patches):
            setattr(owner, attribute, original)


def profile_fixed_structure(pdb, *, repeats=3, scale=0, timer=time.perf_counter):
    if not 1 <= repeats <= 20 or scale not in range(5):
        raise ValueError('invalid_benchmark_parameters')
    pdb = Path(pdb)
    if pdb.suffix.lower() != '.pdb' or not pdb.is_file():
        raise ValueError('expected_fixed_pdb')
    with quiet_native_output():
        surface, geometry, hashes, imports = load_pinned_modules(timer)
        pdb_hash = file_hash(pdb)
        modes = [mode for _ in range(repeats) for mode in ('baseline', 'profiled')]
        random.Random(0).shuffle(modes)
        measurements = []
        for mode in modes:
            with tempfile.TemporaryDirectory(prefix='tnp-stage-profile-') as directory:
                copied = Path(directory) / 'input.pdb'
                shutil.copyfile(pdb, copied)
                if mode == 'profiled':
                    with stage_timings(surface, geometry, timer=timer) as stages:
                        started = timer()
                        value = surface.CreateAnnotation(scale, 7.4, str(copied), 'IG', 'imgt', verbose=False)
                        elapsed = timer() - started
                else:
                    stages = {}
                    started = timer()
                    value = surface.CreateAnnotation(scale, 7.4, str(copied), 'IG', 'imgt', verbose=False)
                    elapsed = timer() - started
                metrics = {key: float(value[scale][key]) for key in METRIC_KEYS}
                if not all(math.isfinite(x) for x in metrics.values()):
                    raise ValueError('nonfinite_metric')
                measured = sum(x.get('wall_seconds', 0.0) for x in stages.values())
                measurements.append({
                    'mode': mode, 'annotation_wall_seconds': elapsed,
                    'instrumented_call_seconds': measured,
                    'unattributed_seconds_including_instrumentation': elapsed - measured,
                    'negative_attribution_residual': elapsed < measured,
                    'stages': stages, 'metrics': metrics,
                    'annotated_pdb_sha256': file_hash(copied.with_name('input_Annotated.pdb')),
                })
    baseline = statistics.median(x['annotation_wall_seconds'] for x in measurements if x['mode'] == 'baseline')
    profiled = statistics.median(x['annotation_wall_seconds'] for x in measurements if x['mode'] == 'profiled')
    reference = measurements[0]
    return {
        'schema': 'pr100-tnp-fixed-pdb-stage-profile-v1', 'status': 'complete',
        'tnp_revision': PINNED_REVISION, 'source_hashes': hashes,
        'input_pdb_sha256': pdb_hash, 'import_wall_seconds': imports,
        'baseline_annotation_median_seconds': baseline,
        'profiled_annotation_median_seconds': profiled,
        'instrumentation_wall_ratio': profiled / baseline if baseline else None,
        'all_metrics_identical': all(x['metrics'] == reference['metrics'] for x in measurements),
        'all_annotations_identical': all(x['annotated_pdb_sha256'] == reference['annotated_pdb_sha256'] for x in measurements),
        'measurements': measurements,
        'limitations': [
            'fixed_structure_only_no_prediction_or_refinement',
            'annotation_includes_psh_ppc_pnc_not_psh_alone',
            'instrumentation_has_overhead_compare_separate_baseline',
            'process_startup_excluded_import_timing_may_include_cached_modules',
            'caller_must_enforce_process_deadline',
        ],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdb', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--hydrophobicity-scale', type=int, default=0)
    args = parser.parse_args()
    try:
        result = profile_fixed_structure(args.pdb, repeats=args.repeats, scale=args.hydrophobicity_scale)
    except Exception as exc:
        result = {'schema': 'pr100-tnp-fixed-pdb-stage-profile-v1', 'status': 'error', 'error_type': type(exc).__name__}
        if isinstance(exc, SourceIdentityMismatch):
            result['observed_source_hashes'] = exc.observed
            result['expected_source_hashes'] = PINNED_HASHES
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
