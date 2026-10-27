from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import pytest
from nanobody_opt import pr100_psh_profile as profile


def modules():
    calls = []
    class Residue:
        def distance(self, a, b):
            calls.append((a, b))
            return 3.25
        def sb_distance(self, a, b):
            return 2.75
    class PDBchain:
        def __init__(self, *args):
            pass
    return SimpleNamespace(runPSA=lambda *a, **kw: 1, parsePSA=lambda *a: 2, parsePSAArea=lambda *a: 3), SimpleNamespace(Residue=Residue, PDBchain=PDBchain), calls


def test_results_directions_counts_and_restore():
    surface, geometry, calls = modules()
    original = geometry.Residue.distance
    a, b = geometry.Residue(), geometry.Residue()
    with profile.stage_timings(surface, geometry, timer=iter(range(100)).__next__) as report:
        assert a.distance(a, b) == b.distance(b, a) == 3.25
        assert surface.runPSA() == 1
    assert calls == [(a, b), (b, a)]
    assert report['all_atom_distances'] == {'calls': 2, 'wall_seconds': 2.0}
    assert report['distance_pair_counts'] == {'calls': 2, 'unordered_unique': 1, 'residue_objects': 2}
    assert geometry.Residue.distance is original


def test_exception_and_partial_setup_restore():
    surface, geometry, _ = modules()
    original = surface.runPSA
    with pytest.raises(KeyboardInterrupt):
        with profile.stage_timings(surface, geometry):
            raise KeyboardInterrupt()
    assert surface.runPSA is original
    del surface.parsePSAArea
    with pytest.raises(AttributeError):
        with profile.stage_timings(surface, geometry):
            pass
    assert surface.runPSA is original


def test_original_exception_propagates_and_is_timed():
    surface, geometry, _ = modules()
    error = RuntimeError('preserve original')
    def fail():
        raise error
    surface.runPSA = fail
    with profile.stage_timings(surface, geometry, timer=iter(range(100)).__next__) as report:
        with pytest.raises(RuntimeError) as caught:
            surface.runPSA()
    assert caught.value is error
    assert report['surface_executable'] == {'calls': 1, 'wall_seconds': 1.0}
    assert surface.runPSA is fail


def test_pin_mismatch_rejected(monkeypatch, tmp_path):
    path = tmp_path / 'changed.py'
    path.write_text('different implementation')
    monkeypatch.setattr(profile.importlib, 'import_module', lambda name: SimpleNamespace(__file__=str(path)))
    with pytest.raises(ValueError, match='pinned_source_mismatch'):
        profile.load_pinned_modules()


def test_cli_allowlisted_error_only():
    result = subprocess.run([sys.executable, profile.__file__, '--pdb', '/missing/private-input.pdb'], capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stderr == ''
    assert json.loads(result.stdout) == {'schema': 'pr100-tnp-fixed-pdb-stage-profile-v1', 'status': 'error', 'error_type': 'ValueError'}


@pytest.mark.parametrize('changed,nonfinite', [(False, False), (True, False), (False, True)])
def test_fixed_structure_parity_and_input_preservation(monkeypatch, tmp_path, changed, nonfinite):
    surface, geometry, _ = modules()
    invocations = []
    source = tmp_path / 'fixed.pdb'
    source.write_text('ATOM      1  CA  ALA H   1       0.000   0.000   0.000\n')
    original = source.read_bytes()
    def annotate(scale, ph, filename, chains, numbering, verbose):
        invocations.append(filename)
        assert (scale, ph, chains, numbering, verbose) == (0, 7.4, 'IG', 'imgt', False)
        assert filename != str(source)
        assert Path(filename).read_bytes() == original
        surface.runPSA(filename, 'internal.psa')
        geometry.PDBchain(filename, 'H')
        a, b = geometry.Residue(), geometry.Residue()
        a.distance(a, b)
        b.distance(b, a)
        Path(filename).with_name('input_Annotated.pdb').write_text('same artifact\n')
        metric = float('nan') if nonfinite else (90.0 if changed and len(invocations) == 2 else 88.9999)
        return {0: dict(zip(profile.METRIC_KEYS, (metric, 0.2, 0.3)))}
    surface.CreateAnnotation = annotate
    monkeypatch.setattr(profile, 'load_pinned_modules', lambda timer: (surface, geometry, profile.PINNED_HASHES, 0.25))
    if nonfinite:
        with pytest.raises(ValueError, match='nonfinite_metric'):
            profile.profile_fixed_structure(source, repeats=2)
    else:
        result = profile.profile_fixed_structure(source, repeats=2)
        assert result['all_metrics_identical'] is (not changed)
        assert result['all_annotations_identical'] is True
        assert len(invocations) == len(set(invocations)) == 4
        assert sorted(x['mode'] for x in result['measurements']) == ['baseline', 'baseline', 'profiled', 'profiled']
        assert all(x['stages']['distance_pair_counts']['calls'] == 2 for x in result['measurements'] if x['mode'] == 'profiled')
    assert source.read_bytes() == original
    assert not source.with_name('fixed_Annotated.pdb').exists()
    assert not any(Path(p).exists() for p in invocations)
