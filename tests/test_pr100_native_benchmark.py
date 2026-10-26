"""Control-flow tests only. No synthetic profile certifies native equivalence."""
import copy
import json
from pathlib import Path
import pytest
import sys
import subprocess

from nanobody_opt.pr100_native_benchmark import compare_results, digest, freeze_population, MODES
from scripts.pr100_native_probe import safe_profile, validate_inputs, verify_runtime

POPULATION = Path(__file__).resolve().parents[1] / 'docs/experiments/pr100-native-reuse-v1.population.json'


def population():
    return json.loads(POPULATION.read_text())


def profile():
    return dict(total_cdr_length=32, cdr3_length=16, cdr3_compactness=1.0,
                surface_hydrophobic_patches=88.0, positive_charge_patches=.2,
                negative_charge_patches=.3, aggrescan3d=dict(
                    status='ok', passed=True, a3d_average=-.8, a3d_maximum=1.2,
                    a3d_minimum=-1.0, a3d_total=-96.0, a3d_positive_auc=2.0,
                    residue_count=120, positive_residue_count=10, input_structure_sha256='a'*64,
                    normalized_structure_sha256='b'*64, sequence_hash='c'*64,
                    ph_scale='default-scale', surface_method='FreeSASA-2.0.1-NACCESS-LR100',
                    mode='static-no-minimization', radius_angstrom=10))


def test_frozen_real_population_covers_failures_lengths_thresholds_and_balances_order():
    p = population()
    assert p['population_sha256'] == digest(p['records'])
    assert len({r['sequence_hash'] for r in p['records']}) == 24
    assert {r['reason'] for r in p['records']} == {
        'passed', 'psh_threshold', 'developability_red_flags', 'aggrescan_unavailable_or_failed'}
    assert {int(c.split(':')[1]) for r in p['records'] for c in r['cells']} == {5, 16, 18}
    assert all(sum(r['mode_order'][0] == m for r in p['records']) == 8 for m in MODES)
    assert all(any(tag.startswith('near_'+field) for r in p['records'] for tag in r['selection_tags'])
               for field in ('psh', 'a3d_average', 'a3d_maximum'))
    assert not p['cache_reads'] and not p['publication_enabled']


def test_research_artifacts_and_regressions_are_mandatory_shipping_inputs():
    from scripts.shipping_checks import TEST_FILES
    from scripts.release_manifest import SOURCE_PATHS, RESEARCH_FILES
    assert 'tests/test_pr100_native_benchmark.py' in TEST_FILES
    for name in ('docs/PR100_DEVELOPABILITY_EXPERIMENT.md', 'docs/PR100_NATIVE_CLOSURE_CHECK.md',
                 'docs/experiments/pr100-native-reuse-v1.population.json'):
        assert name in SOURCE_PATHS and name in RESEARCH_FILES


def test_population_freeze_is_order_invariant_and_rejects_overlapping_sources():
    p = population()
    rows = [dict(r, buckets=r['cells']) for r in p['records']]
    a = freeze_population({'fixtures': {'candidate_measurements': rows}})
    b = freeze_population({'fixtures': {'candidate_measurements': list(reversed(rows))}})
    assert a['records'] == b['records']
    with pytest.raises(ValueError, match='overlapping'):
        freeze_population({'a': {'candidate_measurements': rows}, 'b': {'candidate_measurements': rows}})


def test_missing_failed_and_repeated_observations_cannot_look_like_parity():
    p = population()
    key = p['records'][0]['sequence_hash']
    report = compare_results(p, [{'sequence_hash': key, 'mode': 'warm', 'status': 'failed'}])
    assert report['missing'] == 71 and report['complete_pairs'] == 0
    assert report['failure_counts'] == {'failed': 1}
    assert report['decision'] == 'NO_GO_INCOMPLETE'
    assert not report['qualification'] and not report['native_equivalence_claim']
    with pytest.raises(ValueError, match='repeated'):
        compare_results(p, [{'sequence_hash': key, 'mode': 'warm', 'status': 'failed'}]*2)


def test_cli_noise_does_not_hide_warm_gate_flip_or_numeric_delta():
    p = population()
    key = p['records'][0]['sequence_hash']
    rows = [dict(sequence_hash=key, mode=mode, status='ok', profile=profile(),
                 eligible=mode != 'cli_a', reason='passed' if mode != 'cli_a' else 'psh_threshold',
                 wall_seconds=10, cpu_seconds=8) for mode in MODES]
    rows[1]['profile']['surface_hydrophobic_patches'] = 88.2
    rows[2]['profile']['surface_hydrophobic_patches'] = 88.5
    report = compare_results(p, rows)
    assert report['gate_flips'] == 2  # includes both CLI repeat and warm-vs-CLI
    assert report['pairs'][1]['numeric_deltas']['surface_hydrophobic_patches'] == .5
    assert report['complete_pairs'] == 3 and report['missing'] == 69


def test_complete_execution_has_an_explicit_parity_and_cost_verdict():
    p = population()
    rows = [dict(sequence_hash=r['sequence_hash'], mode=mode, status='ok', profile=profile(),
                 eligible=True, reason='passed', wall_seconds=8 if mode == 'warm' else 10,
                 cpu_seconds=8) for r in p['records'] for mode in MODES]
    good = compare_results(p, rows)
    assert good['decision'] == 'GO_TO_LARGER_EXPERIMENT_ONLY'
    assert not good['qualification'] and not good['native_equivalence_claim']
    rows[0]['eligible'] = False
    assert compare_results(p, rows)['decision'] == 'NO_GO_PARITY'
    rows[0]['eligible'] = True
    rows[0]['profile']['aggrescan3d']['a3d_total'] = -96.1
    assert compare_results(p, rows)['decision'] == 'NO_GO_PARITY'
    rows[0]['profile']['aggrescan3d']['a3d_total'] = -96.0
    for r in rows:
        r['wall_seconds'] = 10
    assert compare_results(p, rows)['decision'] == 'NO_GO_PERFORMANCE'
    rows[0].update(status='failed')
    assert compare_results(p, rows)['decision'] == 'NO_GO_INCOMPLETE'


def test_safe_profile_drops_backend_text_and_rejects_nonfinite_or_wrong_provenance():
    p = profile()
    p['debug'] = 'untrusted backend text'
    p['aggrescan3d']['raw_stdout'] = 'untrusted backend text'
    clean = safe_profile(p)
    assert 'debug' not in clean and 'raw_stdout' not in clean['aggrescan3d']
    for field, value in [('cdr3_compactness', float('nan')), ('cdr3_length', True)]:
        bad = copy.deepcopy(p)
        bad[field] = value
        with pytest.raises(ValueError, match='nonfinite'):
            safe_profile(bad)
    p['aggrescan3d']['mode'] = 'unqualified'
    with pytest.raises(ValueError, match='provenance'):
        safe_profile(p)


def test_no_runtime_execution_from_an_unreviewed_file_list():
    with pytest.raises(ValueError, match='reviewed'):
        verify_runtime({'schema': 'pr100-native-runtime-manifest-v1', 'closure_reviewed': False})
    with pytest.raises(ValueError, match='missing inputs'):
        validate_inputs(population(), [])


def test_linux_containment_rehearsal_or_explicit_platform_refusal():
    from scripts.pr100_native_probe import containment_rehearsal
    if sys.platform != 'linux':
        with pytest.raises(ValueError, match='Linux'):
            containment_rehearsal()
    else:
        # Give the rehearsal its own ancestry: never stop pytest's other children.
        script = Path(__file__).resolve().parents[1] / 'scripts/pr100_native_probe.py'
        process = subprocess.run([sys.executable, str(script), '--self-test-containment'],
                                 capture_output=True, text=True, timeout=12)
        assert process.returncode == 0
        result = json.loads(process.stdout)
        assert result['passed'] and result['remaining_descendants'] == 0


def test_structure_callback_runs_before_cleanup_and_legacy_profile_unchanged(tmp_path, monkeypatch):
    # Injected model/metrics boundary, explicitly not a native test.
    from nanobody_opt import tnp_daemon
    class Builder:
        def predict(self, sequences):
            assert len(sequences) == 1
            return self

        def save(self, path, **kwargs):
            assert kwargs == {'check_for_strained_bonds': True, 'n_threads': 1}
            Path(path).write_text('structure fixture')

    profiler = tnp_daemon._TNPProfiler.__new__(tnp_daemon._TNPProfiler)
    profiler.builder = Builder()
    profiler._tmp_root = str(tmp_path)
    profiler._region = lambda *a, **k: (None, {'imgt': dict(cdrh1=8, cdrh2=8, cdrh3=16)})
    profiler._compactness = lambda *a, **k: 16
    profiler._hydrophob = lambda *a, **k: [dict(Patch_Hydrophob_CDR=88, Patch_Pos_Charge_CDR=.2, Patch_Neg_Charge_CDR=.3)]
    monkeypatch.setattr(tnp_daemon, '_strip_hydrogens', lambda _: None)
    captured = []
    def score(sequence, path):
        assert Path(path).exists()
        captured.append(Path(path))
        return {'status': 'fixture'}
    profiler._structure_scorer = score
    augmented = profiler.profile_one('A'*120, 1)
    assert augmented['aggrescan3d'] == {'status': 'fixture'}
    assert not captured[0].exists()
    profiler._structure_scorer = None
    legacy = profiler.profile_one('A'*120, 1)
    augmented.pop('aggrescan3d')
    assert augmented == legacy
