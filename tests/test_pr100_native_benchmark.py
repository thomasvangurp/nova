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
    return dict(passed=True, red_flags=[], total_cdr_length=32, cdr3_length=16, cdr3_compactness=1.0,
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
    assert 'scripts/pr100_runtime_preflight.py' in RESEARCH_FILES


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
        process = subprocess.run([sys.executable, '-B', '-S', str(script), '--self-test-containment'], start_new_session=True,
                                 capture_output=True, text=True, timeout=12)
        assert process.returncode == 0
        result = json.loads(process.stdout)
        assert result['passed'] and result['remaining_descendants'] == 0


def test_type_reason_and_subgate_mismatches_are_not_hidden_by_equal_deltas():
    p = population()
    key = p['records'][0]['sequence_hash']
    rows = [dict(sequence_hash=key, mode=m, status='ok', profile=profile(), eligible=True,
                 reason='passed', wall_seconds=10, cpu_seconds=8) for m in MODES]
    rows[1]['profile']['cdr3_length'] = 16.0
    rows[2]['profile']['aggrescan3d']['passed'] = False
    rows[2]['reason'] = 'aggrescan_unavailable_or_failed'
    result = compare_results(p, rows)
    assert result['numeric_mismatches'] == 2
    assert result['reason_mismatches'] == 2 and result['gate_flips'] == 2
    assert all(p['numeric_deltas']['cdr3_length'] == 0 for p in result['pairs'] if p['complete'])


def test_fixed_point_cleanup_and_durable_preflight_receipt(tmp_path, monkeypatch):
    from scripts import pr100_native_probe as probe
    states = iter([{1001}, {1001, 1002}, {1001, 1002}, {1001, 1002}, {1001, 1002}, set()])
    monkeypatch.setattr(probe, 'descendants', lambda: next(states))
    signals = []
    monkeypatch.setattr(probe.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    def no_children(*_):
        raise ChildProcessError
    monkeypatch.setattr(probe.os, 'waitpid', no_children)
    cleanup = probe.stop_descendants(timeout=2)
    assert cleanup['fixed_point_reached'] and cleanup['freeze_scan_count'] == 4
    assert cleanup['remaining_descendants'] == 0
    def refuse_controller():
        raise ValueError('unrelated children')
    monkeypatch.setattr(probe, 'dedicated_controller', refuse_controller)
    out = tmp_path / 'receipt'
    result = probe.run(population(), {}, {}, out)
    assert result['decision'] == 'NO_GO_EXECUTION'
    assert 'preflight_error' in result['conditions']
    assert json.loads((out / 'report.json').read_text()) == result


def test_atomic_event_results_survive_and_initialization_diagnostics_keep_primary(tmp_path):
    from scripts.pr100_native_probe import FileEvents
    from nanobody_opt.pr100_native_benchmark import timing_diagnostics
    (tmp_path / 'results').mkdir()
    sink = FileEvents(tmp_path)
    sink.put(dict(event='begin', worker=0, sequence_hash='a'*64, mode='warm', started=1))
    sink.put(dict(event='result', worker=0, sequence_hash='a'*64, mode='warm', status='failed'))
    assert not (tmp_path / 'active.json').exists()
    assert json.loads(next((tmp_path / 'results').iterdir()).read_text())['status'] == 'failed'
    rows = [dict(sequence_hash=str(i), mode=m, wall_seconds=10, cpu_seconds=8,
                 mode_position=1, initialization_seconds=3 if m == 'warm' and i == 0 else 0)
            for i in range(2) for m in MODES]
    report = timing_diagnostics(rows)
    assert report['primary_includes_all_initialization']
    assert report['by_mode']['warm']['summed_profile_wall_seconds'] == 20
    assert report['without_initialization_diagnostic_only']['warm']['profiles'] == 1


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


def test_runtime_deadlines_and_bytecode_disable_are_enforced(tmp_path, monkeypatch):
    from scripts import pr100_native_probe as probe
    from scripts.release_manifest import digest as file_digest
    source = tmp_path/'public.py'; source.write_text('pass\n')
    with pytest.raises(TimeoutError):
        file_digest(source, deadline=0)
    monkeypatch.setattr(probe.sys, 'dont_write_bytecode', False)
    with pytest.raises(ValueError, match='-B'):
        probe.assert_bytecode_disabled()
    monkeypatch.setattr(probe.sys, 'dont_write_bytecode', True)
    probe.assert_bytecode_disabled()
    runtime = {'n0_measurements': {'verification': {'seconds': 20}, 'first_loaded_audit_seconds': 10}}
    assert probe.verification_budget(runtime)['projected_total_seconds'] == 1170
    runtime['n0_measurements']['verification']['seconds'] = 40
    with pytest.raises(ValueError, match='does not fit'):
        probe.verification_budget(runtime)


def test_mapping_exceptions_are_exact_nonexecutable_data_only():
    from scripts.pr100_native_probe import mapping_classification
    policy = [{'path': '/dev/shm/registered-data (deleted)', 'permissions': 'rw-s', 'kind': 'non_executable_data'}]
    assert mapping_classification(policy[0]['path'], 'rw-s', policy) == 'registered_non_executable_data'
    assert mapping_classification(policy[0]['path'], 'r-xs', policy) is None
    assert mapping_classification('/memfd:unknown (deleted)', 'rw-s', policy) is None
    assert mapping_classification('/tmp/unknown-code.so (deleted)', 'r-xs', []) is None


def test_n0_import_guard_and_plugin_categories_are_narrow():
    from scripts.pr100_runtime_preflight import forbidden_call, plugin_categories
    assert forbidden_call('ImmuneBuilder.NanoBodyBuilder2', '__init__', 'NanoBodyBuilder2')
    assert forbidden_call('openmm.openmm', '__init__', 'Context')
    assert forbidden_call('torch.cuda', 'is_available')
    assert forbidden_call('theraprofnano.CDR_Profiler.CDR_Assigner', 'main')
    assert forbidden_call('openmm.openmm', 'getPlatformByName', 'Platform') is None
    assert plugin_categories(['libOpenMMHIP.so: libamdhip64.so.6 unavailable'])[0]['scope'] == 'unused_CPU_only'
    with pytest.raises(ValueError, match='unclassified'):
        plugin_categories(['libOpenMMCPU.so failed'])


def test_n0_failed_preflight_writes_durable_no_go(tmp_path, monkeypatch):
    from scripts import pr100_runtime_preflight as n0
    def refuse():
        raise ValueError('control fixture refusal')
    monkeypatch.setattr(n0.probe, 'assert_bytecode_disabled', refuse)
    out = tmp_path/'n0'
    summary = n0.check({}, out)
    report = json.loads((out/'receipt.json').read_text())
    assert summary['status'] == report['status'] == 'NO_GO'
    assert report['native_profiles_permitted'] == report['gpu_calls_permitted'] == 0
    assert report['closure_reviewed'] is False
    assert not (out/'runtime-candidate.json').exists()


def test_forcefield_exception_binds_caller_argument_and_observed_counts():
    from scripts.pr100_runtime_preflight import FORCEFIELD_IMPORT as r, ForceFieldImportGuard, forbidden_call
    exact = (r['caller_module'], r['caller_function'], r['caller_path'], r['caller_line'],
             r['constructor_path'], (r['xml_argument'],))
    guard = ForceFieldImportGuard()
    guard.enter(*exact)
    guard.xml_open(r['xml_path'], 'r', 0)
    guard.active = False
    assert guard.finish()['constructor_count'] == guard.finish()['xml_open_count'] == 1
    with pytest.raises(ValueError):
        guard.enter(*exact)
    for index, replacement in [(0, 'another.module'), (1, 'refine_once'), (2, '/tmp/refine.py'),
                               (3, r['caller_line']+1), (4, '/tmp/forcefield.py'), (5, ('other.xml',))]:
        args = list(exact); args[index] = replacement
        with pytest.raises(ValueError):
            ForceFieldImportGuard().enter(*args)
    for path, mode, flags in [('/tmp/protein.ff14SB.xml', 'r', 0), (r['xml_path'], 'w', 1)]:
        guard = ForceFieldImportGuard(); guard.enter(*exact)
        with pytest.raises(ValueError):
            guard.xml_open(path, mode, flags)
    missing = ForceFieldImportGuard(); missing.enter(*exact); missing.active = False
    with pytest.raises(ValueError, match='incomplete'):
        missing.finish()
    assert forbidden_call('openmm.app.forcefield', 'createSystem') == 'physics_execution'
    assert forbidden_call('openmm.openmm', '__init__', 'Context') == 'physics_constructor'


def loopback_fixture(monkeypatch):
    from scripts import pr100_runtime_preflight as n0
    class Socket:
        family=n0.socket.AF_INET6
        type=n0.socket.SOCK_STREAM
        proto=0
        fd=9
        def fileno(self):return self.fd
    monkeypatch.setattr(n0.socket, 'socket', Socket)
    r=n0.LOOPBACK_IMPORT
    args=[Socket(), ('::1',0), r['declared_import'],
          (r['caller_module'],r['caller_function'],r['caller_path'],r['caller_line']),
          (r['caller_module'],r['parent_function'],r['caller_path'],r['parent_line'])]
    return n0, args


def test_loopback_exception_requires_one_bind_and_closed_retained_socket(monkeypatch):
    n0,args=loopback_fixture(monkeypatch)
    guard=n0.LoopbackImportGuard()
    with pytest.raises(ValueError,match='incomplete'):guard.finish()
    guard.bind(*args)
    with pytest.raises(ValueError,match='incomplete'):guard.finish()
    with pytest.raises(ValueError,match='unregistered'):guard.bind(*args)
    args[0].fd=-1
    assert guard.finish()=={'bind_count':1,'socket_closed':True,
        'scope':'registered_IPv6_loopback_capability_probe_only','external_network_operations_permitted':0}


@pytest.mark.parametrize('failure', ['address', 'port', 'bool_port', 'list', 'family', 'type', 'protocol',
    'closed', 'object', 'declared_import', 'caller_module', 'caller_function', 'caller_path',
    'caller_line', 'parent_module', 'parent_function', 'parent_path', 'parent_line'])
def test_loopback_exception_rejects_every_unregistered_site_or_socket(monkeypatch,failure):
    n0,args=loopback_fixture(monkeypatch)
    if failure in {'address','port','bool_port','list'}:
        args[1]={'address':('::',0),'port':('::1',1),'bool_port':('::1',False),'list':['::1',0]}[failure]
    elif failure in {'family','type','protocol','closed'}:
        name,value={'family':('family',n0.socket.AF_INET),'type':('type',n0.socket.SOCK_DGRAM),
                    'protocol':('proto',6),'closed':('fd',-1)}[failure]
        setattr(args[0],name,value)
    elif failure=='object':args[0]=object()
    elif failure=='declared_import':args[2]='other'
    else:
        prefix,field=failure.split('_',1)
        at=3 if prefix=='caller' else 4
        fields=list(args[at]);fields[{'module':0,'function':1,'path':2,'line':3}[field]]='unregistered'
        args[at]=tuple(fields)
    with pytest.raises(ValueError,match='unregistered'):n0.LoopbackImportGuard().bind(*args)


def test_loopback_registration_requires_exact_frozen_source_and_policy(monkeypatch):
    from scripts import pr100_runtime_preflight as n0
    r=n0.LOOPBACK_IMPORT
    runtime={'n0_loopback_import':r,'files':{r['caller_path']:r['caller_sha256']}}
    monkeypatch.setattr(n0.probe,'sha_file',lambda _:r['caller_sha256'])
    n0.validate_loopback_registration(runtime)
    with pytest.raises(ValueError):n0.validate_loopback_registration(dict(runtime,n0_loopback_import={}))
    with pytest.raises(ValueError):n0.validate_loopback_registration(dict(runtime,files={}))
    monkeypatch.setattr(n0.probe,'sha_file',lambda _:'0'*64)
    with pytest.raises(ValueError):n0.validate_loopback_registration(runtime)


def test_research_helpers_do_not_shadow_regular_scientific_scripts_package(tmp_path):
    # Reproduce the actual native package collision in a fresh interpreter.
    package = tmp_path/'scripts'; package.mkdir()
    (package/'__init__.py').write_text('SCIENTIFIC = True\n')
    (package/'region_definitions.py').write_text('REGIONS = True\n')
    root = Path(__file__).resolve().parents[1]
    code = '''import importlib.util,json,sys
from pathlib import Path
root,site=map(Path,sys.argv[1:])
sys.path[:0]=[str(root),str(site)]
import scripts
assert scripts.SCIENTIFIC
spec=importlib.util.spec_from_file_location('_n0_collision_test',root/'scripts/pr100_runtime_preflight.py')
n0=importlib.util.module_from_spec(spec);spec.loader.exec_module(n0)
try:
 n0.probe.verify_runtime({'schema':'pr100-native-runtime-manifest-v1','source_stage_manifest_sha256':'0'*64},preflight_only=True)
except (FileNotFoundError,ValueError):pass
assert '_pr100_release_manifest' in sys.modules
import scripts.region_definitions
assert scripts.region_definitions.REGIONS and scripts.SCIENTIFIC
print(json.dumps({'research_helper_loaded':True,'scientific_package_preserved':True}))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(root), str(tmp_path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {'research_helper_loaded': True, 'scientific_package_preserved': True}


def test_n0_environment_failure_has_a_specific_safe_stage(tmp_path):
    from scripts import pr100_runtime_preflight as n0
    spec = tmp_path/'spec.json'
    spec.write_text(json.dumps({'private':str(tmp_path),'runtime':{'path':'/unregistered', 'library_path':''}}))
    assert n0.import_child(spec) == 1
    receipt = json.loads((tmp_path/'child.json').read_text())
    assert receipt['stage'] == 'environment' and receipt['error_class'] == 'ValueError'
    assert set(receipt) == {'status', 'stage', 'error_class'}


def test_unresolved_origin_paths_are_actionable_only_inside_public_roots(tmp_path):
    from scripts.pr100_native_probe import origin_diagnostic
    root = tmp_path/'public'
    inside = origin_diagnostic(str(root/'module.so'), [str(root)])
    assert inside['public_origin_path'] == str(root/'module.so')
    for path in (str(tmp_path/'private/secret.json'),str(root/'.env'),str(root/'credentials'),'/memfd:private (deleted)'):
        row = origin_diagnostic(path, [str(root)])
        assert 'public_origin_path' not in row and len(row['origin_locator_sha256']) == 64
    assert origin_diagnostic('/memfd:private (deleted)', [])['mapping_classes'] == ['deleted','memfd']


def test_staged_namespace_and_filter_origins_reject_prior_qualification_source():
    from types import SimpleNamespace
    from scripts.pr100_native_probe import assert_stage_origins, ROOT
    modules = {'metanano':SimpleNamespace(__file__=str(ROOT/'NOVA-nanobody-filter/metanano/__init__.py')),
               'nanobody_opt':SimpleNamespace(__file__=None,__path__=[str(ROOT/'nanobody_opt')])}
    assert set(assert_stage_origins(modules)) == set(modules)
    modules['metanano.filters.developability'] = SimpleNamespace(__file__='/root/pr100-qualification-20261026/NOVA-nanobody-filter/metanano/filters/developability.py')
    with pytest.raises(ValueError, match='outside staged'):
        assert_stage_origins(modules)


def test_n1_verifies_receipt_bytes_status_measurements_and_runtime_binding(tmp_path):
    from scripts import pr100_native_probe as probe
    base = {'closure_reviewed':False, 'marker':'frozen-runtime'}
    receipt = dict(status='complete_for_independent_review', conditions=[], child_exitcode=0,
        runtime_input_sha256=digest(base), harness_sha256=probe.sha_file(probe.ROOT/'scripts/pr100_runtime_preflight.py'),
        verification={'seconds':2,'hashed_bytes':10}, post_verification={'seconds':1,'hashed_bytes':10},
        cleanup={'remaining_descendants':0,'fixed_point_reached':True},
        child={'status':'complete','cpu_platform':'CPU','cpu_threads_property':True,'declared_imports':list(probe.N0_IMPORTS),
               'first_loaded_audit_seconds':1,'loaded_identity':{'unregistered_origins':[],'registered_data_mappings':[]},
               'staged_module_origins':{'fixture':['registered-source']},
               'forcefield_import':{'constructor_count':1,'xml_open_count':1},
               'loopback_import':{'bind_count':1,'socket_closed':True,'external_network_operations_permitted':0}})
    (tmp_path/'n0').mkdir()
    path = tmp_path/'n0/receipt.json';path.write_text(json.dumps(receipt))
    outer={'receipt_sha256':probe.sha_file(path),'child_exit_code':0,
           'outer_cleanup':{'remaining_descendants':0,'fixed_point_reached':True}}
    (tmp_path/'supervisor-receipt.json').write_text(json.dumps(outer))
    (tmp_path/'containment.json').write_text(json.dumps({'passed':True,'remaining_descendants':0}))
    runtime = dict(base, closure_reviewed=True, closure_reviewer='test-fixture-reviewer',
        n0_receipt_path=str(path),n0_receipt_sha256=probe.sha_file(path),
        n0_measurements=probe.n0_measurements_from_receipt(receipt),
        n0_supervisor_receipt_sha256=probe.sha_file(tmp_path/'supervisor-receipt.json'),
        n0_containment_sha256=probe.sha_file(tmp_path/'containment.json'))
    assert probe.verify_n0_receipt(runtime)['status'] == receipt['status']
    for key in ('n0_supervisor_receipt_sha256','n0_containment_sha256'):
        with pytest.raises(ValueError,match='outer receipt identity'):
            probe.verify_n0_receipt(dict(runtime,**{key:'0'*64}))
    for failure in ('timeout','exit','descendants','fixedpoint','receipt'):
        bad_outer=copy.deepcopy(outer)
        if failure=='timeout':bad_outer['supervisor_timeout']=True
        elif failure=='exit':bad_outer['child_exit_code']=1
        elif failure=='descendants':bad_outer['outer_cleanup']['remaining_descendants']=1
        elif failure=='fixedpoint':bad_outer['outer_cleanup']['fixed_point_reached']=False
        else:bad_outer['receipt_sha256']='0'*64
        (tmp_path/'supervisor-receipt.json').write_text(json.dumps(bad_outer))
        with pytest.raises(ValueError,match='clean containment'):
            probe.verify_n0_receipt(dict(runtime,n0_supervisor_receipt_sha256=probe.sha_file(tmp_path/'supervisor-receipt.json')))
    (tmp_path/'supervisor-receipt.json').write_text(json.dumps(outer))
    bad = dict(runtime,n0_measurements={'verification':{'seconds':.1},'first_loaded_audit_seconds':.1})
    with pytest.raises(ValueError, match='timing evidence'):
        probe.verify_n0_receipt(bad)
    with pytest.raises(ValueError, match='another runtime'):
        probe.verify_n0_receipt(dict(runtime,marker='changed'))
    with pytest.raises(ValueError, match='digest'):
        probe.verify_n0_receipt(dict(runtime,n0_receipt_sha256='not-a-digest'))
    receipt['status']='NO_GO';path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='identity mismatch'):
        probe.verify_n0_receipt(runtime)
    runtime['n0_receipt_sha256']=probe.sha_file(path)
    with pytest.raises(ValueError, match='not complete'):
        probe.verify_n0_receipt(runtime)


def test_n0_wall_revision_is_explicit_and_older_contract_rejected():
    from scripts.pr100_runtime_preflight import N0_WALL_SECONDS, validate_forcefield_registration, FORCEFIELD_IMPORT
    assert N0_WALL_SECONDS == 120
    with pytest.raises(ValueError, match='unregistered'):
        validate_forcefield_registration({'n0_protocol_revision':4,'n0_wall_seconds':60,'n0_forcefield_import':FORCEFIELD_IMPORT})


@pytest.mark.parametrize('failure, expected', [('crash','child_receipt_missing'),('rss','sampled_rss_abort')])
def test_n0_child_crash_and_resource_abort_are_distinct(tmp_path,monkeypatch,failure,expected):
    # Controller boundary fixtures; no actual subprocess/native calls.
    from scripts import pr100_runtime_preflight as n0
    monkeypatch.setattr(n0.probe,'assert_bytecode_disabled',lambda:None)
    monkeypatch.setattr(n0.sys,'platform','linux')
    monkeypatch.setattr(n0.probe,'dedicated_controller',lambda:None)
    monkeypatch.setattr(n0.os,'sched_getaffinity',lambda _:set(range(4)),raising=False)
    monkeypatch.setattr(n0.os,'sched_setaffinity',lambda *_:None,raising=False)
    monkeypatch.setattr(n0.probe,'available_memory',lambda:{'cgroup_resolution':'resolved_visible_ancestors','available_bytes':200*1024**3})
    monkeypatch.setattr(n0.probe,'verify_runtime',lambda *a,**k:{'seconds':1,'hashed_bytes':10})
    monkeypatch.setattr(n0.probe,'stop_descendants',lambda *a,**k:{'remaining_descendants':0,'fixed_point_reached':True})
    monkeypatch.setattr(n0.probe,'descendants',lambda:set())
    monkeypatch.setattr(n0.probe,'process_rss',lambda _:5*1024**3)
    class Child:
        returncode=-11 if failure=='crash' else None
        def poll(self):return self.returncode
    monkeypatch.setattr(n0.subprocess,'Popen',lambda *a,**k:Child())
    out=tmp_path/'n0'
    runtime={'source_stage_root':str(n0.ROOT),'path':'/registered','entrypoints':{'logging_shim':'/registered/shim.py'}}
    result=n0.check(runtime,out)
    report=json.loads((out/'receipt.json').read_text())
    assert expected in result['conditions'] and result['status']=='NO_GO'
    if failure=='crash':assert report['child_exitcode']==-11
