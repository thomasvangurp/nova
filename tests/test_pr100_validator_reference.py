"""Contract regressions; synthetic trace tests do not certify a hardware runtime."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from nanobody_opt import pr100_validator_reference as reference
from nanobody_opt.pr100_context_observer import SCOPES

FIXTURES = Path(__file__).parent / 'fixtures/pr100'


def runtime():
    ref = reference.contract()
    return dict(sources=dict(ref['source_files_sha256']),
                versions=dict(openmm='8.5.0', torch='2.7.1+cu126', pdbfixer='1.12.0',
                              biopython='1.85', numpy='2.0.2', anarci='2026.2.13.2', abnumber='0.4.4'),
                filter_revision=ref['upstream']['filter_sha'],
                model_weights={str(i): hashlib.sha256(f'weight{i}'.encode()).hexdigest() for i in range(1, 5)},
                tnp_cli=dict(body_sha256=ref['tnp_cli_body_sha256'], file_sha256='c'*64,
                             shebang_matches_interpreter=True),
                interpreter_sha256='d'*64, python_version=[3, 12, 14],
                native_tools={k: 'a'*64 for k in ('aggrescan3d-rs', 'freesasa', 'psa')})


def validate_runtime(value):
    return reference.validate_runtime(value, expected_weights=runtime()['model_weights'], expected_numpy='2.0.2')


def trace():
    rows = []
    def emit(event, **fields):
        index = len(rows) + 1
        rows.append(dict(schema='pr100-context-observer-v1', event=event, pid=31,
                         event_index=index, monotonic=float(index), **fields))
        return index
    emit('observer_started', scopes=list(SCOPES), observer_sha256='b'*64, scientific_overrides=False)
    emit('prediction', sequence_sha256='a'*64, cuda_available=True, device='cuda', torch_threads=1,
         model_ids=[1, 2, 3, 4], ranking=[2, 1, 0, 3],
         coordinates=[dict(model_index=i, coordinate_sha256=str(i)*64) for i in range(4)])
    emit('unrefined_saved', model_index=2, pdb_sha256='c'*64)
    emit('refine_started', model_index=2, n_threads=1, tries=3, n=6, check_for_strained_bonds=True)
    outer = emit('phase_started', role='refine', parent_phase_id=None)
    missing = emit('phase_started', role='missing_atoms', parent_phase_id=outer)
    emit('phase_finished', role='missing_atoms', phase_id=missing, context_count=0, error_type=None)
    minimize = emit('phase_started', role='minimize_energy', parent_phase_id=outer)
    hydrogen = emit('phase_started', role='hydrogens', parent_phase_id=minimize)
    emit('context_created', role='hydrogens', phase_id=hydrogen, platform='CUDA', properties={'Precision': 'single'})
    emit('phase_finished', role='hydrogens', phase_id=hydrogen, context_count=1, error_type=None)
    emit('context_created', role='minimize_energy', phase_id=minimize, platform='CPU', properties={'Threads': '1'})
    emit('phase_finished', role='minimize_energy', phase_id=minimize, context_count=1, error_type=None)
    emit('phase_finished', role='refine', phase_id=outer, context_count=0, error_type=None)
    emit('refine_finished', model_index=2, success=True)
    emit('model_saved', model_index=2, refinement_success=True, pdb_sha256='d'*64)
    emit('observer_finished', contexts=2, failed_contexts=0, hooks_intact=True)
    return rows


def validate_trace(rows):
    return reference.validate_trace(rows, sequence_hash='a'*64, expected_observer_sha256='b'*64,
                                    expected_cuda_available=True)


def test_independent_fixture_is_the_published_requirement_file():
    raw = (FIXTURES / 'validator_reference_requirements.txt').read_bytes()
    expected = reference.contract()['upstream']['files_sha256']['requirements/requirements.txt']
    assert hashlib.sha256(raw).hexdigest() == expected
    assert b'openmm==8.5.0' in raw and b'biopython==1.85' in raw
    assert b'cosmic-amanita/ImmuneBuilder.git@ba353b3c24b02bff48d1508f0ccff9d3a68fad15' in raw
    assert reference.contract()['actual_validator_worker_attested'] is False


def test_patched_bytes_accepted_regardless_of_stale_repository_label():
    value = runtime()
    value['declared_origin'] = 'oxpig/ImmuneBuilder@0df4e2'
    result = validate_runtime(value)
    assert result['published_source_match']
    assert not result['actual_validator_worker_attested']
    value['sources']['ImmuneBuilder.refine'] = '3216f62059f1f6a2cc1c1d5b3cbc295e87ddc6a6757268a3b9b8c161f6e1dec2'
    with pytest.raises(ValueError, match='source_bytes_mismatch'):
        validate_runtime(value)


@pytest.mark.parametrize('name,version', [('openmm', '8.5.1'), ('torch', '2.7.2'),
                                         ('pdbfixer', '1.11.0'), ('biopython', '1.84'),
                                         ('anarci', '2024.05.21'), ('abnumber', '0.2.7')])
def test_independent_published_versions_reject_self_consistent_local_pins(name, version):
    value = runtime()
    value['versions'][name] = version
    with pytest.raises(ValueError, match='published_version_mismatch'):
        validate_runtime(value)


def test_numpy_is_exact_experiment_binding_not_invented_upstream_pin():
    value = runtime()
    value['versions']['numpy'] = '1.26.4'
    with pytest.raises(ValueError, match='numpy_experiment_mismatch'):
        validate_runtime(value)
    result = reference.validate_runtime(value, expected_weights=value['model_weights'], expected_numpy='1.26.4')
    assert result['numpy_scope'] == 'exact_experiment_pin_not_published_requirement'


@pytest.mark.parametrize('change', ['weight_missing', 'weight_changed', 'tool_missing', 'cli_body', 'shebang'])
def test_full_runtime_assets_required(change):
    value = runtime()
    if change == 'weight_missing': value['model_weights'].pop('4')
    elif change == 'weight_changed': value['model_weights']['1'] = 'f'*64
    elif change == 'tool_missing': value['native_tools'].pop('psa')
    elif change == 'cli_body': value['tnp_cli']['body_sha256'] = 'f'*64
    else: value['tnp_cli']['shebang_matches_interpreter'] = False
    with pytest.raises(ValueError): validate_runtime(value)


def test_gpu_prediction_cpu_minimization_and_observed_hydrogen_context():
    result = validate_trace(trace())
    assert result['prediction_device'] == 'cuda'
    assert result['all_contexts_observed']


@pytest.mark.parametrize('change', ['simulation_only', 'hidden_hydrogens', 'inverse_device', 'forced_cpu_prediction',
                                  'missing_finished', 'changed_hooks', 'wrong_final_model', 'wrong_counts'])
def test_incomplete_or_inverted_reference_trace_is_rejected(change):
    rows = trace()
    if change == 'simulation_only': rows[0]['scopes'] = ['openmm.app.Simulation']
    elif change == 'hidden_hydrogens':
        for row in rows:
            if row.get('role') == 'hydrogens': row['role'] = 'minimize_energy'
    elif change == 'inverse_device':
        next(r for r in rows if r.get('event') == 'context_created' and r.get('role') == 'minimize_energy')['platform'] = 'CUDA'
    elif change == 'forced_cpu_prediction': rows[1]['device'] = 'cpu'
    elif change == 'missing_finished': rows.pop()
    elif change == 'changed_hooks': rows[-1]['hooks_intact'] = False
    elif change == 'wrong_final_model': rows[-2]['model_index'] = 1
    else: rows[-1]['contexts'] = 1
    with pytest.raises(ValueError): validate_trace(rows)


def receipt():
    value = runtime()
    rows = trace()
    return dict(schema=reference.SCHEMA, source_sha='1'*40, sequence_hash='a'*64,
                request_sha256='2'*64, reference_contract_sha256=reference.digest(reference.contract()),
                runtime=value, runtime_identity_sha256=reference.digest(value), trace=rows,
                trace_sha256=reference.digest(rows), actual_validator_worker_attested=False,
                observation=dict(passed=True, metrics=dict(total_cdr_length=35, cdr3_length=19,
                    cdr3_compactness=1.4274984530521655, positive_charge_patches=0.3685,
                    negative_charge_patches=1.2551, surface_hydrophobic_patches=76.3266,
                    a3d_average='-0.7698669293060233', a3d_maximum='1.5540554031642215')))


def validate_receipt(value):
    return reference.validate_receipt(value, expected_source_sha='1'*40, expected_sequence_hash='a'*64,
            expected_request_sha256='2'*64, expected_weights=runtime()['model_weights'], expected_numpy='2.0.2',
            expected_observer_sha256='b'*64, expected_cuda_available=True,
            expected_runtime_identity_sha256=reference.digest(runtime()))


def test_reference_receipt_never_claims_external_worker_or_promotion():
    result = validate_receipt(receipt())
    assert result['published_source_match'] is True
    assert result['actual_validator_worker_attested'] is False
    assert result['promotion_authorized'] is False


def cuda_trace():
    rows = trace()
    for event in rows:
        if event.get('event') != 'context_created':
            continue
        if event['role'] == 'hydrogens':
            event.update(platform='CPU', properties={'Threads': '1'})
        elif event['role'] == 'minimize_energy':
            event.update(platform='CUDA', properties={
                'Precision': 'mixed', 'DeterministicForces': 'true', 'DeviceIndex': '0'})
    return rows


def test_cuda_policy_is_explicit_and_does_not_change_published_cpu_default():
    rows = cuda_trace()
    with pytest.raises(ValueError, match='upstream_refinement_device_mismatch'):
        validate_trace(rows)
    result = reference.validate_trace(rows, sequence_hash='a'*64,
        expected_observer_sha256='b'*64, expected_cuda_available=True,
        expected_final_refinement=reference.CUDA_REFINEMENT, expected_preparation_threads=1)
    assert result['all_contexts_observed']


@pytest.mark.parametrize('change', ['precision', 'determinism', 'device', 'main_cpu', 'hydrogen_cuda', 'prep_threads'])
def test_cuda_policy_requires_exact_main_and_cpu_preparation_properties(change):
    rows = cuda_trace()
    main = next(r for r in rows if r.get('event') == 'context_created' and r['role'] == 'minimize_energy')
    hydrogen = next(r for r in rows if r.get('event') == 'context_created' and r['role'] == 'hydrogens')
    if change == 'precision': main['properties']['Precision'] = 'single'
    elif change == 'determinism': main['properties']['DeterministicForces'] = 'false'
    elif change == 'device': main['properties']['DeviceIndex'] = '1'
    elif change == 'main_cpu': main.update(platform='CPU', properties={'Threads': '1'})
    elif change == 'hydrogen_cuda': hydrogen['platform'] = 'CUDA'
    else: hydrogen['properties']['Threads'] = '4'
    with pytest.raises(ValueError):
        reference.validate_trace(rows, sequence_hash='a'*64, expected_observer_sha256='b'*64,
            expected_cuda_available=True, expected_final_refinement=reference.CUDA_REFINEMENT,
            expected_preparation_threads=1)


def test_cuda_receipt_requires_exact_plugin_binding_as_well_as_platform_trace():
    value = receipt()
    value['trace'] = cuda_trace()
    value['trace_sha256'] = reference.digest(value['trace'])
    plugin = {'distribution': 'OpenMM-CUDA-12', 'version': '8.5.0',
              'wheel_sha256': reference.CUDA_PLUGIN_WHEEL_SHA256, 'manifest_sha256': 'e'*64}
    value['runtime']['cuda_plugin'] = plugin
    value['runtime_identity_sha256'] = reference.digest(value['runtime'])
    options = dict(expected_source_sha='1'*40, expected_sequence_hash='a'*64,
        expected_request_sha256='2'*64, expected_weights=runtime()['model_weights'], expected_numpy='2.0.2',
        expected_observer_sha256='b'*64, expected_cuda_available=True,
        expected_runtime_identity_sha256=value['runtime_identity_sha256'],
        expected_final_refinement=reference.CUDA_REFINEMENT, expected_preparation_threads=1)
    with pytest.raises(ValueError, match='invalid_cuda_plugin_identity'):
        reference.validate_receipt(value, **options)
    assert reference.validate_receipt(value, **options, expected_cuda_plugin=plugin)['published_source_match']
    changed = {**plugin, 'manifest_sha256': 'f'*64}
    with pytest.raises(ValueError, match='cuda_plugin_identity_mismatch'):
        reference.validate_receipt(value, **options, expected_cuda_plugin=changed)


def test_cuda_loader_descriptor_optional_for_research_but_exact_when_present():
    plugin={'distribution':'OpenMM-CUDA-12','version':'8.5.0',
        'wheel_sha256':reference.CUDA_PLUGIN_WHEEL_SHA256,'manifest_sha256':'e'*64}
    assert reference.validate_cuda_plugin_identity(plugin) is plugin
    loader={'core_sha256':'a'*64,'nvrtc_sha256':'b'*64,'torch_compiled_cuda':12060,'cpp_version':'8.5'}
    plugin['loader']=loader
    assert reference.validate_cuda_plugin_identity(plugin) is plugin
    for key,bad in [('core_sha256','short'),('nvrtc_sha256','short'),
                    ('torch_compiled_cuda',12090),('cpp_version','8.5.1')]:
        plugin['loader']={**loader,key:bad}
        with pytest.raises(ValueError,match='invalid_cuda_loader_identity'):
            reference.validate_cuda_plugin_identity(plugin)
    plugin['loader']={**loader,'extra':True}
    with pytest.raises(ValueError,match='invalid_cuda_loader_identity'):
        reference.validate_cuda_plugin_identity(plugin)


@pytest.mark.parametrize('key', ['source_sha', 'sequence_hash', 'request_sha256', 'reference_contract_sha256',
                                'runtime_identity_sha256', 'trace_sha256', 'actual_validator_worker_attested'])
def test_receipt_reuse_and_false_attestation_rejected(key):
    value = receipt()
    value[key] = True if key == 'actual_validator_worker_attested' else 'f'*64
    with pytest.raises(ValueError): validate_receipt(value)


def diagnostic_canary():
    historical = json.loads((FIXTURES / 'backend_reference_26019_08ed265.json').read_text())
    rows = []
    for row in historical['rows']:
        item = {k: row[k] for k in ('sequence', 'sequence_hash')}
        for source, target in [('cpu_observations', 'reference_observations'), ('gpu_observations', 'trial_observations')]:
            item[target] = [dict(passed=o['outcome'] == 'passed', metrics={**o['metrics'],
                            'a3d_average': o['aggrescan3d']['a3d_average'],
                            'a3d_maximum': o['aggrescan3d']['a3d_maximum']}) for o in row[source]]
        rows.append(item)
    return historical, rows


def test_actual_candidate_disagreement_cannot_be_hidden_by_aggregate_variance():
    historical, rows = diagnostic_canary()
    assert historical['upstream_runtime_parity_proven'] is False
    assert historical['critical_canary']['normative_reference_unestablished'] is True
    result = reference.candidate_canary(rows)
    critical = next(r for r in result['candidate_comparisons'] if r['sequence_hash'] == reference.CRITICAL_CANARY)
    assert critical['reference_passes'] == 3 and critical['trial_passes'] == 0
    assert result['reason'] == 'reference_unqualified'
    assert not result['extended_trial_supported'] and not result['promotion_authorized']


def test_new_reference_may_reject_old_cpu_canary_without_forcing_acceptance():
    _, rows = diagnostic_canary()
    for row in rows: row['reference_observations'] = copy.deepcopy(row['trial_observations'])
    validations = []
    for row in rows:
        for index, observation in enumerate(row['reference_observations']):
            value = receipt()
            key = row['sequence_hash']
            request = hashlib.sha256(f'{key}-{index}'.encode()).hexdigest()
            value.update(sequence_hash=key, request_sha256=request, observation=observation)
            value['trace'][1]['sequence_sha256'] = key
            value['trace_sha256'] = reference.digest(value['trace'])
            validations.append(reference.validate_receipt(value, expected_source_sha='1'*40,
                expected_sequence_hash=key, expected_request_sha256=request,
                expected_weights=runtime()['model_weights'], expected_numpy='2.0.2',
                expected_observer_sha256='b'*64, expected_cuda_available=True,
                expected_runtime_identity_sha256=reference.digest(runtime())))
    result = reference.candidate_canary(rows, reference_validations=validations)
    assert result['extended_trial_supported']
    assert not result['promotion_authorized']
    with pytest.raises(ValueError, match='must_be_revalidated'):
        reference.candidate_canary(rows, reference_validations=json.loads(json.dumps(validations)))
    validations[-1] = validations[0]
    with pytest.raises(ValueError, match='receipt_reused'):
        reference.candidate_canary(rows, reference_validations=validations)


def test_omitted_canary_or_forged_metric_decision_cannot_approve():
    _, rows = diagnostic_canary()
    with pytest.raises(ValueError, match='required_canary_omitted'):
        reference.candidate_canary([r for r in rows if r['sequence_hash'] != reference.CRITICAL_CANARY])
    rows[0]['trial_observations'][0]['passed'] = not rows[0]['trial_observations'][0]['passed']
    with pytest.raises(ValueError, match='candidate_decision_disagrees'):
        reference.candidate_canary(rows)


def test_aggregate_only_input_is_not_candidate_evidence():
    with pytest.raises(ValueError, match='missing_candidate_canary'):
        reference.candidate_canary([])


def test_backend_red_region_pass_is_not_validator_psh89_pass():
    historical, rows = diagnostic_canary()
    old = historical['rows'][0]['gpu_observations'][0]
    assert old['backend_passed'] is True and old['outcome'] == 'rejected'
    rows[0]['trial_observations'][0]['passed'] = old['backend_passed']
    with pytest.raises(ValueError, match='candidate_decision_disagrees'):
        reference.candidate_canary(rows)


def test_negative_a3d_maximum_is_rejected_by_actual_pinned_upstream_filter():
    path = reference.ROOT / 'NOVA-nanobody-filter/metanano/filters/aggrescan3d.py'
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == reference.contract()['source_files_sha256']['metanano.filters.aggrescan3d']
    assert b'scores["a3d_maximum"] < 0' in source
    metrics = receipt()['observation']['metrics']
    metrics.update(a3d_average='-2', a3d_maximum='-1')
    with pytest.raises(ValueError, match='invalid_candidate_metrics'):
        reference.measured_gate(metrics)


def test_self_consistent_changed_tool_hash_does_not_match_approved_runtime():
    value = receipt()
    value['runtime']['native_tools']['psa'] = 'f'*64
    value['runtime_identity_sha256'] = reference.digest(value['runtime'])
    with pytest.raises(ValueError, match='approved_runtime_identity_mismatch'):
        validate_receipt(value)


def test_gate_preserves_published_inclusive_caps_and_decimal_precision():
    metrics = receipt()['observation']['metrics']
    metrics.update(surface_hydrophobic_patches=89, a3d_average='-0.68002724866200',
                   a3d_maximum='1.58023066762925')
    assert reference.measured_gate(metrics)
    metrics['a3d_maximum'] = '1.58023066762925001'
    assert not reference.measured_gate(metrics)


def test_charge_patch_caps_do_not_invent_an_upstream_lower_bound():
    metrics = receipt()['observation']['metrics']
    metrics.update(positive_charge_patches=-0.1, negative_charge_patches=-0.2)
    assert reference.measured_gate(metrics)
