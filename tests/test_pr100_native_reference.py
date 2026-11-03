"""Scientific registration and scheduling tests; no synthetic hardware approval."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from nanobody_opt import pr100_native_reference as native
from nanobody_opt import pr100_validator_reference as reference
from test_pr100_validator_reference import receipt as reference_receipt


def real_gate_observations():
    """Unmodified production draws; annotation/publication scaffolding is synthetic."""
    path = Path(__file__).parent / 'fixtures/pr100/registered_gate_real_26030.json'
    return json.loads(path.read_text())['observations']


def test_retained_native_history_authenticates_once_and_revalidates_after_json(monkeypatch):
    from nanobody_opt import pr100_native_evidence_cache as cache
    value = real_gate_observations()[0]['measurement']
    original = native.authoritative_measurement
    calls = []
    def validate(value):
        calls.append(True)
        return original(value)
    monkeypatch.setattr(native, 'authoritative_measurement', validate)
    retained = cache.verified(value)
    identity = native.measurement_identity(value)
    before = len(calls)
    for _ in range(20):
        assert cache.authoritative_measurement(retained)
        assert cache.measurement_identity(retained) == identity
    assert len(calls) == before
    replay = json.loads(json.dumps(retained))
    assert type(replay) is dict and replay == value
    assert cache.authoritative_measurement(replay)
    assert len(calls) == before + 1


@pytest.mark.parametrize('fault', ['metric', 'nested_receipt', 'registry', 'contract', 'container_type'])
def test_retained_receipt_never_hides_mutation_or_changed_approval(tmp_path, monkeypatch, fault):
    from nanobody_opt import pr100_native_evidence_cache as cache
    value = cache.verified(real_gate_observations()[0]['measurement'])
    if fault == 'metric':
        value['surface_hydrophobic_patches'] += 1
    elif fault == 'nested_receipt':
        value['native_reference']['receipt']['observation']['passed'] = False
    elif fault == 'registry':
        registry = tmp_path / 'registry.json'
        registry.write_text(json.dumps({'schema': 'pr100-native-reference-registry-v1',
                                        'registrations': []}))
        monkeypatch.setattr(native, 'REGISTRY', registry)
    elif fault == 'container_type':
        runtime = value['native_reference']['receipt']['runtime']
        runtime['python_version'] = tuple(runtime['python_version'])
    else:
        original = reference.contract()
        monkeypatch.setattr(reference, 'contract', lambda: {**original, 'changed': True})
    assert not cache.authoritative_measurement(value)
    assert cache.measurement_identity(value) is None


def test_failed_native_receipt_cannot_receive_process_local_verification():
    from nanobody_opt import pr100_native_evidence_cache as cache
    value = real_gate_observations()[0]['measurement']
    value['authoritative'] = False
    with pytest.raises(ValueError, match='non_authoritative_measurement'):
        cache.verified(value)


@pytest.mark.parametrize('retained', [False, True])
def test_real_registered_pass_reaches_selection_repeat_and_publication(retained):
    from nanobody_opt import rules, pr100_live_selection as selection
    from test_pr100 import EPOCH, evidence
    row = real_gate_observations()[0]
    sequence, value = row['sequence'], row['measurement']
    if retained:
        from nanobody_opt.pr100_native_evidence_cache import verified
        value = verified(value)
    requirements = rules.pinned_config(EPOCH)['nanobody_requirements']
    policy = {'kind': 'pr100-observed-fleet-v1', 'minimum_smoothed_fraction': .4}
    assert 'passed' not in value['aggrescan3d']  # Exact upstream wrapper contract.
    assert native.authoritative_measurement(value)
    assert rules.developability_reason(value, requirements, sequence=sequence) == ''
    verdict = selection.decide([value], requirements, sequence=sequence, fleet_policy=policy)
    assert verdict['tier'] == 'B'
    assert verdict['probability_summary']['native_authority_validated'] is True
    assert selection.should_repeat([value], requirements, sequence=sequence,
                                   fleet_policy=policy, repeats=3)
    item = evidence(sequence)  # Synthetic annotation, not a live eligibility claim.
    item['developability'] = value
    item['native_fleet_selection'] = verdict['native_fleet_selection']
    rules.validate_evidence(item, sequence, item['challenge'], rules.pinned_config(EPOCH))


@pytest.mark.parametrize('retained', [False, True])
def test_real_registered_failure_is_retained_for_bounded_rescue(retained):
    from nanobody_opt import rules, pr100_live_selection as selection
    row = real_gate_observations()[1]
    sequence, value = row['sequence'], row['measurement']
    if retained:
        from nanobody_opt.pr100_native_evidence_cache import verified
        value = verified(value)
    requirements = rules.pinned_config(26030)['nanobody_requirements']
    policy = {'kind': 'pr100-observed-fleet-v1', 'minimum_smoothed_fraction': .4}
    assert native.authoritative_measurement(value)
    assert value['passed'] is False
    assert rules.developability_reason(value, requirements, sequence=sequence)
    verdict = selection.decide([value], requirements, sequence=sequence, fleet_policy=policy)
    assert verdict['tier'] is None and verdict['reason'] == 'no_measured_pass'
    assert verdict['probability_summary']['failing_draws'] == 1
    assert not selection.should_repeat([value], requirements, sequence=sequence,
                                       fleet_policy=policy, repeats=3)
    assert selection.should_repeat([value], requirements, sequence=sequence,
                                   fleet_policy=policy, repeats=3,
                                   failed_first_rescue_available=True)


@pytest.mark.parametrize('change', ['unknown_registration', 'missing_receipt', 'changed_metric',
                                   'explicit_false', 'explicit_null', 'legacy_backend'])
@pytest.mark.parametrize('retained', [False, True])
def test_missing_legacy_aggregation_flag_requires_authentic_registered_receipt(change, retained):
    from nanobody_opt import rules
    row = real_gate_observations()[0]
    value = row['measurement']
    if retained:
        from nanobody_opt.pr100_native_evidence_cache import verified
        value = verified(value)
    if change == 'unknown_registration': value['native_reference']['registration_sha256'] = '0' * 64
    elif change == 'missing_receipt': value['native_reference'].pop('receipt')
    elif change == 'changed_metric': value['aggrescan3d']['a3d_maximum'] = 0.
    elif change == 'explicit_false': value['aggrescan3d']['passed'] = False
    elif change == 'explicit_null': value['aggrescan3d']['passed'] = None
    else: value['evidence_backend'] = 'cpu_native'
    assert rules.developability_reason(value, rules.pinned_config(26030)['nanobody_requirements'],
                                       sequence=row['sequence']) == 'aggrescan_unavailable_or_failed'


def atom(serial, name, x, element):
    return f'ATOM  {serial:5d} {name:>4s} ALA H   1    {x:8.3f}{2.:8.3f}{3.:8.3f}{1.:6.2f}{0.:6.2f}          {element:>2s}  \n'


def test_historical_registrations_remain_bound_without_relabeling_old_evidence():
    rows = json.loads(native.REGISTRY.read_text())['registrations']
    assert rows
    for row in rows:
        assert reference.sha256(row['implementation_sha256'])
        assert reference.sha256(row['observer_sha256'])
        assert row['actual_validator_worker_attested'] is False
        assert isinstance(row['qualification_scope'], str) and row['qualification_scope']
        if row['name'] == 'published-source-cuda-prediction-cpu-refinement-prep1':
            assert 'trial' in row['qualification_scope']
        assert native.registration(reference.digest(row)) == row


def test_qualified_loader126_row_binds_actual_report_runtime_and_frozen_source():
    from nanobody_opt import pr100_registered_resident as resident
    rows=json.loads(native.REGISTRY.read_text())['registrations']
    assert [reference.digest(row) for row in rows[:3]]==[
        '376a25c5a0162d874e179f18170177f5fe1ce30f75c3b2b083e2941a06ac6ccf',
        'a22d5e4f19e3a1fa4a97f20c60eec050c8534a3ce7c9b2f000a56bbdc8795c97',
        '1c42ca3af222e23fd9aa51c1f9845457cb5371054009334aa4b9c1934d060324']
    row=next(r for r in rows if r['name'].endswith('cuda-refinement-prep1-loader126'))
    assert row['qualification_status']=='qualified' and row['actual_validator_worker_attested'] is False
    # Historical receipts retain their original implementation identity.
    assert reference.digest(row)=='4824cd0daa6b9d322783679c8f143a6d604194f5939310843ea8d242c002c7dd'
    assert row['resident_module_sha256']=='b03276270ee8c5b0ca91e6ed3d2301926b339086d3519da51f587d5c9c1c7697'
    assert row['implementation_sha256']==native.sha(native.__file__)
    proof=native.ROOT/row['qualification_report_file']
    assert native.sha(proof)==row['qualification_report_sha256']
    result=json.loads(proof.read_text())
    assert result['completed']==result['draw_count']==result['fresh_predictions']==28
    assert result['authoritative'] is False and result['sealed_closure_unchanged'] is True
    assert result['cleanup']['remaining_descendants']==0 and result['cleanup']['fixed_point_reached'] is True
    assert result['child_exit_code']==0 and result['released'] is True
    runtime=real_gate_observations()[0]['measurement']['native_reference']['receipt']['runtime']
    assert row['runtime_sha256']==reference.digest({**runtime,'cuda_plugin':row['cuda_plugin']})
    assert row['cuda_plugin']['loader']['nvrtc_sha256']==next(
        r['sha256'] for r in result['actual_maps']['libraries'] if 'libnvrtc.so.' in r['path'])


def test_current_router_registration_binds_real_direct_repair_canary():
    from nanobody_opt import pr100_registered_resident as resident
    rows=json.loads(native.REGISTRY.read_text())['registrations']
    row=next(r for r in rows if r.get('resident_module_sha256')==native.sha(resident.__file__))
    old=next(r for r in rows if reference.digest(r)=='4824cd0daa6b9d322783679c8f143a6d604194f5939310843ea8d242c002c7dd')
    assert row['qualification_status']=='qualified' and row['actual_validator_worker_attested'] is False
    for key in ('runtime_sha256','observer_sha256','implementation_sha256','environment_manifest_sha256',
                'bundle_manifest_sha256','weights','numpy','policy','cuda_plugin'):
        assert row[key]==old[key]
    proof=native.ROOT/row['qualification_report_file']
    assert native.sha(proof)==row['qualification_report_sha256']
    result=json.loads(proof.read_text())
    assert result['completed']==result['draw_count']==result['fresh_predictions']==result['distinct_sequences']==16
    assert result['authoritative'] is False and result['production_deployment'] is False
    assert result['sealed_closure_unchanged'] is True and result['prediction_reuse'] is False
    assert result['resident_module_sha256']==native.sha(resident.__file__)
    assert result['cleanup']['remaining_descendants']==result['cleanup']['killed_descendants']==0
    assert result['cleanup']['fixed_point_reached'] is True and result['child_exit_code']==0
    groups=result['groups']
    assert len(groups)==2 and groups[0]['four_worker_pids']==groups[1]['four_worker_pids']
    assert len(set(groups[0]['four_worker_pids']))==4
    contexts=[c for g in groups for d in g['draws'] for c in d['contexts']]
    assert any(c['role']=='sidechain_repair' for c in contexts)
    for context in contexts:
        if context['role'] in ('minimize_energy','sidechain_repair'):
            assert context['platform']=='CUDA'
            assert {k:context['properties'][k] for k in ('Precision','DeviceIndex','DeterministicForces')}=={
                'Precision':'mixed','DeviceIndex':'0','DeterministicForces':'true'}


def test_old_runner_registration_cannot_launch_changed_runner_bytes():
    row = json.loads(native.REGISTRY.read_text())['registrations'][0]
    if row['implementation_sha256'] != native.sha(native.__file__):
        with pytest.raises(ValueError, match='runner_bytes_changed'):
            native.verify_files({'bundle': '/unused', 'environment': '/unused'}, row)


@pytest.fixture
def registered(tmp_path, monkeypatch):
    receipt = reference_receipt()
    original_pdb = atom(1, 'N', 1., 'N') + atom(2, 'H', .5, 'H') + atom(3, 'CA', 1.5, 'C')
    final_pdb = atom(1, 'N', 1., 'N') + atom(2, 'CA', 1.5, 'C')
    model_index = next(i for i, event in enumerate(receipt['trace']) if event['event'] == 'model_saved')
    receipt['trace'].insert(model_index + 1, {'schema': 'pr100-context-observer-v1',
        'event': 'geometry_snapshot', 'stage': 'final_model', 'pid': 31,
        'pdb_sha256': hashlib.sha256(original_pdb.encode()).hexdigest()})
    for index, event in enumerate(receipt['trace'], 1):
        event['event_index'] = index
    for event in receipt['trace']:
        if event.get('event') == 'context_created' and event.get('role') == 'hydrogens':
            event.update(platform='CPU', properties={'Threads': '1'})
    receipt['trace_sha256'] = reference.digest(receipt['trace'])
    row = {'runtime_sha256': receipt['runtime_identity_sha256'], 'observer_sha256': 'b' * 64,
        'implementation_sha256': 'c' * 64, 'environment_manifest_sha256': 'd' * 64,
        'bundle_manifest_sha256': 'e' * 64, 'actual_validator_worker_attested': False,
        'weights': receipt['runtime']['model_weights'], 'numpy': '2.0.2',
        'policy': {'prediction': 'auto_cuda', 'final_refinement': {'platform': 'CPU', 'threads': 1},
                   'preparation_threads': 1}}
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'schema': 'pr100-native-reference-registry-v1', 'registrations': [row]}))
    monkeypatch.setattr(native, 'REGISTRY', registry)
    identity = reference.digest(row)
    request = {'source_sha': receipt['source_sha'], 'registration_sha256': identity,
               'policy': row['policy'], 'implementation_sha256': row['implementation_sha256']}
    structure_sha = hashlib.sha256(final_pdb.encode()).hexdigest()
    result_text = json.dumps({'reference_' + receipt['sequence_hash'][:16]: {
        source: receipt['observation']['metrics'][target] for source, target in native.METRICS.items()}})
    receipt['structure_binding'] = {'transformation': 'published_tnp_hydrogen_removal_pdbio',
        'raw_model_sha256': receipt['trace'][model_index]['pdb_sha256'],
        'observed_geometry_pdb': native.pack_artifact(original_pdb),
        'final_pdb': native.pack_artifact(final_pdb),
        'tnp_result_json': result_text, 'tnp_result_sha256': hashlib.sha256(result_text.encode()).hexdigest()}
    request['structure_binding_sha256'] = reference.digest(receipt['structure_binding'])
    receipt.update(request=request, request_sha256=reference.digest(request), structure_sha256=structure_sha)
    receipt['aggregation'] = {'status': 'ok', 'sequence_hash': 'a' * 64,
        'input_structure_sha256': structure_sha,
        **{key: receipt['observation']['metrics'][key] for key in ('a3d_average', 'a3d_maximum')}}
    value = {'schema': native.SCHEMA, 'evidence_backend': native.BACKEND, 'authoritative': True,
        'sequence_hash': 'a' * 64, 'passed': True,
        **{key: receipt['observation']['metrics'][key] for key in native.METRICS.values()},
        'aggrescan3d': copy.deepcopy(receipt['aggregation']),
        'native_reference': {'registration_sha256': identity, 'policy': copy.deepcopy(row['policy']),
                             'receipt': receipt}}
    settings = {'enabled': True, 'registration_sha256': identity, 'gpus': [0, 1],
                'gpu_allowlist': [0, 1], 'workers_per_gpu': 2, 'environment': '/reference/env',
                'bundle': '/reference/bundle', 'lock_directory': '/shared/locks', 'scorer': '/scorer.py'}
    return value, {'source_sha': '1' * 40, 'native_reference': settings}


def test_registered_complete_gate_accepts_and_has_stable_identity(registered):
    value, _ = registered
    assert native.authoritative_measurement(value)
    assert native.measurement_identity(value)['backend'] == 'native_reference'
    assert native.measurement_identity(value)['policy_sha256'] == reference.digest(
        value['native_reference']['policy'])


@pytest.fixture
def cuda_registered(registered):
    value, job = copy.deepcopy(registered)
    registry = json.loads(native.REGISTRY.read_text())
    row = copy.deepcopy(registry['registrations'][0])
    row['policy']['final_refinement'] = dict(reference.CUDA_REFINEMENT)
    row.update(qualification_status='qualified', qualification_report_sha256='4'*64,
               resident_module_sha256='5'*64,
               cuda_plugin={'distribution': 'OpenMM-CUDA-12', 'version': '8.5.0',
                   'wheel_sha256': reference.CUDA_PLUGIN_WHEEL_SHA256, 'manifest_sha256': '6'*64})
    binding = value['native_reference']; receipt = binding['receipt']
    for event in receipt['trace']:
        if event.get('event') == 'context_created' and event['role'] == 'minimize_energy':
            event.update(platform='CUDA', properties={
                'Precision': 'mixed', 'DeterministicForces': 'true', 'DeviceIndex': '0'})
    receipt['trace_sha256'] = reference.digest(receipt['trace'])
    receipt['runtime']['cuda_plugin'] = row['cuda_plugin']
    receipt['runtime_identity_sha256'] = row['runtime_sha256'] = reference.digest(receipt['runtime'])
    identity = reference.digest(row)
    registry['registrations'].append(row); native.REGISTRY.write_text(json.dumps(registry))
    binding.update(registration_sha256=identity, policy=row['policy'])
    receipt['request'].update(registration_sha256=identity, policy=row['policy'], execution='resident-v1',
        resident_module_sha256=row['resident_module_sha256'], model_state_start_sha256='7'*64,
        model_state_end_sha256='7'*64, resident_batch_id='8'*32, resident_batch_draws=16)
    receipt['request_sha256'] = reference.digest(receipt['request'])
    job['native_reference'].update(registration_sha256=identity, resident=True,
                                    cuda_plugin_root='/sidecar')
    return value, job


def test_registered_cuda_receipt_is_distinct_and_requires_qualified_row(cuda_registered):
    value, job = cuda_registered
    assert native.authoritative_measurement(value)
    settings = native.settings_for(job)
    assert settings['resident'] is True and settings['resident_batch_draws'] == 32
    registry = json.loads(native.REGISTRY.read_text())
    row = registry['registrations'][-1]
    row['qualification_status'] = 'pending'
    native.REGISTRY.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='unqualified_cuda_registration'):
        native.registration(reference.digest(row))


@pytest.mark.parametrize('change', ['missing_execution', 'module', 'state', 'batch_id', 'batch_size', 'main', 'prep'])
def test_cuda_resident_receipt_forgery_or_mixed_policy_rejected(cuda_registered, change):
    value, _ = cuda_registered
    receipt = value['native_reference']['receipt']; request = receipt['request']
    if change == 'missing_execution': request.pop('execution')
    elif change == 'module': request['resident_module_sha256'] = 'f'*64
    elif change == 'state': request['model_state_end_sha256'] = 'f'*64
    elif change == 'batch_id': request['resident_batch_id'] = 'not-a-uuid'
    elif change == 'batch_size': request['resident_batch_draws'] = 33
    else:
        role = 'minimize_energy' if change == 'main' else 'hydrogens'
        event = next(r for r in receipt['trace'] if r.get('event') == 'context_created' and r['role'] == role)
        event['platform'] = 'CPU' if change == 'main' else 'CUDA'
        receipt['trace_sha256'] = reference.digest(receipt['trace'])
    receipt['request_sha256'] = reference.digest(request)
    assert not native.authoritative_measurement(value)


def test_cuda_setting_requires_resident_opt_in_and_bounded_batch(cuda_registered):
    _, job = cuda_registered
    job['native_reference']['resident'] = False
    with pytest.raises(ValueError, match='cuda_requires_registered_resident'):
        native.settings_for(job)
    job['native_reference'].update(resident=True, resident_batch_draws=33)
    with pytest.raises(ValueError, match='invalid_resident_batch_draws'):
        native.settings_for(job)


def test_plugin_manifest_is_exact_confined_and_rejects_unsealed_library(tmp_path):
    root = tmp_path / 'sidecar'; directory = root / 'lib/plugins'; directory.mkdir(parents=True)
    plugin = directory / 'libOpenMMCUDA.so'; plugin.write_bytes(b'inert-test-library')
    manifest = {'schema': 'pr100-openmm-cuda-plugin-v1', 'distribution': 'OpenMM-CUDA-12',
        'version': '8.5.0', 'wheel_sha256': reference.CUDA_PLUGIN_WHEEL_SHA256,
        'plugin_directory': 'lib/plugins', 'library_directories': [],
        'files': {'lib/plugins/libOpenMMCUDA.so': native.sha(plugin)}}
    path = root / 'plugin-manifest.json'; path.write_text(json.dumps(manifest))
    row = {'cuda_plugin': {key: manifest[key] for key in ('distribution', 'version', 'wheel_sha256')}}
    row['cuda_plugin']['manifest_sha256'] = native.sha(path)
    settings = {'cuda_plugin_root': str(root)}
    assert native.cuda_plugin_paths(settings, row)['plugin_directory'] == str(directory)
    extra = directory / 'libUnsealed.so'; extra.write_bytes(b'other')
    with pytest.raises(ValueError, match='unsealed_cuda_library'):
        native.cuda_plugin_paths(settings, row)
    extra.unlink(); plugin.write_bytes(b'changed')
    with pytest.raises(ValueError, match='cuda_plugin_file_changed'):
        native.cuda_plugin_paths(settings, row)


def test_cuda_runtime_extends_original_contract_only_in_memory_and_keeps_core_identity(tmp_path):
    original = reference_receipt()['runtime']
    before = copy.deepcopy(original)
    path = tmp_path / 'runtime-contract.json'
    path.write_text(json.dumps(original))
    sealed_sha = native.sha(path)
    row = {'policy': {'final_refinement': dict(reference.CUDA_REFINEMENT)},
        'environment_manifest_sha256': '1'*64, 'bundle_manifest_sha256': '2'*64,
        'cuda_plugin': {'distribution': 'OpenMM-CUDA-12', 'version': '8.5.0',
            'wheel_sha256': reference.CUDA_PLUGIN_WHEEL_SHA256, 'manifest_sha256': '3'*64}}
    row_before = copy.deepcopy(row)
    extended = native._registered_runtime(original, row)
    assert extended == {**before, 'cuda_plugin': row['cuda_plugin']}
    assert original == before and row == row_before and native.sha(path) == sealed_sha
    assert reference.digest(extended) != reference.digest(original)
    for key in ('sources', 'interpreter_sha256', 'python_version', 'model_weights', 'versions', 'tnp_cli', 'native_tools'):
        assert extended[key] == original[key]
    reference.validate_runtime(extended, expected_weights=original['model_weights'],
                               expected_numpy='2.0.2', expected_cuda_plugin=row['cuda_plugin'])
    with pytest.raises(ValueError, match='base_runtime_already_extended'):
        native._registered_runtime(extended, row)
    cpu = {'policy': {'final_refinement': dict(reference.CPU_REFINEMENT)}}
    assert native._registered_runtime(original, cpu) is original


@pytest.mark.parametrize('change', ['unknown_registration', 'legacy_screen', 'old_version', 'device',
                                   'prep_threads', 'metrics', 'aggregation', 'policy', 'request', 'no_receipt'])
def test_forged_or_mixed_evidence_never_becomes_authoritative(registered, change):
    value, _ = registered
    binding = value['native_reference']; receipt = binding['receipt']
    if change == 'unknown_registration': binding['registration_sha256'] = '0' * 64
    elif change == 'legacy_screen': value['evidence_backend'] = 'cuda_refine'
    elif change == 'old_version': receipt['runtime']['versions']['openmm'] = '8.5.1'
    elif change == 'device': receipt['trace'][1]['device'] = 'cpu'
    elif change == 'prep_threads':
        next(row for row in receipt['trace'] if row.get('event') == 'context_created'
             and row.get('role') == 'hydrogens')['properties']['Threads'] = '224'
        receipt['trace_sha256'] = reference.digest(receipt['trace'])
    elif change == 'metrics': value['surface_hydrophobic_patches'] -= 10
    elif change == 'aggregation': value['aggrescan3d']['a3d_maximum'] = '0'
    elif change == 'policy': binding['policy']['preparation_threads'] = 4
    elif change == 'request': receipt['request']['source_sha'] = '0' * 40
    else: binding.pop('receipt')
    assert not native.authoritative_measurement(value)
    assert native.measurement_identity(value) is None


def test_real_negative_draw_is_retained_as_authoritative_evidence(registered):
    value, _ = registered
    value['passed'] = False
    value['surface_hydrophobic_patches'] = 90
    receipt = value['native_reference']['receipt']
    receipt['observation']['passed'] = False
    receipt['observation']['metrics']['surface_hydrophobic_patches'] = 90
    binding = receipt['structure_binding']
    output = json.loads(binding['tnp_result_json'])
    next(iter(output.values()))['PSH'] = 90
    binding['tnp_result_json'] = json.dumps(output)
    binding['tnp_result_sha256'] = hashlib.sha256(binding['tnp_result_json'].encode()).hexdigest()
    receipt['request']['structure_binding_sha256'] = reference.digest(binding)
    receipt['request_sha256'] = reference.digest(receipt['request'])
    assert native.authoritative_measurement(value)


def test_renumbered_heavy_structure_is_valid_but_changed_coordinate_is_not(registered):
    value, _ = registered
    receipt = value['native_reference']['receipt']
    assert native.authoritative_measurement(value)
    binding = receipt['structure_binding']
    changed = atom(1, 'N', 1., 'N') + atom(2, 'CA', 1.501, 'C')
    binding['final_pdb'] = native.pack_artifact(changed)
    receipt['structure_sha256'] = hashlib.sha256(changed.encode()).hexdigest()
    receipt['aggregation']['input_structure_sha256'] = receipt['structure_sha256']
    value['aggrescan3d'] = copy.deepcopy(receipt['aggregation'])
    receipt['request']['structure_binding_sha256'] = reference.digest(binding)
    receipt['request_sha256'] = reference.digest(receipt['request'])
    assert not native.authoritative_measurement(value)


def test_real_upstream_hydrogen_removal_preserves_983_heavy_atoms():
    value = json.loads((Path(__file__).parent / 'fixtures/pr100/reference_hydrogen_strip_real.json').read_text())
    assert hashlib.sha256(value['observed_geometry_pdb'].encode()).hexdigest() == value['observed_geometry_sha256']
    assert hashlib.sha256(value['final_pdb'].encode()).hexdigest() == value['final_tnp_pdb_sha256']
    left, right = (native._heavy_pdb_rows(value[key]) for key in ('observed_geometry_pdb', 'final_pdb'))
    assert len(left) == value['expected_heavy_atoms'] == 983
    assert left == right
    for key in ('observed_geometry_pdb', 'final_pdb'):
        envelope = native.pack_artifact(value[key])
        assert native.unpack_artifact(envelope) == value[key]
        assert len(envelope['data']) < len(value[key]) / 2


@pytest.mark.parametrize('change', ['trailing', 'wrong_size', 'bomb', 'bad_base64'])
def test_scientific_artifact_decode_is_bounded(change):
    envelope = native.pack_artifact('ATOM\n')
    if change == 'trailing':
        envelope['data'] += 'AAAA'
    elif change == 'wrong_size':
        envelope['raw_bytes'] = 1
    elif change == 'bomb':
        envelope = {'encoding': 'zlib-base64-v1', 'raw_bytes': 10,
            'data': native.base64.b64encode(native.zlib.compress(b'A' * 2_000_001)).decode()}
    else:
        envelope['data'] = 'invalid!'
    with pytest.raises(ValueError):
        native.unpack_artifact(envelope)


def test_explicit_allowlist_intersection_and_empty_refusal(registered):
    _, job = registered
    job['native_reference']['gpu_allowlist'] = [1, 2]
    assert native.settings_for(job)['gpus'] == [1]
    job['native_reference']['gpu_allowlist'] = [2]
    with pytest.raises(ValueError, match='no_allowed_reference_gpu'):
        native.settings_for(job)


def test_singleton_refill_coalesces_with_bounded_per_gpu_capacity(registered, tmp_path):
    _, job = registered
    active = set(); batches = []
    async def launch(gpu, sequences, timeout):
        assert gpu not in active
        active.add(gpu); batches.append((gpu, sequences, timeout))
        await asyncio.sleep(.01)
        active.remove(gpu)
        return [{'sequence': sequence} for sequence in sequences]
    adapter = native.Adapter(job, tmp_path, launcher=launch, verify=False)
    async def run():
        return await asyncio.gather(*(adapter.develop([str(i)], 1, task_timeout_seconds=30)
                                      for i in range(7)))
    rows = asyncio.run(run())
    assert [row[0]['sequence'] for row in rows] == [str(i) for i in range(7)]
    assert len(batches) == 4 and all(len(batch[1]) <= 2 for batch in batches)
    assert {gpu for gpu, _, _ in batches} == {0, 1}
    assert not active


def test_busy_gpu_tail_tries_clean_alternative(registered, tmp_path):
    _, job = registered
    attempts = []
    async def launch(gpu, sequences, timeout):
        attempts.append(gpu)
        await asyncio.sleep(.001)
        return ([native.failure('reference_gpu_unavailable')] if gpu == 0 else [{'ok': True}])
    adapter = native.Adapter(job, tmp_path, launcher=launch, verify=False)
    assert asyncio.run(adapter.develop(['A'], 1)) == [{'ok': True}]
    assert attempts == [0, 1]


def test_started_unavailable_work_is_not_retried_elsewhere(registered, tmp_path):
    _, job = registered
    attempts = []
    async def launch(gpu, sequences, timeout):
        attempts.append(gpu)
        return [native.failure('reference_gpu_unavailable', started=True)]
    adapter = native.Adapter(job, tmp_path, launcher=launch, verify=False)
    asyncio.run(adapter.develop(['A'], 1))
    assert attempts == [0]


def test_cancel_waits_for_lease_cleanup_before_return(registered, tmp_path):
    _, job = registered
    cleanup = []
    async def launch(gpu, sequences, timeout):
        try:
            await asyncio.sleep(60)
        finally:
            await asyncio.sleep(.01)
            cleanup.append(gpu)
    adapter = native.Adapter(job, tmp_path, launcher=launch, verify=False)
    async def run():
        task = asyncio.create_task(adapter.develop(['A'], 1))
        await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleanup == [0]
    asyncio.run(run())


def test_uncertain_cleanup_blocks_all_new_admissions(registered, tmp_path):
    _, job = registered
    job['native_reference'].update(gpus=[0], workers_per_gpu=1)
    attempts = []
    async def launch(gpu, sequences, timeout):
        attempts.extend(sequences)
        raise RuntimeError('unverified_reference_cleanup')
    adapter = native.Adapter(job, tmp_path, launcher=launch, verify=False)
    async def run():
        outcomes = await asyncio.gather(*(adapter.develop([str(i)], 1)
                                          for i in range(3)), return_exceptions=True)
        assert all(isinstance(value, RuntimeError) for value in outcomes)
    asyncio.run(run())
    assert attempts == ['0']


def test_small_batch_uses_available_gpus_in_parallel(registered, tmp_path):
    _, job = registered
    job['native_reference']['workers_per_gpu'] = 4
    batches = []
    async def launch(gpu, sequences, timeout):
        batches.append((gpu, sequences))
        await asyncio.sleep(.001)
        return [{} for _ in sequences]
    adapter = native.Adapter(job, tmp_path, launcher=launch, verify=False)
    asyncio.run(adapter.develop(['A', 'B'], 2))
    assert batches == [(0, ['A']), (1, ['B'])]
