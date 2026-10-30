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


def test_real_registered_pass_reaches_selection_repeat_and_publication():
    from nanobody_opt import rules, pr100_live_selection as selection
    from test_pr100 import EPOCH, evidence
    row = real_gate_observations()[0]
    sequence, value = row['sequence'], row['measurement']
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


def test_real_registered_failure_is_retained_for_bounded_rescue():
    from nanobody_opt import rules, pr100_live_selection as selection
    row = real_gate_observations()[1]
    sequence, value = row['sequence'], row['measurement']
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
def test_missing_legacy_aggregation_flag_requires_authentic_registered_receipt(change):
    from nanobody_opt import rules
    row = real_gate_observations()[0]
    value = row['measurement']
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


def test_shipped_trial_registration_pins_actual_runner_and_observer_bytes():
    rows = json.loads(native.REGISTRY.read_text())['registrations']
    assert rows
    for row in rows:
        assert row['implementation_sha256'] == native.sha(native.__file__)
        assert row['observer_sha256'] == native.sha(native.ROOT / 'nanobody_opt/pr100_context_observer.py')
        assert row['actual_validator_worker_attested'] is False
        assert 'trial' in row['qualification_scope']
        assert native.registration(reference.digest(row)) == row


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
