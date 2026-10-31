"""Independent published-validator contract for isolated reference experiments.

Passing this contract does not attest to an external validator's worker runtime,
prove statistical equivalence, or authorize a production runtime change.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'pr100-reference-receipt-v1'
CRITICAL_CANARY = '23e071c8d23f0f727fb387dbfc1aefd1e0b8d3782e3a6ff2897834191cf4701b'
CPU_REFINEMENT = {'platform': 'CPU', 'threads': 1}
CUDA_REFINEMENT = {'platform': 'CUDA', 'precision': 'mixed', 'deterministic_forces': True}
CUDA_PLUGIN_WHEEL_SHA256 = '5e9b1054820b177366113d954fa2885837b31333ea2f6daa489da23148469173'


class _CheckedReference(dict):
    """In-process validation result; JSON round trips require revalidation."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def contract():
    return json.loads((ROOT / 'config/pr100_validator_reference.json').read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_refinement_policy(policy):
    require(isinstance(policy, dict) and (
        (policy == CPU_REFINEMENT and type(policy.get('threads')) is int)
        or (policy == CUDA_REFINEMENT and policy.get('deterministic_forces') is True)),
        'invalid_refinement_policy')
    return policy


def validate_cuda_plugin_identity(plugin):
    required = {'distribution', 'version', 'wheel_sha256', 'manifest_sha256'}
    require(isinstance(plugin, dict) and set(plugin) in (required, required | {'loader'})
        and plugin['distribution'] == 'OpenMM-CUDA-12' and plugin['version'] == '8.5.0'
        and plugin['wheel_sha256'] == CUDA_PLUGIN_WHEEL_SHA256
        and sha256(plugin['manifest_sha256']), 'invalid_cuda_plugin_identity')
    if 'loader' in plugin:
        loader=plugin['loader']
        require(isinstance(loader,dict) and set(loader)=={
            'core_sha256','nvrtc_sha256','torch_compiled_cuda','cpp_version'}
            and sha256(loader['core_sha256']) and sha256(loader['nvrtc_sha256'])
            and type(loader['torch_compiled_cuda']) is int and loader['torch_compiled_cuda']==12060
            and loader['cpp_version']=='8.5','invalid_cuda_loader_identity')
    return plugin


def validate_runtime(runtime, *, expected_weights, expected_numpy, expected_cuda_plugin=None):
    """Validate measured bytes against an independent upstream fixture.

    Expected weights/NumPy bind the experiment's selected runtime. They are not
    invented upstream pins: the public specification does not pin either.
    """
    ref = contract()
    require(isinstance(runtime, dict), 'missing_runtime')
    if expected_cuda_plugin is not None:
        validate_cuda_plugin_identity(expected_cuda_plugin)
        require(runtime.get('cuda_plugin') == expected_cuda_plugin, 'cuda_plugin_identity_mismatch')
    require(runtime.get('sources') == ref['source_files_sha256'], 'source_bytes_mismatch')
    require(runtime.get('filter_revision') == ref['upstream']['filter_sha'], 'filter_revision_mismatch')
    versions = runtime.get('versions', {})
    for name, version in [('openmm', '8.5.0'), ('pdbfixer', '1.12.0'), ('biopython', '1.85'),
                          ('anarci', '2026.2.13.2'), ('abnumber', '0.4.4')]:
        require(versions.get(name) == version, 'published_version_mismatch:' + name)
    require(versions.get('torch') in ('2.7.1', '2.7.1+cu126'), 'published_version_mismatch:torch')
    require(isinstance(expected_numpy, str) and re.fullmatch(r'\d+\.\d+\.\d+', expected_numpy),
            'invalid_numpy_experiment_pin')
    require(versions.get('numpy') == expected_numpy, 'numpy_experiment_mismatch')
    require(isinstance(expected_weights, dict) and set(expected_weights) == {'1', '2', '3', '4'}
            and all(sha256(v) for v in expected_weights.values()), 'missing_expected_weight_hashes')
    require(runtime.get('model_weights') == expected_weights, 'model_weights_mismatch')
    cli = runtime.get('tnp_cli', {})
    require(cli.get('body_sha256') == ref['tnp_cli_body_sha256'] and sha256(cli.get('file_sha256')),
            'tnp_cli_bytes_mismatch')
    require(cli.get('shebang_matches_interpreter') is True, 'tnp_interpreter_unbound')
    require(sha256(runtime.get('interpreter_sha256')), 'missing_interpreter_hash')
    python = runtime.get('python_version')
    require(isinstance(python, list) and len(python) == 3 and all(type(x) is int for x in python)
            and python[0] == 3 and python[1] in (10, 11, 12) and python[2] >= 0,
            'unsupported_python')
    tools = runtime.get('native_tools')
    require(isinstance(tools, dict) and set(tools) == {'aggrescan3d-rs', 'freesasa', 'psa'}
            and all(sha256(v) for v in tools.values()), 'missing_native_tool_hashes')
    return {'published_source_match': True, 'actual_validator_worker_attested': False,
            'numpy_scope': 'exact_experiment_pin_not_published_requirement',
            'runtime_identity_sha256': digest(runtime)}


def validate_trace(events, *, sequence_hash, expected_observer_sha256, expected_cuda_available,
                   expected_final_refinement=None, expected_preparation_threads=None):
    refinement = validate_refinement_policy(
        CPU_REFINEMENT if expected_final_refinement is None else expected_final_refinement)
    if expected_preparation_threads is not None:
        require(type(expected_preparation_threads) is int and 1 <= expected_preparation_threads <= 32,
                'invalid_preparation_policy')
    required_scopes = contract()['observer_required_scopes']
    require(sha256(sequence_hash) and sha256(expected_observer_sha256), 'invalid_trace_binding')
    require(type(expected_cuda_available) is bool, 'missing_device_policy')
    require(isinstance(events, list) and bool(events), 'missing_trace')
    processes = {}
    for event in events:
        require(isinstance(event, dict) and event.get('schema') == 'pr100-context-observer-v1',
                'invalid_trace_schema')
        pid = event.get('pid')
        require(type(pid) is int and pid > 0, 'invalid_trace_pid')
        processes.setdefault(pid, []).append(event)
    predictions, models = [], []
    for rows in processes.values():
        require([r.get('event_index') for r in rows] == list(range(1, len(rows) + 1)),
                'trace_event_gap_or_reorder')
        require(rows[0].get('event') == 'observer_started' and rows[-1].get('event') == 'observer_finished',
                'incomplete_observer_lifecycle')
        require(sum(r.get('event') == 'observer_started' for r in rows) == 1
                and sum(r.get('event') == 'observer_finished' for r in rows) == 1,
                'duplicate_observer_lifecycle')
        started, finished = rows[0], rows[-1]
        require(started.get('scopes') == required_scopes, 'simulation_only_or_incomplete_observer')
        require(started.get('observer_sha256') == expected_observer_sha256
                and started.get('scientific_overrides') is False, 'unbound_or_mutating_observer')
        require(finished.get('hooks_intact') is True, 'observer_hooks_changed')
        created = [r for r in rows if r.get('event') == 'context_created']
        failed = [r for r in rows if r.get('event') == 'context_failed']
        require(type(finished.get('contexts')) is int and finished['contexts'] == len(created)
                and type(finished.get('failed_contexts')) is int
                and finished['failed_contexts'] == len(failed), 'context_accounting_mismatch')
        require(not any(r.get('event') == 'observer_error' for r in rows), 'observer_error')
        phases = {r['event_index']: r for r in rows if r.get('event') == 'phase_started'}
        ends = [r for r in rows if r.get('event') == 'phase_finished']
        require(len(ends) == len(phases) and {r.get('phase_id') for r in ends} == set(phases),
                'incomplete_context_phases')
        for end in ends:
            phase = phases[end['phase_id']]
            require(phase.get('role') == end.get('role') and end['event_index'] > end['phase_id'],
                    'invalid_phase_binding')
            own = [r for r in created if r.get('phase_id') == end['phase_id']]
            require(end.get('context_count') == len(own), 'phase_context_accounting_mismatch')
            require(all(end['phase_id'] < r['event_index'] < end['event_index'] for r in own),
                    'context_outside_recorded_phase')
        for context in created + failed:
            phase = phases.get(context.get('phase_id'))
            require(phase is not None and phase.get('role') == context.get('role'), 'unclassified_context')
            if context in created:
                require(context.get('platform') in ('CPU', 'CUDA', 'OpenCL', 'Reference')
                        and isinstance(context.get('properties'), dict), 'unobserved_context_platform')
                if context['role'] in ('minimize_energy', 'sidechain_repair'):
                    if refinement['platform'] == 'CPU':
                        require(context['platform'] == 'CPU' and context['properties'].get('Threads') == '1',
                                'upstream_refinement_device_mismatch')
                    else:
                        properties = context['properties']
                        require(context['platform'] == 'CUDA'
                            and properties.get('Precision') == 'mixed'
                            and properties.get('DeterministicForces') == 'true'
                            and properties.get('DeviceIndex') == '0', 'registered_cuda_refinement_mismatch')
                if expected_preparation_threads is not None and context['role'] in ('missing_atoms', 'hydrogens'):
                    require(context['platform'] == 'CPU'
                        and context['properties'].get('Threads') == str(expected_preparation_threads),
                        'preparation_policy_not_observed')
        local_predictions = [r for r in rows if r.get('event') == 'prediction']
        predictions.extend(local_predictions)
        local_models = [r for r in rows if r.get('event') == 'model_saved']
        models.extend(local_models)
        if local_predictions:
            require(len(local_predictions) == len(local_models) == 1, 'ambiguous_prediction_or_model')
            roles = {p.get('role') for p in phases.values()}
            require({'missing_atoms', 'hydrogens', 'minimize_energy'} <= roles,
                    'missing_internal_context_phase_observation')
            require(any(r.get('role') == 'hydrogens' for r in created)
                    and any(r.get('role') == 'minimize_energy' for r in created),
                    'missing_hydrogen_or_minimizer_context')
            starts = [r for r in rows if r.get('event') == 'refine_started']
            stops = [r for r in rows if r.get('event') == 'refine_finished']
            require(bool(starts) and len(starts) == len(stops), 'missing_refinement_outcomes')
            for attempt in starts:
                require(attempt.get('n_threads') == 1 and attempt.get('tries') == 3
                        and attempt.get('n') == 6 and attempt.get('check_for_strained_bonds') is True,
                        'refinement_parameters_changed')
            saves = [r for r in rows if r.get('event') == 'unrefined_saved']
            require(bool(saves) and len(saves) == len(starts)
                    and all(sha256(r.get('pdb_sha256')) for r in saves),
                    'missing_prepared_coordinate_hashes')
            ranking = local_predictions[0].get('ranking', [])
            require(len(starts) <= 8 and len(ranking) == 4, 'invalid_model_fallback_count')
            for index, (saved, start, stop) in enumerate(zip(saves, starts, stops)):
                require(saved.get('model_index') == start.get('model_index') == stop.get('model_index')
                        == ranking[index // 2]
                        and saved['event_index'] < start['event_index'] < stop['event_index']
                        and type(stop.get('success')) is bool,
                        'model_fallback_or_attempt_binding_changed')
                require(index == len(starts) - 1 or stop['success'] is False,
                        'refinement_continued_after_success')
            require(stops[-1]['success'] is True or len(starts) == 8,
                    'incomplete_failed_model_fallback')
            model = local_models[0]
            require(model.get('model_index') == saves[-1].get('model_index') == stops[-1].get('model_index')
                    and type(model.get('refinement_success')) is bool
                    and model['refinement_success'] == stops[-1].get('success')
                    and sha256(model.get('pdb_sha256')), 'unbound_final_model')
    require(len(predictions) == len(models) == 1, 'missing_or_multiple_prediction_processes')
    pred = predictions[0]
    require(pred.get('sequence_sha256') == sequence_hash, 'prediction_sequence_mismatch')
    require(pred.get('cuda_available') is expected_cuda_available
            and pred.get('device') == ('cuda' if expected_cuda_available else 'cpu'),
            'automatic_prediction_device_mismatch')
    require(pred.get('torch_threads') == 1 and pred.get('model_ids') == [1, 2, 3, 4]
            and sorted(pred.get('ranking', [])) == [0, 1, 2, 3], 'prediction_parameters_changed')
    coordinates = pred.get('coordinates', [])
    require(len(coordinates) == 4 and {r.get('model_index') for r in coordinates} == {0, 1, 2, 3}
            and all(sha256(r.get('coordinate_sha256')) for r in coordinates), 'missing_prediction_coordinates')
    return {'all_contexts_observed': True, 'prediction_device': pred['device'],
            'final_refinement_success': models[0]['refinement_success'], 'trace_sha256': digest(events)}


def validate_receipt(receipt, *, expected_source_sha, expected_sequence_hash, expected_request_sha256,
                     expected_weights, expected_numpy, expected_observer_sha256,
                     expected_cuda_available, expected_runtime_identity_sha256,
                     expected_final_refinement=None, expected_preparation_threads=None,
                     expected_cuda_plugin=None):
    refinement = validate_refinement_policy(
        CPU_REFINEMENT if expected_final_refinement is None else expected_final_refinement)
    if refinement['platform'] == 'CUDA':
        validate_cuda_plugin_identity(expected_cuda_plugin)
    require(isinstance(expected_source_sha, str) and re.fullmatch('[0-9a-f]{40}', expected_source_sha)
            and sha256(expected_sequence_hash) and sha256(expected_request_sha256)
            and sha256(expected_runtime_identity_sha256), 'invalid_expected_binding')
    require(isinstance(receipt, dict) and receipt.get('schema') == SCHEMA, 'invalid_reference_receipt')
    for key, expected in [('source_sha', expected_source_sha), ('sequence_hash', expected_sequence_hash),
                          ('request_sha256', expected_request_sha256)]:
        require(receipt.get(key) == expected, 'reference_binding_mismatch:' + key)
    require(receipt.get('reference_contract_sha256') == digest(contract()), 'reference_contract_changed')
    runtime = validate_runtime(receipt.get('runtime'), expected_weights=expected_weights,
                               expected_numpy=expected_numpy, expected_cuda_plugin=expected_cuda_plugin)
    require(receipt.get('runtime_identity_sha256') == runtime['runtime_identity_sha256'],
            'runtime_identity_changed')
    require(runtime['runtime_identity_sha256'] == expected_runtime_identity_sha256,
            'approved_runtime_identity_mismatch')
    trace = validate_trace(receipt.get('trace'), sequence_hash=expected_sequence_hash,
                           expected_observer_sha256=expected_observer_sha256,
                           expected_cuda_available=expected_cuda_available,
                           expected_final_refinement=refinement,
                           expected_preparation_threads=expected_preparation_threads)
    require(receipt.get('trace_sha256') == trace['trace_sha256'], 'trace_digest_changed')
    require(receipt.get('actual_validator_worker_attested') is False, 'unsupported_worker_attestation')
    observation = receipt.get('observation')
    require(isinstance(observation, dict) and type(observation.get('passed')) is bool
            and observation['passed'] is measured_gate(observation.get('metrics')),
            'reference_observation_decision_mismatch')
    return _CheckedReference({**runtime, **trace, 'promotion_authorized': False,
            'sequence_hash': expected_sequence_hash, 'request_sha256': expected_request_sha256,
            'observation_sha256': digest(observation),
            'scope': 'published_source_compatible_observed_reference_experiment'})


def measured_gate(metrics):
    limits = contract()['thresholds']
    ranges = {'surface_hydrophobic_patches': (limits['psh_inclusive_min'], limits['psh_inclusive_max']),
              'total_cdr_length': limits['cdr_total_length_inclusive'],
              'cdr3_length': limits['cdr3_length_inclusive'],
              'cdr3_compactness': limits['cdr3_compactness_inclusive'],
              'positive_charge_patches': (-math.inf, limits['ppc_max_inclusive']),
              'negative_charge_patches': (-math.inf, limits['pnc_max_inclusive'])}
    require(isinstance(metrics, dict) and set(ranges) | {'a3d_average', 'a3d_maximum'} <= set(metrics),
            'missing_candidate_metrics')
    for key in ranges:
        value = metrics[key]
        require(type(value) in (int, float) and math.isfinite(value), 'invalid_candidate_metrics')
    try:
        average, maximum = Decimal(str(metrics['a3d_average'])), Decimal(str(metrics['a3d_maximum']))
    except InvalidOperation:
        raise ValueError('invalid_candidate_metrics') from None
    require(average.is_finite() and maximum.is_finite() and maximum >= 0 and average <= maximum,
            'invalid_candidate_metrics')
    return (all(lo <= metrics[key] <= hi for key, (lo, hi) in ranges.items())
            and average <= Decimal(limits['a3d_average_max_inclusive'])
            and maximum <= Decimal(limits['a3d_maximum_max_inclusive']))


def candidate_canary(rows, *, reference_validations=None, required_hashes=None):
    """Report individual decisions; aggregate variance never establishes parity.

    The historical fixture is diagnostic, so its old CPU labels are not a
    normative validator reference. Even a clean small canary only supports an
    extended trial, not a production parity claim.
    """
    require(isinstance(rows, list) and bool(rows), 'missing_candidate_canary')
    required_hashes = set(required_hashes or [CRITICAL_CANARY])
    seen, comparisons, observations_to_bind = set(), [], []
    for row in rows:
        key = row.get('sequence_hash')
        require(sha256(key) and key not in seen, 'duplicate_or_invalid_canary_candidate')
        seen.add(key)
        require(hashlib.sha256(row.get('sequence', '').encode()).hexdigest() == key,
                'canary_sequence_mismatch')
        counts = []
        for name in ('reference_observations', 'trial_observations'):
            observations = row.get(name)
            require(isinstance(observations, list) and len(observations) >= 3, 'missing_candidate_repeats')
            values = []
            for observation in observations:
                require(type(observation.get('passed')) is bool, 'missing_candidate_decision')
                require(observation['passed'] is measured_gate(observation.get('metrics')),
                        'candidate_decision_disagrees_with_measured_metrics')
                values.append(observation['passed'])
            counts.append((sum(values), len(values)))
        require(counts[0][1] == counts[1][1], 'unmatched_candidate_repeats')
        observations_to_bind.extend((key, digest(o)) for o in row['reference_observations'])
        comparisons.append({'sequence_hash': key, 'reference_passes': counts[0][0],
                            'trial_passes': counts[1][0], 'repeats': counts[0][1],
                            'decision_count_changed': counts[0][0] != counts[1][0]})
    require(required_hashes <= seen, 'required_canary_omitted')
    qualified_reference = reference_validations is not None
    if qualified_reference:
        require(isinstance(reference_validations, list)
                and all(isinstance(v, _CheckedReference) for v in reference_validations),
                'reference_receipts_must_be_revalidated')
        require(len({v['request_sha256'] for v in reference_validations}) == len(reference_validations),
                'reference_draw_receipt_reused')
        require(sorted((v['sequence_hash'], v['observation_sha256']) for v in reference_validations)
                == sorted(observations_to_bind), 'reference_candidate_observation_binding_mismatch')
    changed = [r['sequence_hash'] for r in comparisons if r['decision_count_changed']]
    return {'reference_qualified': qualified_reference is True,
            'candidate_comparisons': comparisons, 'decision_changes': changed,
            'candidate_decisions_match': not changed,
            'extended_trial_supported': qualified_reference is True and not changed,
            'promotion_authorized': False,
            'reason': 'reference_unqualified' if qualified_reference is not True else
                      ('candidate_decision_changes_require_review' if changed else 'small_canary_only')}
