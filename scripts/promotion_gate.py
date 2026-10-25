#!/usr/bin/env python3
"""Fail-closed promotion of a tested release using independently checked evidence.

CI permits staging. Promotion additionally requires paired, isolated production
trials, held-out scores, a Pi dry-run receipt, and a state-preserving rollback
trial. This module never scores a molecule, submits, or restores mutable state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.release_manifest import safe_path

SCHEMA = 'sn68-promotion-v1'
SCENARIOS = frozenset({'normal', 'cold_cache', 'mixed_lengths', 'interrupted_resume',
    'partial_wave', 'epoch_rollover', 'unused_publication', 'late_confirmation',
    'gate_attrition', 'stale_proof', 'pi_pool_spare'})
MAX_AGE = 24 * 60 * 60
MIN_FLEET = 13
MIN_PROPOSALS = 8000
NEW_BUDGET = 4800
MIN_DEPTH = 10
MIN_HELD_OUT = 10


def require(condition, reason):
    if not condition:
        # Fixed reason codes only: never echo supplied documents or exceptions.
        raise ValueError(reason)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, 'nonfinite_json'))


def sha(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None


def finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def referenced(root, reference):
    require(isinstance(reference, dict) and sha(reference.get('sha256')), 'invalid_reference')
    path = safe_path(root, reference['path'])
    require(path.is_file() and digest(path) == reference['sha256'], 'evidence_hash_mismatch')
    return read(path)


def ids(values):
    require(isinstance(values, list) and all(sha(x) for x in values), 'invalid_candidate_ids')
    require(len(values) == len(set(values)), 'duplicate_candidate_ids')
    return set(values)


def _fleet(case, result):
    require(result['epoch'] == case['epoch'], 'wrong_epoch')
    proposals = ids(case['proposal_ids'])
    prior = ids(case['previously_evaluated_ids'])
    require(len(proposals) >= MIN_PROPOSALS and case['new_budget'] == NEW_BUDGET,
            'not_production_scale')
    dispatched, completed = ids(result['dispatched_new_ids']), ids(result['completed_new_ids'])
    require(dispatched <= proposals and not dispatched & prior and len(dispatched) <= NEW_BUDGET,
            'new_budget_contract')
    require(completed <= dispatched, 'completion_without_dispatch')
    if case['scenario'] == 'normal':
        require(len(completed) == NEW_BUDGET, 'normal_trial_did_not_exercise_full_budget')
    entries = result['selected']
    selected = ids([entry['id'] for entry in entries])
    require(len(selected) >= MIN_FLEET and len(selected) == case['fleet_size'], 'fleet_shortfall')
    require(selected <= ids(case['currently_eligible_ids']), 'ineligible_selection')
    means = []
    for entry in entries:
        training = ids(entry['selection_pass_ids'])
        held_out = case['held_out'][entry['id']]
        unseen = ids(held_out['pass_ids'])
        scores = held_out['scores']
        require(len(training) >= MIN_DEPTH and not training & unseen, 'evidence_leakage_or_depth')
        require(len(unseen) >= MIN_HELD_OUT and len(unseen) == len(scores)
                and all(finite(v) and v >= 0 for v in scores), 'invalid_held_out_scores')
        means.append(sum(scores) / len(scores))
    handoff = result['handoff']
    from nanobody_opt.submission_timing import plan as timing_plan, BLACKOUT_BLOCKS
    timing = timing_plan(case['fleet_size'])
    length = case['epoch_length']
    require(type(length) is int and length == 361, 'unsupported_epoch_length')
    for key, reserve in [('scoring_cutoff', timing['scoring_reserve_blocks']),
                         ('publication_cutoff', timing['publication_reserve_blocks']),
                         ('pi_pull_cutoff', timing['pi_pull_reserve_blocks']),
                         ('commit_cutoff', BLACKOUT_BLOCKS)]:
        require(finite(case[key]) and 0 <= case[key] <= length - reserve, 'relaxed_submission_window')
    require(sha(handoff['artifact_sha256']) and
            handoff['artifact_sha256'] == handoff['pi_artifact_sha256'], 'torn_handoff')
    offsets = [handoff[k] for k in ['confirmed_offset', 'published_offset', 'pi_pull_offset']]
    require(all(finite(v) and v >= 0 for v in offsets) and offsets == sorted(offsets), 'handoff_order')
    require(offsets[0] <= case['scoring_cutoff'] and offsets[1] <= case['publication_cutoff']
            and offsets[2] <= case['pi_pull_cutoff'], 'late_handoff')
    require(case['pi_pull_cutoff'] < case['commit_cutoff'] < case['epoch_length'], 'invalid_window')
    require(result['old_epoch_work_running'] is False and result['unavailable_gates_admitted'] == 0,
            'epoch_or_gate_isolation')
    require(result['status'] == 'complete' and result['confirmation_health'] == 'healthy',
            'degraded_completion')
    return {'mean': sum(means) / len(means), 'completed_new': len(completed), 'fleet': len(entries)}


def compare(fixture, baseline, candidate):
    require(fixture['schema'] == 'sn68-production-trial-v1', 'trial_schema')
    cases = fixture['cases']
    names = [c['name'] for c in cases]
    require(len(names) == len(set(names)), 'duplicate_trial_case')
    require(SCENARIOS <= {c['scenario'] for c in cases}, 'missing_failure_scenario')
    require(len({c['epoch'] for c in cases if c['scenario'] == 'normal'}) >= 3, 'insufficient_epochs')
    results = []
    for report in [baseline, candidate]:
        require(report['fixture_sha256'] == fixture['_sha256'], 'different_trial_inputs')
        require([r['name'] for r in report['cases']] == names, 'incomplete_trial_results')
    for case, before, after in zip(cases, baseline['cases'], candidate['cases']):
        require(sha(case['field_sha256']), 'missing_field_identity')
        new = _fleet(case, after)
        if before['status'] == 'failed':
            require(case['scenario'] != 'normal', 'baseline_not_known_good')
            results.append({'name': case['name'], 'repaired_failure': True, **new})
            continue
        old = _fleet(case, before)
        require(new['completed_new'] >= old['completed_new'], 'discovery_regression')
        require(new['mean'] <= old['mean'] + 1e-9, 'held_out_quality_regression')
        results.append({'name': case['name'], 'baseline_mean': old['mean'], **new})
    return results


def verify(bundle, *, candidate_sha, host, target, epoch=None, now=None):
    bundle = Path(bundle).resolve()
    plan = read(bundle / 'promotion.json')
    clock = time.time() if now is None else now
    require(plan['schema'] == SCHEMA, 'promotion_schema')
    require(sha(candidate_sha, 40) and plan['candidate_sha'] == candidate_sha, 'candidate_revision')
    require(sha(plan['baseline_sha'], 40) and plan['baseline_sha'] != candidate_sha, 'baseline_revision')
    require(plan['change_kind'] in {'repair', 'experiment', 'rollback'}, 'mixed_or_unknown_change_kind')
    require(plan['host'] == host and plan['target'] == target, 'wrong_deployment_target')
    require(finite(plan['created_at']) and finite(plan['expires_at'])
            and 0 <= clock - plan['created_at'] <= MAX_AGE
            and clock < plan['expires_at'] <= plan['created_at'] + MAX_AGE, 'stale_promotion')
    require(type(plan['activation_epoch']) is int and plan['activation_epoch'] > 0, 'activation_epoch')
    if epoch is not None:
        require(plan['activation_epoch'] == epoch, 'wrong_activation_epoch')
    fixture = referenced(bundle, plan['fixture'])
    require(fixture['target'] == target, 'fixture_target')
    fixture['_sha256'] = plan['fixture']['sha256']
    baseline = referenced(bundle, plan['baseline_report'])
    candidate = referenced(bundle, plan['candidate_report'])
    require(baseline['revision'] == plan['baseline_sha'] and candidate['revision'] == candidate_sha,
            'trial_revision')
    comparison = compare(fixture, baseline, candidate)
    if plan['change_kind'] == 'experiment':
        measured = [r for r in comparison if 'baseline_mean' in r]
        require(sum(r['mean'] for r in measured) < sum(r['baseline_mean'] for r in measured) - 1e-9,
                'experiment_has_no_demonstrated_gain')

    isolation = referenced(bundle, plan['shadow_receipt'])
    require(isolation['schema'] == 'sn68-isolated-trial-v1', 'shadow_schema')
    require(isolation['fixture_sha256'] == plan['fixture']['sha256'], 'shadow_fixture')
    require(isolation['network'] == 'none' and isolation['root_read_only'] is True
            and isolation['host_credentials_mounted'] is False
            and isolation['capabilities'] == [] and isolation['live_submission_access'] is False,
            'unsafe_shadow')
    require(sha(isolation['image_digest']), 'unpinned_shadow_image')
    for kind, revision in [('baseline', plan['baseline_sha']), ('candidate', candidate_sha)]:
        run = isolation['runs'][kind]
        require(run['revision'] == revision and run['exit_code'] == 0
                and run['report_sha256'] == plan[kind + '_report']['sha256']
                and sha(run['manifest_sha256']), 'shadow_run_mismatch')
    rollback = referenced(bundle, plan['rollback_receipt'])
    require(rollback['baseline_sha'] == plan['baseline_sha'] and rollback['candidate_sha'] == candidate_sha,
            'rollback_revision')
    require(rollback['policy'] == 'preserve-durable-invalidate-derived-v1'
            and rollback['restores_durable_state'] is False, 'unsafe_state_rollback')
    require(sha(rollback['snapshot_manifest_sha256']) and sha(rollback['post_candidate_state_sha256']),
            'missing_rollback_snapshot')
    snapshot = referenced(bundle, plan['state_snapshot'])
    require(plan['state_snapshot']['sha256'] == rollback['snapshot_manifest_sha256']
            and snapshot['schema'] == 'sn68-state-snapshot-v1'
            and snapshot['candidate_sha'] == candidate_sha
            and snapshot['baseline_sha'] == plan['baseline_sha']
            and snapshot['restores_durable_state'] is False
            and bool(snapshot['files']) and all(sha(v) for v in snapshot['files'].values()),
            'snapshot_scope_mismatch')
    resume = isolation['runs']['rollback']
    require(resume['revision'] == plan['baseline_sha'] and resume['exit_code'] == 0
            and resume['resume_state_sha256'] == rollback['post_candidate_state_sha256'],
            'rollback_not_rehearsed')
    rollback_report = referenced(bundle, plan['rollback_report'])
    require(resume['report_sha256'] == plan['rollback_report']['sha256']
            and rollback_report['revision'] == plan['baseline_sha']
            and rollback_report['protected_state_before'] == rollback_report['protected_state_after']
            and bool(rollback_report['protected_state_before']), 'rollback_state_loss')
    compare(fixture, baseline, rollback_report)

    pi = referenced(bundle, plan['pi_receipt'])
    require(pi['schema'] == 'sn68-pi-shadow-v1' and pi['candidate_sha'] == candidate_sha
            and pi['target'] == target and pi['epoch'] == plan['activation_epoch'], 'pi_receipt_scope')
    require(pi['host'] == 'pi' and pi['read_only'] is True and pi['submission_attempts'] == 0,
            'pi_shadow_wrote_submission')
    require(pi['accepted_entries'] >= MIN_FLEET and pi['rejected_entries'] == 0
            and sha(pi['artifact_sha256']) and sha(pi['consumer_sha256']), 'pi_consumer_rejected')
    require(0 <= clock - pi['verified_at'] <= 900, 'pi_receipt_stale')
    require(pi['artifact_sha256'] == candidate['handoff_artifact_sha256'], 'pi_checked_different_artifact')
    require(sha(plan['baseline_manifest_sha256']) and
            plan['baseline_manifest_sha256'] == isolation['runs']['baseline']['manifest_sha256'],
            'baseline_manifest_mismatch')
    return {'status': 'qualified', 'candidate_sha': candidate_sha, 'baseline_sha': plan['baseline_sha'],
            'activation_epoch': plan['activation_epoch'], 'target': target,
            'cases': len(comparison), 'change_kind': plan['change_kind'],
            'baseline_manifest_sha256': plan['baseline_manifest_sha256']}


def verify_restart(root, revision):
    root = Path(root)
    if read(root / 'build_stamp.json').get('deployment_profile') == 'release-controls':
        from scripts.control_release import verify_installed
        return verify_installed(root, revision)
    receipt = read(root / 'artifacts/release_promotion.json')
    require(receipt['status'] == 'active' and receipt['candidate_sha'] == revision
            and receipt['manifest_sha256'] == digest(root / 'build_stamp.json')
            and sha(receipt['qualification_sha256']), 'unqualified_restart')
    from scripts.release_manifest import verify as verify_release
    verify_release(root, read(root / 'build_stamp.json'), expected_sha=revision, runtime_only=True)
    return {'status': 'qualified_restart', 'revision': revision}


def verify_host(bundle, *, runtime_root, candidate_sha, host, target):
    root = Path(runtime_root)
    state = read(root / 'artifacts/chain_state.json')
    require(state['nanobody_target'] == target, 'live_target_changed')
    require(0 <= time.time() - state['updated_at_unix'] <= 90, 'stale_chain_state')
    result = verify(bundle, candidate_sha=candidate_sha, host=host, target=target, epoch=state['epoch'])
    require(state['block'] % state['epoch_length'] < 12, 'not_epoch_boundary')
    require(digest(root / 'build_stamp.json') == result['baseline_manifest_sha256'], 'live_baseline_changed')
    manifest = read(root / 'build_stamp.json')
    require(manifest.get('deployment_profile', 'application') == 'application',
            'application_release_requires_application_baseline')
    from scripts.release_manifest import verify as verify_release
    verify_release(root, manifest, expected_sha=result['baseline_sha'], runtime_only=True, require_tests=False)
    # Do not kill in-flight work to make a release fit its activation window.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            require(not any(x in args for x in [b'nanobody_opt_v2.sequential_exact_race',
                    b'nanobody_opt.sequential_exact_race']), 'active_epoch_work')
        except OSError:
            continue
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--candidate-sha', required=True)
    parser.add_argument('--host')
    parser.add_argument('--target')
    parser.add_argument('--runtime-root', type=Path)
    parser.add_argument('--restart', action='store_true')
    args = parser.parse_args()
    try:
        kwargs = dict(candidate_sha=args.candidate_sha, host=args.host, target=args.target)
        result = (verify_restart(args.runtime_root, args.candidate_sha) if args.restart else
                  verify_host(args.bundle, runtime_root=args.runtime_root, **kwargs)
                  if args.runtime_root else verify(args.bundle, **kwargs))
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        print(json.dumps({'status': 'promotion_refused'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
