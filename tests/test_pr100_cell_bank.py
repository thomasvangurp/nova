"""Storage, use-time re-validation and build planning for the challenge-indexed bank."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nanobody_opt import pr100_cell_bank as bank_module
from nanobody_opt import rules

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'pr100_cell_bank_build', ROOT / 'scripts' / 'pr100_cell_bank_build.py')
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

EPOCH, LENGTH = 25957, 361
V_GENE, CDR3 = 'IGHV3-23', 18
SEQUENCE = 'QVQLVESGGGLVQ'


@pytest.fixture(autouse=True)
def stub_gates(monkeypatch):
    """Keep the gate semantics in rules; stub only their heavy runtime imports."""
    monkeypatch.setattr(rules, 'annotation_reason',
                        lambda annotation, requirements, *, sequence=None, epoch_challenge=None:
                        (annotation or {}).get('reason', ''))
    monkeypatch.setattr(rules, 'developability_reason',
                        lambda result, requirements, *, sequence=None:
                        '' if (result or {}).get('ok') else 'developability_red_flags')


def config():
    return rules.pinned_config(EPOCH)


def challenge(v_gene=V_GENE, cdr3_length=CDR3):
    return bank_module.synthetic_challenge(config(), EPOCH, v_gene, cdr3_length,
                                           epoch_length=LENGTH)


def archive_csv(sequences):
    return ('sequence,target\n' + ''.join(f'{s},Q9NZQ7\n' for s in sequences)).encode()


def profile(psh):
    return {'surface_hydrophobic_patches': psh, 'ok': True}


def record(*, sequence=SEQUENCE, psh=(78.0, 79.0), archive=None, rows=None, prefix=None,
           scores=None, measured_at=None, runtime_extra=None, v_gene=V_GENE,
           cdr3_length=CDR3, annotation=None, now=1_000_000.0):
    view = bank_module.archive_view(archive if archive is not None else archive_csv(['AAA']))
    novelty = {'validator_novel': True, 'exact_historical': False, 'maximum_identity': 0.4,
               'coarse_candidates': 3, 'archive_sha256': view['sha256'],
               'archive_rows': view['rows'] if rows is None else rows,
               'archive_prefix_sha256': prefix if prefix is not None else bank_module.prefix_digest(
                   view['sequences'], view['rows'] if rows is None else rows),
               'checked_at_unix': now}
    runtime = bank_module.measurement_runtime(
        annotation_identity='annot-1', native_runtime='tnp-1', host='master',
        measured_at_unix=now if measured_at is None else measured_at)
    runtime.update(runtime_extra or {})
    return bank_module.build_record(
        target=rules.TARGET, v_gene=v_gene, cdr3_length=cdr3_length, sequence=sequence,
        annotation=annotation if annotation is not None else {'annotation_identity': 'annot-1'},
        measurements=[profile(v) for v in psh], runtime=runtime, novelty=novelty,
        provenance={'source_sha': 'a' * 40}, scores=scores, now=now)


def observation(**over):
    value = {'target': rules.TARGET, 'metrics': {'affinity': 1.0}, 'scorer_sha256': 'b' * 64,
             'settings': {'steps': 35}, 'settings_file_sha256': 'c' * 64,
             'observed_at_unix': 1_000_000.0}
    value.update(over)
    return bank_module.score_observation(**value)


# --- bucket algebra ---------------------------------------------------------------

def test_the_challenge_space_is_171_buckets_and_the_singletons_carry_three_quarters():
    weights = bank_module.bucket_probabilities(config())
    assert len(weights) == 171
    assert sum(weights.values()) == pytest.approx(1.0)
    singletons = sum(w for key, w in weights.items()
                     if key.split(':')[0] in ('IGHV3-23', 'IGHV3-66', 'IGHV3-74'))
    assert singletons == pytest.approx(0.75)
    assert weights['IGHV3-23:18'] == pytest.approx(1 / 76)
    assert weights['IGHV3-7:18'] == pytest.approx(1 / 456)


def test_synthetic_challenge_is_a_real_challenge_for_the_requested_bucket():
    value = challenge('IGHV3-74', 7)
    assert (value['allowed_v'], value['allowed_cdr3_length']) == ('IGHV3-74', 7)
    # Re-derivable from its own hash by the pinned rules, exactly as a worker checks.
    assert rules.challenge(config(), EPOCH, value['epoch_start_block_hash'], LENGTH) == value
    assert bank_module.challenge_bucket(value) == 'IGHV3-74:7'


def test_synthetic_challenge_is_deterministic_per_bucket():
    assert challenge('IGHV3-66', 12) == challenge('IGHV3-66', 12)


# --- storage ----------------------------------------------------------------------

def test_add_and_read_round_trip_with_a_schema_versioned_manifest(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    stored = bank.add(record())
    assert stored.is_file()
    manifest = bank.manifest()
    assert manifest['schema_version'] == bank_module.SCHEMA_VERSION
    rows = bank.records(rules.TARGET, 'IGHV3-23:18')
    assert [r['sequence'] for r in rows] == [SEQUENCE]
    assert rows[0]['psh_mean'] == pytest.approx(78.5)
    assert rows[0]['tier'] == 'A'


def test_a_later_record_supersedes_the_earlier_one_but_the_file_keeps_both(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record())
    bank.add(record(scores=[observation()]))
    path = bank.bucket_file(rules.TARGET, 'IGHV3-23:18')
    assert len(path.read_text().strip().splitlines()) == 2
    rows = bank.records(rules.TARGET, 'IGHV3-23:18')
    assert len(rows) == 1 and len(rows[0]['scores']) == 1


def test_a_torn_trailing_line_is_skipped_rather_than_failing_the_read(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record())
    path = bank.bucket_file(rules.TARGET, 'IGHV3-23:18')
    with open(path, 'a') as handle:
        handle.write('{"record_kind": "pr100-cell')
    assert len(bank.records(rules.TARGET, 'IGHV3-23:18')) == 1


def test_a_foreign_schema_version_is_refused_rather_than_merged(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.ensure()
    bank_module._atomic_json(bank.manifest_path(), {'schema_version': 99})
    with pytest.raises(ValueError, match='schema version'):
        bank.ensure()
    with pytest.raises(ValueError, match='not a current bank record'):
        bank.add({**record(), 'schema_version': 99})


def test_a_record_without_a_measurement_or_provenance_is_never_stored():
    with pytest.raises(ValueError, match='at least one measurement'):
        bank_module.build_record(target=rules.TARGET, v_gene=V_GENE, cdr3_length=CDR3,
                                 sequence=SEQUENCE, annotation={}, measurements=[],
                                 runtime={}, novelty={}, provenance={'source_sha': 'a' * 40})
    with pytest.raises(ValueError, match='source sha'):
        bank_module.build_record(target=rules.TARGET, v_gene=V_GENE, cdr3_length=CDR3,
                                 sequence=SEQUENCE, annotation={}, measurements=[profile(70.0)],
                                 runtime={}, novelty={'validator_novel': True}, provenance={})


def test_bucket_status_counts_tiers_and_weights_coverage_by_probability(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record(scores=[observation()]))
    bank.add(record(sequence='QVQLVESGGGLVA', psh=(85.0, 86.0)))
    bank.add(record(sequence='QVQLVESGGGLVC', v_gene='IGHV3-7', cdr3_length=9))
    status = bank.bucket_status(rules.TARGET, config=config(), now=1_000_000.0)
    assert status['non_empty_buckets'] == 2
    assert status['buckets']['IGHV3-23:18'] == {'bucket_id': 'IGHV3-23:18', 'total': 2,
                                               'tier_a': 1, 'tier_b': 1, 'scored': 1}
    assert status['probability_any'] == pytest.approx(1 / 76 + 1 / 456)
    assert status['possible_buckets'] == 171


# --- use time ---------------------------------------------------------------------

def select(bank, *, archive=None, **over):
    value = challenge()
    return bank.select_for_epoch(value, archive if archive is not None else archive_csv(['AAA']),
                                 now=1_000_000.0, **over)


def test_select_for_epoch_returns_a_tiered_ordered_publishable_list(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record(sequence='QVQLVESGGGLVB', psh=(85.0, 86.0), scores=[observation()]))
    bank.add(record(scores=[observation()]))
    out = select(bank)
    assert [r['tier'] for r in out['candidates']] == ['A', 'B']
    assert out['candidates'][0]['sequence'] == SEQUENCE
    assert out['novelty_checked'] is True and out['novelty_mode'] == 'incremental_watermark'
    assert out['tier_counts'] == {'A': 1, 'B': 1}
    first = out['candidates'][0]
    # Shaped like a native-phase candidate, so the publication path is unchanged.
    assert first['layer2_passed'] is True and first['pr100_evidence']['challenge'] == out['challenge']
    assert first['pr100_evidence']['developability'] is first['measurements'][0]


def test_the_limit_bounds_the_fleet(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    for index in range(4):
        bank.add(record(sequence=SEQUENCE + 'ACDE'[index], scores=[observation()]))
    assert len(select(bank, limit=2)['candidates']) == 2
    with pytest.raises(ValueError, match='limit'):
        select(bank, limit=0)


def test_a_record_from_another_cell_is_dropped_by_live_annotation_revalidation(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    # Stored under this bucket but its annotation no longer matches the live challenge.
    bank.add(record(annotation={'reason': 'wrong_epoch_cdr3_length'}, scores=[observation()]))
    out = select(bank)
    assert out['candidates'] == []
    assert out['rejected'] == [{'sequence_hash': rules.sequence_hash(SEQUENCE),
                                'reason': 'wrong_epoch_cdr3_length'}]


def test_incremental_novelty_only_checks_rows_added_since_the_watermark(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record(archive=archive_csv(['AAA']), scores=[observation()]))
    grown = archive_csv(['AAA', 'BBB', 'CCC'])
    out = select(bank, archive=grown)
    assert len(out['candidates']) == 1
    assert out['candidates'][0]['novelty']['incremental_rows_checked'] == 2
    assert out['candidates'][0]['novelty']['rechecked_rows'] == 3


def test_a_sequence_that_appeared_in_the_new_rows_is_rejected(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record(archive=archive_csv(['AAA']), scores=[observation()]))
    out = select(bank, archive=archive_csv(['AAA', SEQUENCE]))
    assert out['candidates'] == []
    assert out['rejected'][0]['reason'] == 'novelty_not_passed'


def test_a_rewritten_archive_forces_a_full_recheck_instead_of_trusting_the_watermark(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record(archive=archive_csv(['AAA', 'BBB']), scores=[observation()]))
    out = select(bank, archive=archive_csv(['ZZZ', 'BBB', 'CCC']))
    assert out['candidates'] == []
    assert out['rejected'][0]['reason'] == 'novelty_recheck_required'
    shrunk = select(bank, archive=archive_csv(['AAA']))
    assert shrunk['rejected'][0]['reason'] == 'novelty_recheck_required'


def test_the_live_duplicate_gate_is_applied_when_supplied(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record(scores=[observation()]))
    out = select(bank, duplicate_reason=lambda sequence, view: 'layer_1_duplicate_gate_failed')
    assert out['candidates'] == []
    assert out['rejected'][0]['reason'] == 'layer_1_duplicate_gate_failed'


def test_a_stale_tier_a_record_is_downgraded_to_tier_b_not_published_as_margin_safe(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record(measured_at=1_000_000.0 - 40 * 24 * 3600, scores=[observation()]))
    out = select(bank)
    row = out['candidates'][0]
    assert row['stored_tier'] == 'A' and row['tier'] == 'B'
    assert row['margin_safe'] is False and row['downgraded'] is True
    assert out['downgraded_count'] == 1


def test_a_measurement_from_another_runtime_identity_cannot_support_tier_a(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record(scores=[observation()]))
    pinned = select(bank, runtime_identity={'native_runtime': 'tnp-1'})
    assert pinned['candidates'][0]['tier'] == 'A'
    drifted = select(bank, runtime_identity={'native_runtime': 'tnp-2'})
    assert drifted['candidates'][0]['tier'] == 'B'
    assert drifted['candidates'][0]['downgraded'] is True


def test_a_single_repeat_is_never_tier_a(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record(psh=(70.0,), scores=[observation()]))
    assert select(bank)['candidates'][0]['tier'] == 'B'


def test_a_failing_stored_measurement_is_rejected_on_the_gate_reason(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    broken = record(scores=[observation()])
    broken['measurements'][1]['ok'] = False
    bank.add(broken)
    out = select(bank)
    assert out['candidates'] == []
    assert out['rejected'][0]['reason'] == 'developability_red_flags'


def test_an_unscored_record_is_not_published_unless_scores_are_optional(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record())
    assert select(bank)['rejected'][0]['reason'] == 'score_missing'
    assert len(select(bank, require_score=False)['candidates']) == 1


def test_tier_b_is_excluded_when_marginal_candidates_are_not_allowed(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record(psh=(85.0, 86.0), scores=[observation()]))
    assert select(bank, allow_marginal=False)['candidates'] == []


def test_select_for_epoch_refuses_a_challenge_that_does_not_match_the_rules(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    forged = {**challenge(), 'allowed_cdr3_length': 5}
    with pytest.raises(ValueError, match='challenge does not match rules'):
        bank.select_for_epoch(forged, archive_csv(['AAA']))


# --- build planning ---------------------------------------------------------------

def test_build_priority_puts_the_high_mass_buckets_first(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    status = bank.bucket_status(rules.TARGET, config=config(), per_bucket_target=20)
    rows = bank_module.build_priority(config(), status, per_bucket_target=20)
    assert len(rows) == 171
    assert {r['v_gene'] for r in rows[:57]} == {'IGHV3-23', 'IGHV3-66', 'IGHV3-74'}
    assert rows[0]['shortfall'] == 20


def test_a_bucket_already_at_target_leaves_the_plan(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.add(record())
    status = bank.bucket_status(rules.TARGET, config=config(), per_bucket_target=1)
    assert 'IGHV3-23:18' not in {r['bucket_id'] for r in
                                 bank_module.build_priority(config(), status, per_bucket_target=1)}


def test_the_budget_model_reproduces_the_audit_sizing():
    rows = [{'shortfall': 20}] * 57
    estimate = bank_module.budget_estimate(rows, repeats=1)
    # 8_epoch_algorithm.md 3.1: about 977 core-hours and 5.4 GPU-hours for 20 per bucket.
    assert estimate['native_core_hours'] == pytest.approx(977, rel=0.02)
    assert estimate['gpu_hours'] == pytest.approx(5.35, rel=0.02)
    assert bank_module.budget_estimate([{'shortfall': 20}] * 171,
                                       repeats=1)['native_core_hours'] == pytest.approx(2930, rel=0.02)


# --- builder ----------------------------------------------------------------------

def build_args(tmp_path, *extra):
    return builder.parse_args(['--bank', str(tmp_path / 'bank'), '--run-root',
                               str(tmp_path / 'run'), '--dry-run', *extra])


def test_the_dry_run_plans_the_high_mass_buckets_first_and_writes_nothing(tmp_path, capsys):
    assert builder.main(['--bank', str(tmp_path / 'bank'), '--run-root', str(tmp_path / 'run'),
                         '--dry-run', '--max-buckets', '3']) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['dry_run'] is True and len(printed['buckets']) == 3
    assert {b.split(':')[0] for b in printed['buckets']} <= {'IGHV3-23', 'IGHV3-66', 'IGHV3-74'}
    plan = json.loads((tmp_path / 'run' / 'plan.json').read_text())
    assert len(plan['buckets']) == 3
    assert not (tmp_path / 'bank' / 'records').exists()
    assert not builder.progress_path(tmp_path / 'run').exists()


def test_the_core_hour_budget_stops_the_plan(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    one = bank_module.budget_estimate([{'shortfall': 20}], repeats=3)['native_core_hours']
    plan = builder.plan_buckets(bank, config(), build_args(tmp_path, '--core-hour-budget',
                                                           str(one * 2.5)))
    assert len(plan['buckets']) == 2
    assert plan['planned_core_hours'] == pytest.approx(one * 2)


def test_the_gpu_hour_budget_stops_the_plan(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    one = bank_module.budget_estimate([{'shortfall': 20}], repeats=3)['gpu_hours']
    plan = builder.plan_buckets(bank, config(), build_args(tmp_path, '--gpu-hour-budget',
                                                           str(one * 1.5)))
    assert len(plan['buckets']) == 1


def test_an_explicit_bucket_list_restricts_the_plan(tmp_path):
    bank = bank_module.CellBank(tmp_path / 'bank')
    plan = builder.plan_buckets(bank, config(),
                                build_args(tmp_path, '--buckets', 'IGHV3-7:9', 'IGHV3-66:12'))
    assert [r['bucket_id'] for r in plan['buckets']] == ['IGHV3-66:12', 'IGHV3-7:9']


def test_a_real_build_requires_its_pinned_inputs(tmp_path):
    with pytest.raises(SystemExit):
        builder.parse_args(['--bank', str(tmp_path / 'b'), '--run-root', str(tmp_path / 'r')])


def test_the_job_contract_carries_the_worker_keys_for_every_phase(tmp_path):
    args = build_args(tmp_path, '--gpus', '0', '1', '--workers', '32', '--repeats', '2')
    args.references, args.native_tools = '/bank.json', '/tools'
    args.release, args.source_sha = '/release', 'a' * 40
    args.scorer_settings = '/settings.json'
    jobs = builder.bucket_jobs(args, challenge(), tmp_path / 'b', 'd' * 64, 1.0)
    assert set(jobs) == {'sequence', 'native', 'score'}
    assert jobs['sequence']['proposals'] == 600 and jobs['sequence']['references'] == '/bank.json'
    assert jobs['native']['workers'] == 32 and jobs['native']['repeats'] == 2
    assert jobs['native']['allow_marginal'] is True
    assert jobs['score']['gpus'] == [0, 1] and jobs['score']['source_sha'] == 'a' * 40
    for job in jobs.values():
        assert job['archive_sha256'] == 'd' * 64 and job['challenge'] == challenge()


def test_no_gpus_means_a_cpu_only_build_with_no_score_phase(tmp_path):
    args = build_args(tmp_path)
    jobs = builder.bucket_jobs(args, challenge(), tmp_path / 'b', 'd' * 64, 1.0)
    assert 'score' not in jobs


def test_run_bucket_stores_native_passers_and_resumes_a_finished_phase(tmp_path):
    archive = tmp_path / 'archive.csv'
    archive.write_bytes(archive_csv(['AAA']))
    args = build_args(tmp_path, '--archive', str(archive))
    args.references, args.native_tools, args.source_sha = '/bank.json', '/tools', 'a' * 40
    args.native_runtime_identity, args.host_identity = 'tnp-1', 'master'
    bank = bank_module.CellBank(tmp_path / 'bank')
    bank.ensure()
    view = bank_module.archive_view(archive_csv(['AAA']))
    calls = []

    def runner(args_, phase, job_path, directory):
        calls.append(phase)
        job = json.loads(Path(job_path).read_text())
        if phase == 'sequence':
            result = {'challenge': job['challenge'], 'status': 'complete',
                      'candidates': [{'sequence': SEQUENCE, 'sequence_hash': 'x',
                                      'pr100_annotation': {'annotation_identity': 'annot-1'}}]}
        else:
            result = {'challenge': job['challenge'], 'status': 'complete',
                      'archive_sha256': view['sha256'], 'archive_rows': view['rows'],
                      'archive_checked_at_unix': 1.0, 'novelty_checked': True,
                      'candidates': [{'sequence': SEQUENCE,
                                      'pr100_annotation': {'annotation_identity': 'annot-1'},
                                      'measurements': [profile(78.0), profile(79.0)],
                                      'tier': 'A', 'repeats': 2,
                                      'novelty': {'validator_novel': True,
                                                  'exact_historical': False}}]}
        bank_module._atomic_json(Path(directory) / 'result.json', result)
        return {'returncode': 0, 'result': result}

    row = {'bucket_id': 'IGHV3-23:18', 'v_gene': V_GENE, 'cdr3_length': CDR3, 'shortfall': 20}
    outcome = builder.run_bucket(bank, args, row, tmp_path / 'run', runner=runner, now=1_000_000.0)
    assert calls == ['sequence', 'native']
    assert outcome['stored'] == [rules.sequence_hash(SEQUENCE)]
    stored = bank.records(rules.TARGET, 'IGHV3-23:18')
    assert len(stored) == 1 and stored[0]['tier'] == 'A'
    assert stored[0]['novelty']['archive_prefix_sha256'] == bank_module.prefix_digest(
        view['sequences'], view['rows'])
    assert stored[0]['provenance']['synthetic_challenge_hash'].startswith('0x')
    # Resumable: a finished phase is not re-run.
    again = builder.run_bucket(bank, args, row, tmp_path / 'run', runner=runner, now=1_000_000.0)
    assert calls == ['sequence', 'native'] and again['phases'] == {'sequence': 'resumed',
                                                                   'native': 'resumed'}
