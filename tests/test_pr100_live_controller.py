"""The controller's clock arithmetic, phase transport and monotone publication rule."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from nanobody_opt import pr100_live_controller as controller
from nanobody_opt import pr100_live_trial_export, rules

EPOCH, LENGTH = 25957, 361


class Chain:
    """Minimal stand-in for the subtensor reads the controller performs."""

    def __init__(self, block=EPOCH * LENGTH, start_hash='0xstart'):
        self.block, self.start_hash = block, start_hash

    def get_current_block(self):
        return self.block

    def tempo(self, netuid):
        assert netuid == 68
        return LENGTH - 1

    def determine_block_hash(self, block):
        return self.start_hash


def clock(block=EPOCH * LENGTH, *, age=0.0):
    return {'block': block, 'epoch': block // LENGTH, 'epoch_length': LENGTH,
            'updated_at_unix': time.time() - age}


def challenge():
    return {'epoch': EPOCH, 'epoch_length': LENGTH, 'epoch_start_block_hash': '0xstart',
            'target': rules.TARGET, 'allowed_v': 'IGHV3-66', 'allowed_cdr3_length': 12}


def config(tmp_path, **overrides):
    host = {'address': 'host.invalid', 'port': 2222, 'release': '/release',
            'annotation_python': '/py/annotation', 'native_python': '/py/native',
            'scoring_python': '/py/scoring', 'python_path': '/extra', 'path': '/bin',
            'references': '/bank.json', 'scorer': 'registered', 'scorer_settings': '/settings.json'}
    value = {'start_epoch': EPOCH, 'source_sha': 'a' * 40, 'state_root': str(tmp_path / 'state'),
             'publication_path': str(tmp_path / 'nanobody_result.json'),
             'remote_root': '/remote', 'gpu': [0, 1],
             'worker': {'native_tools': '/tools', 'allow_marginal': True},
             'hosts': {'master': dict(host), 'slave': dict(host)}}
    value.update(overrides)
    return value


def candidate(sequence, tier, mean):
    return {'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence), 'tier': tier,
            'psh_mean': mean, 'repeats': 3, 'margin_safe': tier == 'A', 'layer2_passed': True,
            'novelty': {'validator_novel': True}}


# --- clock and deadline arithmetic -------------------------------------------------

def test_clock_state_reads_block_tempo_and_derives_the_epoch():
    state = controller.clock_state(Chain(block=EPOCH * LENGTH + 5))
    assert state['epoch'] == EPOCH and state['epoch_length'] == LENGTH
    assert state['block'] == EPOCH * LENGTH + 5


def test_blocks_remaining_discounts_the_age_of_the_reading():
    fresh = controller.blocks_remaining(clock(), EPOCH)
    assert fresh == pytest.approx(LENGTH, abs=0.01)
    aged = controller.blocks_remaining(clock(age=120), EPOCH)
    assert aged == pytest.approx(LENGTH - 10, abs=0.01)


def test_admissible_uses_the_reserve_and_rejects_other_epochs():
    late = clock(block=EPOCH * LENGTH + LENGTH - 45)
    assert controller.admissible(clock(), EPOCH)
    assert not controller.admissible(late, EPOCH)
    assert not controller.admissible(clock(), EPOCH + 1)
    # A start needs enough time left for the native budget and the novelty pass.
    middle = clock(block=EPOCH * LENGTH + 100)
    assert controller.admissible(middle, EPOCH, reserve=controller.START_MINIMUM_BLOCKS)
    assert not controller.admissible(middle, EPOCH, reserve=300)
    too_late = clock(block=EPOCH * LENGTH + 200)
    assert not controller.admissible(too_late, EPOCH, reserve=controller.START_MINIMUM_BLOCKS)


def test_publication_cutoff_matches_the_block_reserve():
    now = time.time()
    cutoff = controller.publication_cutoff(clock(), EPOCH, now=now)
    expected = (LENGTH - controller.RESERVE_BLOCKS) * controller.BLOCK_SECONDS
    assert cutoff - now == pytest.approx(expected, abs=0.5)
    assert controller.RESERVE_BLOCKS > controller.MINIMUM_RESERVE_BLOCKS


def test_phase_caps_are_shares_of_the_window_and_never_oversubscribe_it():
    now = time.time()
    stop_at = controller.publication_cutoff(clock(), EPOCH, now=now)
    sequence = controller.phase_cap('sequence', stop_at, now=now)
    native = controller.phase_cap('native', stop_at, now=now + sequence)
    score = controller.phase_cap('score', stop_at, now=now + sequence + native)
    assert sequence > 0 and native > 0 and score > 0
    total = sequence + native + score
    assert total <= stop_at - now
    # The whole usable window is far more than the old flat 1800 s budget.
    assert stop_at - now > 3700


def test_phase_cap_refuses_a_window_too_small_for_the_phase():
    now = time.time()
    with pytest.raises(TimeoutError):
        controller.phase_cap('score', now + 100, now=now)


# --- phase requests ----------------------------------------------------------------

def test_phase_request_passes_the_gpu_list_and_worker_knobs_to_the_score_job(tmp_path):
    value = config(tmp_path)
    value['worker'].update(proposals=600, workers=48, repeats=3)
    stop_at = time.time() + 2000
    host, request = controller.phase_request(value, challenge(), 'score', {}, stop_at, clock())
    job = json.loads(bytes.fromhex('') + __import__('base64').b64decode(request['files']['job.json']))
    assert host == 'master'
    assert job['gpus'] == [0, 1] and job['proposals'] == 600 and job['workers'] == 48
    assert request['spec']['environment']['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert request['supervisor'] == '/release/scripts/pr100_live_supervisor.py'
    assert request['spec']['reserve_blocks'] == controller.RESERVE_BLOCKS
    assert 0 < request['spec']['wall_seconds'] <= stop_at - time.time()


def test_native_phase_carries_the_archive_so_the_worker_can_check_novelty(tmp_path):
    import base64
    files = {'input.json': b'{}', 'archive.csv': b'sequence\nAAA\n'}
    host, request = controller.phase_request(config(tmp_path), challenge(), 'native', files,
                                             time.time() + 2000, clock())
    job = json.loads(base64.b64decode(request['files']['job.json']))
    assert host == 'master'
    assert 'archive.csv' in request['files']
    assert job['archive_sha256'] == __import__('hashlib').sha256(files['archive.csv']).hexdigest()
    assert 'archive_fetched_at' in job and job['native_tools'] == '/tools'


def test_sequence_phase_runs_on_the_slave_with_the_reference_bank(tmp_path):
    import base64
    host, request = controller.phase_request(config(tmp_path), challenge(), 'sequence',
                                             {'archive.csv': b'x'}, time.time() + 2000, clock())
    job = json.loads(base64.b64decode(request['files']['job.json']))
    assert host == 'slave' and job['references'] == '/bank.json'


# --- run_phase polling -------------------------------------------------------------

def fake_transport(responses):
    calls = []

    def transport(config_value, host, request):
        calls.append((host, request['operation']))
        return responses[len(calls) - 1]

    transport.calls = calls
    return transport


def test_run_phase_polls_until_the_receipt_appears_and_records_diagnostics(tmp_path):
    import base64
    result = base64.b64encode(json.dumps({'candidates': []}).encode()).decode()
    transport = fake_transport([
        {'running': True},
        {'running': True, 'tails': {'supervisor-stderr.log': 'warming up'}},
        {'running': False, 'receipt': {'reason': 'completed', 'exit_code': 0, 'killed': False,
                                       'elapsed_seconds': 12.0, 'stderr_tail': 'token abcdefgh'},
         'files': {'result.json': result}},
    ])
    diagnostics = {}
    local = tmp_path / 'local'
    local.mkdir()
    files = controller.run_phase(config(tmp_path), challenge(), 'native', {'input.json': b'{}'},
                                 time.time() + 2000, Chain(), local,
                                 transport=transport, sleep=lambda s: None,
                                 diagnostics=diagnostics)
    assert json.loads(files['result.json']) == {'candidates': []}
    assert [c[1] for c in transport.calls] == ['start', 'poll', 'poll']
    assert diagnostics['native']['reason'] == 'completed'
    assert 'abcdefgh' not in diagnostics['native']['stderr_tail']
    assert (local / 'native-supervisor.json').is_file()


def test_run_phase_raises_with_diagnostics_when_a_phase_is_killed_without_a_result(tmp_path):
    transport = fake_transport([
        {'running': False, 'receipt': {'reason': 'wall_seconds', 'exit_code': -9, 'killed': True,
                                       'elapsed_seconds': 900.0, 'stderr_tail': 'boom'},
         'files': {}},
    ])
    local = tmp_path / 'local'
    local.mkdir()
    diagnostics = {}
    with pytest.raises(RuntimeError) as caught:
        controller.run_phase(config(tmp_path), challenge(), 'native', {'input.json': b'{}'},
                             time.time() + 2000, Chain(), local, transport=transport,
                             sleep=lambda s: None, diagnostics=diagnostics)
    assert caught.value.diagnostics['reason'] == 'wall_seconds'
    assert caught.value.diagnostics['stderr_tail'] == 'boom'
    assert diagnostics['native']['killed'] is True


def test_run_phase_refuses_to_launch_after_the_cutoff(tmp_path):
    local = tmp_path / 'local'
    local.mkdir()
    with pytest.raises(TimeoutError):
        controller.run_phase(config(tmp_path), challenge(), 'native', {}, time.time() - 1,
                             Chain(), local, transport=fake_transport([]), sleep=lambda s: None)


# --- scores ------------------------------------------------------------------------

def test_merge_scores_joins_valid_groups_and_rejects_a_changed_target():
    def payload(sequence):
        return json.dumps({'target': rules.TARGET,
                           'per_nanobody_components': {'0': {sequence: {'x': 1.0}}},
                           'structures': {sequence: 'pdb'}}).encode()
    files = {'score/report.json': json.dumps(
        {'groups': [{'length': 120, 'valid': True, 'file': 'gpu-0/length120/exact_all.json'},
                    {'length': 121, 'valid': False, 'file': 'gpu-1/length121/exact_all.json'},
                    {'length': 122, 'valid': True, 'file': 'gpu-1/length122/exact_all.json'}],
         'registered_scorer_sha256': 'c' * 64, 'elapsed_seconds': 42.0}).encode(),
        'score/gpu-0/length120/exact_all.json': payload('AAA'),
        'score/scorer_settings.json': json.dumps({'samples': 5}).encode()}
    merged, settings, scorer = controller.merge_scores(files, rules.TARGET)
    assert merged['n'] == 1 and merged['wall_seconds'] == 42.0
    assert settings == {'samples': 5} and scorer == 'c' * 64

    # A valid group whose file never arrived (partial transfer) is skipped, not fatal.
    files['score/gpu-0/length120/exact_all.json'] = json.dumps(
        {'target': 'OTHER', 'per_nanobody_components': {'0': {}}, 'structures': {}}).encode()
    with pytest.raises(ValueError):
        controller.merge_scores(files, rules.TARGET)
    del files['score/scorer_settings.json']
    with pytest.raises(ValueError, match='no scored group'):
        controller.merge_scores(files, rules.TARGET)


# --- ordering and quality ----------------------------------------------------------

def test_publication_sequence_publishes_margin_safe_first_then_the_upgrade():
    ordered = [candidate('A', 'A', 78.0), candidate('B', 'B', 86.0)]
    attempts = controller.publication_sequence(ordered, allow_marginal=True)
    assert [len(a) for a in attempts] == [1, 2]
    assert controller.publication_sequence(ordered, allow_marginal=False) == [[ordered[0]]]
    only_marginal = [candidate('B', 'B', 86.0)]
    assert controller.publication_sequence(only_marginal) == [only_marginal]


def test_quality_orders_by_tier_then_margin():
    tiers = controller.candidate_tiers([candidate('A', 'A', 78.0), candidate('B', 'B', 86.0),
                                        candidate('C', 'A', 74.0)])
    one = controller.fleet_quality({'sequence_pool': ['A']}, tiers)
    two = controller.fleet_quality({'sequence_pool': ['A', 'B']}, tiers)
    better = controller.fleet_quality({'sequence_pool': ['A', 'C']}, tiers)
    assert controller.quality_key(two) > controller.quality_key(one)
    assert controller.quality_key(better) > controller.quality_key(two)
    assert controller.monotone_decision(one, two) == 'replace'
    assert controller.monotone_decision(two, one) == 'rejected'
    assert controller.monotone_decision(two, two) == 'idempotent'
    assert controller.monotone_decision(None, one) == 'replace'


# --- publication -------------------------------------------------------------------

@pytest.fixture
def publishable(monkeypatch, tmp_path):
    """Stub the export/validation/archive edges so publish() itself is under test."""
    monkeypatch.setattr(controller, 'fetch_archive', lambda target: b'archive')
    digest = __import__('hashlib').sha256(b'archive').hexdigest()

    def build(validity, scores, settings, **kwargs):
        pool = [row['sequence'] for row in validity['candidates']]
        return {'epoch': validity['challenge']['epoch'], 'sequence_pool': pool,
                'sequence_pool_size': len(pool), 'status': 'live_trial'}

    monkeypatch.setattr(pr100_live_trial_export, 'build', build)
    monkeypatch.setattr(rules, 'verified_fleet',
                        lambda fleet, *a, **k: {'sequence_pool': fleet['sequence_pool']})
    return digest


def publish_once(value, local, rows, tiers, *, dry_run=False):
    validity = {'challenge': challenge(), 'candidates': rows,
                'archive_sha256': __import__('hashlib').sha256(b'archive').hexdigest()}
    return controller.publish(value, Chain(), challenge(), validity, {'target': rules.TARGET},
                              {'samples': 5}, 'c' * 64, 'd' * 64, local,
                              tiers=tiers, dry_run=dry_run)


def test_publish_writes_the_fleet_and_records_its_quality(tmp_path, publishable):
    value = config(tmp_path)
    local = tmp_path / 'local'
    local.mkdir()
    rows = [candidate('AAA', 'A', 78.0)]
    tiers = controller.candidate_tiers(rows)
    outcome = publish_once(value, local, rows, tiers)
    assert outcome['status'] == 'published' and outcome['count'] == 1
    assert outcome['tier_a_count'] == 1 and outcome['published'] is True
    assert json.loads(Path(value['publication_path']).read_text())['sequence_pool'] == ['AAA']
    assert controller.published_quality(local, EPOCH)['count'] == 1


def test_a_later_better_fleet_replaces_the_published_one(tmp_path, publishable):
    value = config(tmp_path)
    local = tmp_path / 'local'
    local.mkdir()
    rows = [candidate('AAA', 'A', 78.0), candidate('BBB', 'A', 74.0)]
    tiers = controller.candidate_tiers(rows)
    publish_once(value, local, rows[:1], tiers)
    upgrade = publish_once(value, local, rows, tiers)
    assert upgrade['decision'] == 'replace' and upgrade['count'] == 2
    assert json.loads(Path(value['publication_path']).read_text())['sequence_pool'] == ['AAA', 'BBB']


def test_a_worse_fleet_is_rejected_and_an_equal_one_is_idempotent(tmp_path, publishable):
    value = config(tmp_path)
    local = tmp_path / 'local'
    local.mkdir()
    rows = [candidate('AAA', 'A', 78.0), candidate('BBB', 'A', 74.0)]
    tiers = controller.candidate_tiers(rows)
    publish_once(value, local, rows, tiers)
    with pytest.raises(ValueError, match='degrade'):
        publish_once(value, local, [candidate('CCC', 'B', 86.0)], tiers)
    again = publish_once(value, local, rows, tiers)
    assert again['decision'] == 'idempotent' and again['count'] == 2
    assert json.loads(Path(value['publication_path']).read_text())['sequence_pool'] == ['AAA', 'BBB']


def test_a_tier_a_member_outranks_a_larger_marginal_fleet(tmp_path, publishable):
    value = config(tmp_path)
    local = tmp_path / 'local'
    local.mkdir()
    marginal = [candidate('B1', 'B', 86.0), candidate('B2', 'B', 87.0)]
    safe = marginal + [candidate('A1', 'A', 77.0)]
    tiers = controller.candidate_tiers(safe)
    publish_once(value, local, marginal, tiers)
    outcome = publish_once(value, local, safe, tiers)
    assert outcome['decision'] == 'replace' and outcome['tier_a_count'] == 1


def test_an_unrecorded_previous_fleet_still_cannot_be_silently_discarded(tmp_path, publishable):
    value = config(tmp_path)
    local = tmp_path / 'local'
    local.mkdir()
    # A fleet published by another producer: no quality sidecar exists for it.
    Path(value['publication_path']).write_text(json.dumps({'epoch': EPOCH, 'sequence_pool': ['ZZZ']}))
    rows = [candidate('AAA', 'A', 78.0)]
    with pytest.raises(ValueError, match='discard'):
        publish_once(value, local, rows, controller.candidate_tiers(rows))


def test_dry_run_validates_everything_but_never_writes_the_destination(tmp_path, publishable):
    value = config(tmp_path)
    local = tmp_path / 'local'
    local.mkdir()
    rows = [candidate('AAA', 'A', 78.0)]
    outcome = publish_once(value, local, rows, controller.candidate_tiers(rows), dry_run=True)
    assert outcome['status'] == 'dry_run' and outcome['published'] is False
    assert not Path(value['publication_path']).exists()
    assert (local / 'dry-run-fleet.json').is_file()
    assert controller.published_quality(local, EPOCH) is None


def test_publish_refuses_after_the_cutoff_and_on_a_changed_epoch_start(tmp_path, publishable):
    value = config(tmp_path)
    local = tmp_path / 'local'
    local.mkdir()
    rows = [candidate('AAA', 'A', 78.0)]
    tiers = controller.candidate_tiers(rows)
    validity = {'challenge': challenge(), 'candidates': rows,
                'archive_sha256': __import__('hashlib').sha256(b'archive').hexdigest()}
    late = Chain(block=EPOCH * LENGTH + LENGTH - 10)
    with pytest.raises(TimeoutError):
        controller.publish(value, late, challenge(), validity, {'target': rules.TARGET},
                           {'samples': 5}, 'c' * 64, 'd' * 64, local, tiers=tiers)
    with pytest.raises(ValueError, match='epoch start'):
        controller.publish(value, Chain(start_hash='0xother'), challenge(), validity,
                           {'target': rules.TARGET}, {'samples': 5}, 'c' * 64, 'd' * 64,
                           local, tiers=tiers)


# --- config validation and build stamp ---------------------------------------------

def stamped_release(tmp_path, value, root=None):
    import hashlib
    import importlib
    root = Path(root or Path(controller.__file__).resolve().parents[1])
    release = tmp_path / 'release'
    module = importlib.import_module('importlib.util')
    spec = module.spec_from_file_location('pr100_stamp_release',
                                          root / 'scripts' / 'pr100_stamp_release.py')
    stamper = module.module_from_spec(spec)
    spec.loader.exec_module(stamper)
    for name in stamper.CONTROLLER_FILES:
        target = release / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    stamper.stamp(release, git_sha=value['source_sha'])
    return release


def test_build_stamp_output_is_accepted_by_the_controller(tmp_path):
    value = config(tmp_path)
    release = stamped_release(tmp_path, value)
    stamp = controller.verify_build_stamp(value, root=release)
    assert stamp['git_sha'] == value['source_sha']
    assert 'scripts/pr100_live_supervisor.py' in stamp['files']
    # The remote transport performs the same two checks, so drift must be detected.
    (release / 'nanobody_opt/pr100_live_controller.py').write_text('# tampered\n')
    with pytest.raises(ValueError, match='drift'):
        controller.verify_build_stamp(value, root=release)


def test_build_stamp_revision_mismatch_is_rejected(tmp_path):
    value = config(tmp_path)
    release = stamped_release(tmp_path, value)
    with pytest.raises(ValueError, match='revision'):
        controller.verify_build_stamp(dict(value, source_sha='b' * 40), root=release)


@pytest.mark.parametrize('mutation, message', [
    ({'start_epoch': 1}, 'start_epoch'),
    ({'gpu': 1}, 'gpu must be'),
    ({'gpu': []}, 'gpu must be'),
    ({'gpu': [0, 0]}, 'duplicate'),
    ({'state_root': 'relative/path'}, 'state_root'),
    ({'publication_path': 'relative.json'}, 'publication_path'),
    ({'remote_root': 'remote'}, 'remote_root'),
    ({'source_sha': 'nothex'}, 'source_sha'),
    ({'worker': {}}, 'native_tools'),
    ({'reserve_blocks': 10}, 'reserve_blocks'),
])
def test_validate_config_reports_a_clear_error(tmp_path, monkeypatch, mutation, message):
    value = config(tmp_path, **mutation)
    monkeypatch.setattr(controller, 'verify_build_stamp', lambda c, root=None: {})
    with pytest.raises(ValueError, match=message):
        controller.validate_config(value)


def test_validate_config_requires_both_hosts_and_their_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'verify_build_stamp', lambda c, root=None: {})
    value = config(tmp_path)
    del value['hosts']['slave']
    with pytest.raises(ValueError, match='master and slave'):
        controller.validate_config(value)
    value = config(tmp_path)
    del value['hosts']['master']['scorer']
    with pytest.raises(ValueError, match='missing: scorer'):
        controller.validate_config(value)
    value = config(tmp_path)
    value['hosts']['master']['port'] = 0
    with pytest.raises(ValueError, match='tcp port'):
        controller.validate_config(value)
    value = config(tmp_path)
    del value['source_sha']
    with pytest.raises(ValueError, match='missing required keys'):
        controller.validate_config(value)


def test_the_shipped_example_config_has_the_schema_and_no_real_endpoints():
    path = Path(controller.__file__).resolve().parents[1] / 'config/pr100_live_controller.example.json'
    value = json.loads(path.read_text())
    assert {'start_epoch', 'hosts', 'remote_root', 'state_root', 'publication_path',
            'source_sha', 'worker', 'gpu'} <= set(value)
    assert isinstance(value['gpu'], list) and value['gpu']
    for name in ('master', 'slave'):
        assert set(controller.HOST_KEYS) <= set(value['hosts'][name])
        assert 'PLACEHOLDER' in value['hosts'][name]['address']
    assert 'PLACEHOLDER' in value['source_sha']
    assert value['hosts']['master']['port'] == 22
    assert 'native_tools' in value['worker']


def test_sanitize_bounds_and_redacts_diagnostic_text():
    text = 'Authorization: Bearer ' + 'z' * 60 + '\nplain line'
    cleaned = controller.sanitize(text)
    assert 'zzzz' not in cleaned and '<redacted>' in cleaned
    assert 'plain line' in cleaned
    assert len(controller.sanitize('y' * 9000)) <= controller.TAIL_BYTES
