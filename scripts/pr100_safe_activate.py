"""Run on Pi: stamped release handoff at a verified continuation boundary.

Usage: python3 SCRIPT SHA preserve --staging-receipt PATH --novelty-qualification-receipt PATH
Both mode additionally requires --qualification-receipt PATH.
All subprocess output is captured; only allowlisted operational status is emitted.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import sys

# The activation runs as a bare `python scripts/pr100_safe_activate.py` from a
# systemd unit whose cwd is not the release, so `nanobody_opt` is not on
# sys.path by default. Add the release root so in-process imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OLD = 'b294f7f06b6f3c1efd05961f9a1365645af60410'
THROUGHPUT_RELEASE = '036a6c48c32b762b3f803ffa920eaf3cf66d454a'
QUALIFIED_BOTH_RELEASE = '22407a4ae6324b8c051ef89f738eb74fddaad978'
QUALIFIED_NOVELTY_RELEASE = '4cec2deeca89dfa1a22ad65477fc631eeaff63a8'
ARCHIVE_CACHE_RELEASE = '98ef1cba2511bf71786d9dd7a3e732828272d154'
TERMINAL_INVALIDATION_RELEASE = 'd9b13d773651066b396fcd52835b9f48456b6408'
PRODUCTION_YIELD_RELEASE = 'da348c997ebe4892742a39961a9782d9f2130547'
FORCE_ACTIVATION_RELEASE = 'd48b6be549210e94088c7951ed53850f3fa352cd'
RECOVERY_RELEASE = '880619c539449cfff1bfc446152c6e3e129e8619'
APPROVED_PREDECESSORS = frozenset({OLD, THROUGHPUT_RELEASE, QUALIFIED_BOTH_RELEASE,
                                    QUALIFIED_NOVELTY_RELEASE, ARCHIVE_CACHE_RELEASE,
                                    TERMINAL_INVALIDATION_RELEASE, PRODUCTION_YIELD_RELEASE,
                                    FORCE_ACTIVATION_RELEASE, RECOVERY_RELEASE,
                                    'ca091f95553ec4302c2221f703883bd35488795d',
                                    'bcac5ba7895886aca5e2671c6d574e197ae58048',
                                    '3a26afe5f92e6fa9c368bb0b79eb2466bcca5728',
                                    '1d44712661a57a1b8e459824584bc6da646a59a4'})
BASE = Path('/home/thomasg')
PYTHON = '/opt/nova-submit/.pixi/envs/default/bin/python'
UNIT_NAME = 'nova-pr100-controller.service'
UNIT = Path('/etc/systemd/system') / UNIT_NAME
NATIVE_PREFIX = '/root/.pixi/envs/metanano'
NATIVE = {'native_python': NATIVE_PREFIX + '/bin/python3.12',
          'native_path': NATIVE_PREFIX + '/bin:/usr/local/bin:/usr/bin:/bin',
          'native_python_path': '/root/metanano_shim'}
ALTERNATE_PREFIX = '/root/pr100-native-alt-20261027'
NATIVE_WRAPPER_SHA256 = 'ab23f3e807a65753207d219a323d567924f9d00c8633c6cad811981dcd6e02e8'
NATIVE_ALTERNATE = {'native_python': ALTERNATE_PREFIX + '/bin/native-python',
                    'native_path': ALTERNATE_PREFIX + '/bin:/usr/local/bin:/usr/bin:/bin',
                    'native_python_path': ALTERNATE_PREFIX + '/shim'}
NOVELTY_PREFIX = '/root/pr100-novelty-runtime-20261027'
NOVELTY_PATHS = {'python': NOVELTY_PREFIX + '/bin/novelty-python',
                'path': NOVELTY_PREFIX + '/bin:/usr/local/bin:/usr/bin:/bin',
                'python_path': NOVELTY_PREFIX + '/shim'}
NOVELTY_SOURCES = frozenset({
    'nanobody_opt/check_validator_novelty.py', 'nanobody_opt/pr100_novelty_runtime.py',
    'NOVA-nanobody-filter/metanano/config.py',
    'NOVA-nanobody-filter/metanano/search/search_engine.py',
    'NOVA-nanobody-filter/metanano/search/index_manager.py',
    'NOVA-nanobody-filter/metanano/utils/alignment.py',
    'NOVA-nanobody-filter/metanano/utils/cdr_utils.py',
    'NOVA-nanobody-filter/metanano/utils/kmer.py'})
NOVELTY_ARCHIVES = {
    'old': ('2e426028e1bb34b0d14ba17759c539fc91f7a825c9b7e5fe4f9ec5c3618ba260', 8553),
    'new': ('faa66bc72600e2fd4979754ab02c2637ef7d8725ee487fb5cd48cfa260b7eb2e', 8575)}
NOVELTY_INPUT_SHA256 = '47940cc481d43ab28650f1c4cfc0d1155426973ec764337d1dec4e849d887905'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def hash_value(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def verified_novelty_runtimes(value):
    if not isinstance(value, dict) or set(value) != {'master', 'slave'}:
        raise ValueError('both_novelty_runtimes_required')
    for runtime in value.values():
        if (not isinstance(runtime, dict) or set(runtime) != set(NOVELTY_PATHS) | {'identity_sha256'}
                or any(runtime.get(key) != path for key, path in NOVELTY_PATHS.items())
                or not hash_value(runtime.get('identity_sha256'))):
            raise ValueError('unqualified_novelty_runtime_layout')
    return copy.deepcopy(value)


def verify_novelty_qualification(release, path):
    """Require complete two-host replay and bind scientific sources to the release."""
    value = read(path, {}) if path else {}
    if (not isinstance(value, dict) or value.get('schema') != 'pr100-novelty-runtime-qualification-v1'
            or value.get('artifact_hash_encoding') != 'canonical-json-v1'
            or not isinstance(value.get('hosts'), dict) or set(value['hosts']) != {'master', 'slave'}):
        raise ValueError('both_host_novelty_qualification_required')
    if any(not isinstance(item, dict) for item in value['hosts'].values()):
        raise ValueError('novelty_host_receipt_invalid')
    runtimes = verified_novelty_runtimes({host: item.get('runtime') for host, item in value['hosts'].items()})
    cohort = None
    for host, item in value['hosts'].items():
        identity, result, supervisor = item.get('identity', {}), item.get('result', {}), item.get('supervisor', {})
        if (not all(isinstance(obj, dict) for obj in (identity, result, supervisor))
                or item.get('artifact_sha256') != {name: digest(item.get(name)) for name in ('identity', 'result', 'supervisor')}):
            raise ValueError('novelty_artifact_digest_mismatch')
        runtime_hash = runtimes[host]['identity_sha256']
        if (not isinstance(identity, dict) or identity.get('schema') != 'pr100-qualified-novelty-v1'
                or digest(identity) != runtime_hash or not hash_value(identity.get('python_binary_sha256'))
                or not hash_value(identity.get('execution_environment_sha256'))
                or not hash_value(identity.get('annotation_identity'))
                or not isinstance(identity.get('sources'), dict) or set(identity['sources']) != NOVELTY_SOURCES
                or not isinstance(identity.get('assets'), dict)
                or set(identity['assets']) != {'assets-manifest.json', 'pydantic-assets-manifest.json', 'hmmer-assets-manifest.json'}
                or not all(hash_value(v) for v in identity['assets'].values())):
            raise ValueError('novelty_runtime_identity_unverified')
        for name, expected in identity['sources'].items():
            source = (Path(release) / name).resolve()
            if (not hash_value(expected) or not source.is_relative_to(Path(release).resolve())
                    or hashlib.sha256(source.read_bytes()).hexdigest() != expected):
                raise ValueError('novelty_scientific_source_mismatch')
        cleanup = supervisor.get('cleanup', {})
        if not isinstance(cleanup, dict):
            raise ValueError('novelty_owner_cleanup_unverified')
        if (supervisor.get('reason') != 'completed' or type(supervisor.get('exit_code')) is not int
                or supervisor['exit_code'] != 0 or type(cleanup.get('remaining_descendants')) is not int
                or cleanup['remaining_descendants'] != 0 or cleanup.get('fixed_point_reached') is not True):
            raise ValueError('novelty_owner_cleanup_unverified')
        if (result.get('status') != 'complete' or result.get('runtime_sha256') != runtime_hash
                or type(result.get('input_count')) is not int or result['input_count'] != 22
                or result.get('input_sha256') != NOVELTY_INPUT_SHA256
                or not isinstance(result.get('snapshots'), dict) or set(result['snapshots']) != set(NOVELTY_ARCHIVES)):
            raise ValueError('novelty_replay_incomplete_or_unbound')
        for name, (archive_hash, count) in NOVELTY_ARCHIVES.items():
            snapshot = result['snapshots'][name]
            if (not isinstance(snapshot, dict) or snapshot.get('archive_sha256') != archive_hash or type(snapshot.get('archive_rows')) is not int
                    or snapshot['archive_rows'] != count or type(snapshot.get('archive_cdr_available')) is not int
                    or snapshot['archive_cdr_available'] != count or type(snapshot.get('kept_count')) is not int
                    or snapshot['kept_count'] != 22 or not isinstance(snapshot.get('verdicts'), list)
                    or len(snapshot['verdicts']) != 22):
                raise ValueError('novelty_full_archive_coverage_unverified')
            hashes = []
            for verdict in snapshot['verdicts']:
                if not isinstance(verdict, dict):
                    raise ValueError('novelty_positive_verdict_unverified')
                score = verdict.get('maximum_identity')
                if (not hash_value(verdict.get('sequence_hash')) or verdict.get('validator_novel') is not True
                        or verdict.get('exact_historical') is not False
                        or type(verdict.get('coarse_candidates')) is not int or verdict['coarse_candidates'] < 0
                        or isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError('novelty_positive_verdict_unverified')
                hashes.append(verdict['sequence_hash'])
            if len(set(hashes)) != 22 or cohort is not None and hashes != cohort:
                raise ValueError('novelty_replay_cohort_mismatch')
            cohort = hashes
    return runtimes


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic(path, value, *, raw=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(value if raw else json.dumps(value, allow_nan=False).encode())
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try: os.unlink(name)
        except FileNotFoundError: pass


def ready_for_activation(clock, heartbeat, state, now, *, heartbeat_max_age=30, recoverable_attempt=False):
    """A receipt says nothing about current continuation ownership."""
    if not all(isinstance(v, dict) for v in (clock, heartbeat, state)):
        return False
    epoch = clock.get('epoch'); length = clock.get('epoch_length'); block = clock.get('block')
    # Deployment itself performs no scientific work. The controller enforces
    # submission/admission cutoffs; the quiet gap is a useful handoff window.
    return (type(epoch) is int and type(length) is int and length > 0 and type(block) is int
        and block // length == epoch
        and heartbeat.get('epoch') == epoch and heartbeat.get('status') == 'watching'
        and isinstance(heartbeat.get('updated_at_unix'), (float, int))
        and 0 <= now - heartbeat['updated_at_unix'] <= heartbeat_max_age
        and state.get('challenge', {}).get('epoch') == epoch
        and 'active' in state and (state['active'] is None or recoverable_attempt)
        and not state.get('legacy_uncertain_phase'))


def clean_pending_novelty_migration(config, clock, state):
    """A preserved pending attempt is migratable only after all owners are clean.

    Native launch intent is persisted before any remote start. Empty host intent
    directories therefore do not establish a live owner, despite a conservative
    aggregate ownership_unresolved flag after pre-dispatch failure.
    """
    active = state.get('active')
    if (not isinstance(active, dict) or type(active.get('number')) is not int or active['number'] < 1
            or active.get('phase') not in ('native', 'score', 'novelty', 'publish')):
        return False
    directory = Path(config['state_root']) / str(clock['epoch']) / 'attempts' / str(active['number']).zfill(5)
    plan = read(directory / 'native-dispatch.json', {})
    if (not isinstance(plan, dict) or plan.get('challenge') != state.get('challenge')
            or plan.get('source_sha') != config['source_sha']
            or set(plan.get('hosts', [])) != set(config.get('native_hosts', ['master']))):
        return False
    requests = [directory / (phase + '-request.json') for phase in ('sequence', 'native', 'score')]
    requests += [directory / 'native-hosts' / host / 'native-request.json' for host in ('master', 'slave')]
    for group, phase in (('novelty-rechecks', 'novelty'), ('score-batches', 'score')):
        requests.extend((directory / group).glob('*/' + phase + '-request.json'))
    for intent in requests:
        if not intent.is_file():
            continue
        saved = read(intent, {}); request = saved.get('request', {})
        phase = intent.name.removesuffix('-request.json')
        supervisor = read(intent.parent / (phase + '-supervisor.json'), {})
        cleanup = supervisor.get('cleanup', {})
        if (request.get('spec', {}).get('epoch') != clock['epoch']
                or saved.get('host') not in ('master', 'slave')
                or not (intent.parent / (phase + '-collected.json')).is_file()
                or type(cleanup.get('remaining_descendants')) is not int
                or cleanup['remaining_descendants'] != 0 or cleanup.get('fixed_point_reached') is not True):
            return False
    return True


def qualification_identity_verified(receipt):
    """Application scheduling revisions may differ; scientific identity may not drift silently."""
    actual = receipt.get('runtime_identity', {})
    reference = receipt.get('reference_runtime_identity', {})
    source = receipt.get('application_source_sha', receipt.get('source_sha', ''))
    if not isinstance(source, str) or not re.fullmatch('[0-9a-f]{40}', source):
        return False
    def valid(identity):
        if not isinstance(identity, dict) or not re.fullmatch('[0-9a-f]{64}', str(identity.get('manifest_sha256', ''))):
            return False
        for name in ('scientific_sources', 'model_weights', 'tools'):
            values = identity.get(name)
            if not isinstance(values, dict) or not values or not all(
                    isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in values.values()):
                return False
        return len(identity['model_weights']) >= 4
    if not valid(actual) or not valid(reference):
        return False
    if actual == reference:
        return True
    approval = receipt.get('root_runtime_difference_approval', {})
    identity_hash = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return (approval.get('approved') is True and approval.get('runtime_identity_sha256') == identity_hash
            and isinstance(receipt.get('runtime_differences'), list) and bool(receipt['runtime_differences']))


def qualified_native_paths(receipt):
    runtime = receipt.get('runtime', {})
    for allowed in (NATIVE, NATIVE_ALTERNATE):
        if all(runtime.get(k) == v for k, v in allowed.items()):
            if allowed is NATIVE_ALTERNATE and receipt.get('runtime_identity', {}).get('tools', {}).get(
                    'native_python_wrapper') != NATIVE_WRAPPER_SHA256:
                raise ValueError('native_wrapper_identity_unverified')
            if allowed is NATIVE_ALTERNATE and runtime.get('native_tools') != ALTERNATE_PREFIX + '/bin':
                raise ValueError('alternate_native_tools_path_mismatch')
            return dict(allowed)
    raise ValueError('unapproved_native_runtime_layout')


def clean_qualification(receipt):
    """The guard must have finished successfully and reaped every owned child."""
    if not isinstance(receipt, dict):
        return False
    cleanup = receipt.get('cleanup')
    return (type(receipt.get('returncode')) is int and receipt['returncode'] == 0
            and type(receipt.get('runtime_inconclusive_count')) is int
            and receipt['runtime_inconclusive_count'] == 0
            and isinstance(cleanup, dict)
            and type(cleanup.get('remaining_descendants')) is int
            and cleanup['remaining_descendants'] == 0
            and cleanup.get('fixed_point_reached') is True)


def verify_prerequisites(sha, release, staging_path, *, mode, qualification_path=None,
                         interleave_confirmations=False, native_refill_hosts=(), missing_cell_donors=False):
    stamp = read(release / 'build_stamp.json', {})
    check = stamp.get('validation', {})
    if (stamp.get('git_sha') != sha or check.get('passed') is not True
            or check.get('failures') != 0 or check.get('errors') != 0
            or check.get('skipped') != 0 or check.get('tests', 0) <= 0):
        raise ValueError('shipping_ci_not_verified')
    expected_tests = {'tests/test_pr100_native_throughput.py', 'tests/test_pr100_native_dispatch_recovery.py'}
    if sha != THROUGHPUT_RELEASE:
        expected_tests.add('tests/test_pr100_fresh_native_budget.py')
        expected_tests.add('tests/test_pr100_native_execution_identity.py')
    if interleave_confirmations:
        expected_tests.add('tests/test_pr100_confirmation_deadline_review.py')
    feature_tests = set()
    if native_refill_hosts:
        feature_tests.update({'tests/test_pr100_native_refill.py',
                              'tests/test_pr100_native_lifecycle.py',
                              'tests/test_pr100_safe_activate.py'})
    if missing_cell_donors:
        feature_tests.update({'tests/test_pr100_reference_donors.py',
                              'tests/test_pr100_safe_activate.py'})
    expected_tests.update(feature_tests)
    if not expected_tests <= set(check.get('test_files', [])):
        raise ValueError('throughput_shipping_tests_missing')
    manifest = stamp.get('files')
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('empty_release_manifest')
    if not feature_tests <= set(manifest):
        raise ValueError('rollout_feature_suites_not_staged')
    if interleave_confirmations and 'tests/test_pr100_confirmation_deadline_review.py' not in manifest:
        raise ValueError('confirmation_deadline_suite_not_staged')
    for name, digest in manifest.items():
        relative = Path(name)
        forbidden = {'.env', '.netrc', '.npmrc', '.git', '.ssh', '.aws', '.azure', 'credentials', 'cookies'}
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                or any(part in forbidden or part.startswith('.env.') for part in relative.parts)
                or relative.suffix in ('.pem', '.key')
                or not (release / name).resolve().is_relative_to(release.resolve())):
            raise ValueError('invalid_release_manifest_path')
        if hashlib.sha256((release / name).read_bytes()).hexdigest() != digest:
            raise ValueError('release_manifest_mismatch')
    stage = read(staging_path, {})
    hosts = stage.get('hosts', [])
    if (stage.get('source_sha') != sha or len(hosts) != 3
            or {h.get('host') for h in hosts} != {'pi', 'master', 'slave'}
            or not all(h.get('source_sha') == sha and h.get('manifest_files') == len(manifest)
                       and h.get('mismatch_files') == [] for h in hosts)):
        raise ValueError('three_host_staging_not_verified')
    if mode == 'both':
        q = read(qualification_path, {}) if qualification_path else {}
        if (q.get('schema') != 'pr100-native-runtime-qualification-v1' or q.get('status') != 'passed'
                or q.get('host') != 'slave' or not qualification_identity_verified(q)
                or not clean_qualification(q)
                or q.get('runtime_manifest_verified') is not True or q.get('cpu_affinity_verified') is not True
                or type(q.get('workers')) is not int or q['workers'] < 32
                or type(q.get('completed_profiles')) is not int or q['completed_profiles'] < 32
                or not str(q.get('runtime', {}).get('native_tools', '')).startswith('/')):
            raise ValueError('slave_runtime_not_qualified')
        qualified_native_paths(q)
        return q
    return None


def selected_mode(current, mode):
    existing = current.get('native_hosts', ['master'])
    if mode == 'preserve':
        if existing == ['master']:
            return 'master'
        if set(existing) == {'master', 'slave'} and len(existing) == 2:
            return 'both'
        raise ValueError('cannot_infer_existing_native_mode')
    if mode == 'master' and 'slave' in existing:
        raise ValueError('code_rollout_cannot_disable_qualified_slave')
    return mode


def enable_native_screen(value, backend):
    """Enable the non-authoritative native developability pre-filter.

    Screen mode must not carry a CPU-interleaving confirmation policy or any
    per-host work-conserving refill flag: the live controller treats those as
    mutually exclusive operating modes.
    """
    value['worker']['native_screen'] = {**value['worker'].get('native_screen', {}),
                                      'enabled': True, 'backend': backend}
    if value['worker'].get('native_interleave_confirmations'):
        value['worker']['native_interleave_confirmations'] = False
    for override in value.get('native_worker_overrides', {}).values():
        override.pop('native_work_conserving', None)
    return value


def rollout_options(value, *, native_refill_hosts=(), missing_cell_donors=False):
    """Enable explicitly selected policies without widening worker allocation."""
    if (not isinstance(native_refill_hosts, (tuple, list))
            or any(host not in ('master', 'slave') for host in native_refill_hosts)
            or len(set(native_refill_hosts)) != len(native_refill_hosts)):
        raise ValueError('invalid_native_refill_hosts')
    if type(missing_cell_donors) is not bool:
        raise ValueError('missing_cell_donors_must_be_boolean')
    worker = value['worker']
    if native_refill_hosts:
        if (worker.get('native_interleave_confirmations') is not True
                or worker.get('native_screen', {}).get('enabled', False)):
            raise ValueError('refill_requires_original_cpu_interleaving')
        if not set(native_refill_hosts) <= set(value.get('native_hosts', ['master'])):
            raise ValueError('refill_host_not_enabled')
        for host in native_refill_hosts:
            override = value.setdefault('native_worker_overrides', {}).setdefault(host, {})
            override['native_work_conserving'] = True
    if missing_cell_donors:
        worker['missing_cell_donor_fallback'] = True
    return value


def recovery_rollout(value, receipt, sha):
    """Apply only source-bound, measured recovery options to a candidate config.

    This runs before service validation and the final boundary/race checks.
    It never changes a live config, continuation, or scientific evidence file.
    """
    if receipt is None:
        return value
    allowed = {'schema', 'source_sha', 'native_screen_hosts', 'reference_bank', 'edit_generation'}
    if (not isinstance(receipt, dict) or set(receipt) - allowed
            or receipt.get('schema') != 'pr100-recovery-rollout-v1'
            or receipt.get('source_sha') != sha):
        raise ValueError('invalid_recovery_rollout_identity')
    candidate = copy.deepcopy(value)

    def positive(number):
        return type(number) is int and number > 0

    def clean(proof):
        cleanup = proof.get('cleanup', {})
        return (isinstance(cleanup, dict) and type(cleanup.get('remaining_descendants')) is int
                and cleanup['remaining_descendants'] == 0 and cleanup.get('fixed_point_reached') is True)

    def bound(proof):
        return isinstance(proof, dict) and proof.get('source_sha') == sha and proof.get('status') == 'complete'

    hosts = receipt.get('native_screen_hosts', {})
    if (not isinstance(hosts, dict)
            or set(hosts) - set(candidate.get('native_hosts', ['master']))):
        raise ValueError('invalid_recovery_screen_hosts')
    if hosts:
        from nanobody_opt import pr100_native_screen as screen
        keys = {'enabled', 'backend', 'workers_per_gpu', 'workers', 'gpu_minimum_free_mib',
                'gpus', 'runtime_versions'}
        for host, entry in hosts.items():
            supplied = entry.get('settings') if isinstance(entry, dict) else None
            if (not isinstance(entry, dict) or set(entry) - {'settings', 'smoke'}
                    or not isinstance(supplied, dict) or set(supplied) - keys):
                raise ValueError('invalid_recovery_screen_settings')
            override = candidate.setdefault('native_worker_overrides', {}).setdefault(host, {})
            worker = {**candidate['worker'], **override, 'native_screen': supplied}
            normalized = screen.settings(worker)
            if normalized['workers'] > worker.get('cpu_slots', worker.get('workers', 48)):
                raise ValueError('recovery_screen_exceeds_cpu_allocation')
            if normalized['enabled']:
                smoke = entry.get('smoke', {})
                if (not bound(smoke) or not clean(smoke)
                        or smoke.get('host') != host
                        or not positive(smoke.get('measured_count'))
                        or type(smoke.get('runtime_inconclusive_count')) is not int
                        or smoke['runtime_inconclusive_count'] != 0
                        or smoke.get('settings_sha256') != screen.digest(normalized)
                        or smoke.get('protocol_sha256') != screen.digest(screen.protocol(
                            normalized['backend'], normalized['runtime_versions']))
                        or normalized['backend'] == 'cuda_refine' and not normalized['gpus']):
                    raise ValueError('recovery_screen_smoke_not_qualified')
                if candidate['worker'].get('native_interleave_confirmations') is True:
                    raise ValueError('recovery_screen_conflicts_with_interleaving')
                if override.get('native_work_conserving') is True:
                    raise ValueError('recovery_screen_conflicts_with_refill')
                if normalized['backend'] == 'cuda_refine':
                    candidate.setdefault('native_gpu_by_host', {})[host] = normalized['gpus']
            override['native_screen'] = normalized

    bank = receipt.get('reference_bank')
    if bank is not None:
        if (not isinstance(bank, dict)
                or set(bank) - {'path', 'sha256', 'rows', 'annotation_qualification'}):
            raise ValueError('invalid_recovery_reference_bank')
        path, proof = bank.get('path'), bank.get('annotation_qualification', {})
        runtime = candidate.get('hosts', {}).get('slave', {}).get('novelty_runtime', {}).get('identity_sha256')
        if (not isinstance(path, str) or not re.fullmatch(r'/root/[A-Za-z0-9_./-]+[.]json', path)
                or '..' in Path(path).parts or not hash_value(bank.get('sha256'))
                or not positive(bank.get('rows')) or not bound(proof)
                or proof.get('host') != 'slave'
                or proof.get('bank_sha256') != bank['sha256']
                or type(proof.get('annotated_rows')) is not int or proof['annotated_rows'] != bank['rows']
                or not hash_value(runtime) or proof.get('runtime_sha256') != runtime):
            raise ValueError('recovery_reference_bank_not_qualified')
        candidate['hosts']['slave'].update(references=path, reference_bank_sha256=bank['sha256'])

    edits = receipt.get('edit_generation')
    if edits is not None:
        from nanobody_opt.pr100_live_worker import edit_generation_settings
        supplied = edits.get('settings') if isinstance(edits, dict) else None
        if (not isinstance(edits, dict) or set(edits) - {'settings', 'qualification'}
                or not isinstance(supplied, dict) or set(supplied) - {'enabled', 'limit', 'policy'}):
            raise ValueError('invalid_recovery_edit_settings')
        settings = edit_generation_settings({'edit_generation': supplied})
        if settings['enabled']:
            from nanobody_opt import rules
            proof = edits.get('qualification', {})
            bank_hash = candidate['hosts']['slave'].get('reference_bank_sha256')
            if (not bound(proof) or not clean(proof) or not hash_value(bank_hash)
                    or proof.get('bank_sha256') != bank_hash
                    or proof.get('settings_sha256') != digest(settings)
                    or not hash_value(proof.get('archive_sha256'))
                    or proof.get('filter_sha') != rules.FILTER_SHA
                    or not positive(proof.get('tested_candidates'))
                    or not positive(proof.get('novelty_passed_count'))
                    or proof['novelty_passed_count'] > proof['tested_candidates']):
                raise ValueError('recovery_edit_novelty_not_qualified')
        candidate['worker']['edit_generation'] = settings
    return candidate


def proposed_config(current, sha, epoch, mode, qualification=None, *, novelty_runtimes=None,
                    interleave_confirmations=False, native_refill_hosts=(), missing_cell_donors=False,
                    native_screen=None, force=False):
    if current.get('source_sha') not in APPROVED_PREDECESSORS | {sha}:
        raise ValueError('unexpected_current_release')
    mode = selected_mode(current, mode)
    if mode not in ('master', 'both'):
        raise ValueError('invalid_activation_mode')
    if not isinstance(interleave_confirmations, bool):
        raise ValueError('interleave_confirmations_must_be_boolean')
    if native_screen is not None and native_screen not in ('resident_cpu', 'cuda_refine'):
        raise ValueError('invalid_native_screen_backend')
    if native_screen is not None and interleave_confirmations:
        raise ValueError('native_screen_and_interleave_confirmations_are_mutually_exclusive')
    if native_screen is not None and native_refill_hosts:
        raise ValueError('native_screen_and_native_refill_are_mutually_exclusive')
    if interleave_confirmations and current.get('worker', {}).get('native_screen', {}).get('enabled', False):
        raise ValueError('interleaved_confirmations_require_original_cpu_profiling')
    value = copy.deepcopy(current); previous = current['source_sha']
    if interleave_confirmations:
        value['worker']['native_interleave_confirmations'] = True
    if novelty_runtimes is not None:
        # This rollout repairs annotation only. Preserve qualified native
        # runtimes, worker admission, affinity, GPUs and scientific settings.
        runtimes = verified_novelty_runtimes(novelty_runtimes)
        if mode != selected_mode(current, 'preserve'):
            raise ValueError('novelty_rollout_cannot_change_native_topology')
        if mode == 'both' and qualification is None:
            raise ValueError('slave_runtime_not_qualified')
        value.update(source_sha=sha, start_epoch=epoch)
        for host in ('master', 'slave'):
            runtime = value['hosts'][host]
            if not runtime['release'].endswith('/' + previous):
                raise ValueError('unexpected_host_release')
            runtime['release'] = runtime['release'][:-len(previous)] + sha
            runtime['novelty_runtime'] = runtimes[host]
        if native_screen is not None:
            enable_native_screen(value, native_screen)
        return rollout_options(value, native_refill_hosts=native_refill_hosts,
                               missing_cell_donors=missing_cell_donors)
    value.update(source_sha=sha, start_epoch=epoch, native_hosts=['master'] if mode == 'master' else ['master', 'slave'])
    if native_screen is not None:
        enable_native_screen(value, native_screen)
    for host in ('master', 'slave'):
        runtime = value['hosts'][host]
        if not runtime['release'].endswith('/' + previous):
            raise ValueError('unexpected_host_release')
        runtime['release'] = runtime['release'][:-len(previous)] + sha
    if mode == 'both':
        if qualification is None:
            raise ValueError('slave_runtime_not_qualified')
        value['hosts']['slave'].update(qualified_native_paths(qualification))
        value['hosts']['slave']['native_tools'] = qualification['runtime']['native_tools']
    return rollout_options(value, native_refill_hosts=native_refill_hosts,
                           missing_cell_donors=missing_cell_donors)


def command(args, *, cwd=None, timeout=30):
    result = subprocess.run(args, cwd=cwd, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('subprocess_failed')
    return result.stdout


def verify_consumer_reference(config, release, consumer_root):
    """The independently installed submitter must accept the selected native rows.

    Staging a controller release does not update /opt/nova-submit. Checking its
    actual registry and verifier prevents publishing CUDA checkpoints that the
    submitter will reject after the controller has already accepted them.
    """
    identities = set()
    for host in config.get('native_hosts', ['master']):
        settings = config.get('native_worker_overrides', {}).get(host, {}).get(
            'native_reference', config.get('worker', {}).get('native_reference', {}))
        if settings.get('enabled') is True:
            identity = settings.get('registration_sha256')
            if not hash_value(identity):
                raise ValueError('consumer_registration_identity_missing')
            identities.add(identity)
    if not identities:
        return
    release, consumer_root = Path(release), Path(consumer_root)
    required = ('nanobody_opt/pr100_native_reference.py',
                'nanobody_opt/pr100_validator_reference.py',
                'config/pr100_validator_reference.json')
    for name in required:
        source, installed = release / name, consumer_root / name
        if (not source.is_file() or not installed.is_file()
                or hashlib.sha256(source.read_bytes()).digest()
                != hashlib.sha256(installed.read_bytes()).digest()):
            raise ValueError('consumer_reference_source_mismatch')
    source_registry = read(release / 'config/pr100_native_reference.json', {})
    installed_registry = read(consumer_root / 'config/pr100_native_reference.json', {})
    def registered(value):
        return {digest(row) for row in value.get('registrations', [])}
    if not identities <= registered(source_registry) & registered(installed_registry):
        raise ValueError('consumer_native_registration_missing')
    # Execute from the actual consumer installation, not the controller's
    # already-imported modules. registration() also validates CUDA loader schema.
    code = ('import json,sys;sys.path.insert(0,sys.argv[1]);'
            'from nanobody_opt.pr100_native_reference import registration;'
            '[registration(k) for k in json.loads(sys.argv[2])]')
    command([PYTHON, '-B', '-c', code, str(consumer_root), json.dumps(sorted(identities))],
            cwd=consumer_root)


class ClockUnavailable(RuntimeError):
    pass


def current_clock():
    code = ('import bittensor as bt,json;c=bt.Subtensor(network="finney");'
            'b=int(c.get_current_block());n=int(c.tempo(68))+1;'
            'print(json.dumps({"epoch":b//n,"block":b,"epoch_length":n}))')
    try:
        raw = command([PYTHON, '-B', '-c', code], timeout=25)
        value = json.loads(raw.splitlines()[-1])
        return {k: value[k] for k in ('epoch', 'block', 'epoch_length')}
    except Exception as error:
        raise ClockUnavailable('fresh_clock_unavailable') from error


class Service:
    def __init__(self, release, config_path, backup, unit=UNIT):
        self.release, self.config_path, self.backup, self.unit = release, config_path, backup, unit
    def validate(self, candidate):
        command([PYTHON, '-B', '-c', 'import json,sys;from nanobody_opt.pr100_live_controller import validate_config;'
                 'validate_config(json.load(open(sys.argv[1])))', str(candidate)], cwd=self.release)
        verify_consumer_reference(read(candidate), self.release, Path('/opt/nova-submit'))
    def stop(self): command(['sudo', '-n', 'systemctl', 'stop', UNIT_NAME], timeout=120)
    def start(self): command(['sudo', '-n', 'systemctl', 'start', UNIT_NAME])
    def install(self, staged):
        command(['sudo', '-n', 'install', '-o', 'root', '-g', 'root', '-m', '0644', str(staged), str(self.unit)])
        command(['sudo', '-n', 'systemctl', 'daemon-reload'])
    def verify(self):
        raw = command(['systemctl', 'show', UNIT_NAME, '--property=ActiveState,UnitFileState,MainPID'])
        fields = dict(line.split('=', 1) for line in raw.decode().splitlines())
        pid = int(fields['MainPID'])
        if fields['ActiveState'] != 'active' or fields['UnitFileState'] != 'enabled' or pid <= 0:
            raise RuntimeError('service_not_running')
        if Path('/proc', str(pid), 'cwd').resolve() != self.release:
            raise RuntimeError('service_release_mismatch')


def boundary(config, clock, now, *, heartbeat_max_age=30, novelty_migration=False,
             protect_partial_native=False):
    root = Path(config['state_root'])
    state = read(root / str(clock['epoch']) / 'continuation.json', {})
    if protect_partial_native and state.get('pending_confirmation'):
        from nanobody_opt.pr100_live_controller import admissible, RESERVE_BLOCKS
        # A changed adapter identity must not strand confirmations that can
        # still finish this epoch. Expired evidence stays retained on disk.
        if admissible({**clock, 'updated_at_unix': now}, clock['epoch'],
                      reserve=config.get('reserve_blocks', RESERVE_BLOCKS), now=now):
            return False
    recoverable = novelty_migration and clean_pending_novelty_migration(config, clock, state)
    return ready_for_activation(clock, read(root / 'heartbeat.json', {}),
                                state, now, heartbeat_max_age=heartbeat_max_age,
                                recoverable_attempt=recoverable)


def rendered_unit(release, path):
    unit = (release / 'deploy/submit_host' / UNIT_NAME).read_text()
    return unit.replace('<RELEASE_DIR>', str(release)).replace(
        '<CONFIG_PATH>', str(path)).replace('<CONTROLLER_PYTHON>', PYTHON).encode()


def verify_started_service(service, release, path):
    service.verify()
    if service.unit.read_bytes() != rendered_unit(release, path):
        raise RuntimeError('installed_unit_mismatch')


def recovery_required(sha, epoch, error, *, start_attempted):
    return {'status': 'activation_unverified_new_release_preserved', 'source_sha': sha,
            'epoch': epoch, 'error_type': type(error).__name__, 'recovery_required': True,
            'new_service_start_attempted': start_attempted,
            'scientific_ownership_must_be_checked_before_recovery': True}


def registered_native_rollout(config, release, identity):
    """Explicitly replace a CUDA resident implementation with identical science.

    The row and its real qualification report must already ship in this release.
    Only the selected registration changes; allocation and sealed runtime stay.
    """
    if identity is None:
        return config
    if not hash_value(identity):
        raise ValueError('invalid_native_registration_identity')
    release = Path(release)
    rows = read(release / 'config/pr100_native_reference.json', {}).get('registrations', [])
    by_identity = {digest(row): row for row in rows}
    row = by_identity.get(identity, {})
    module = release / 'nanobody_opt/pr100_registered_resident.py'
    proof = release / row.get('qualification_report_file', '')
    if (row.get('qualification_status') != 'qualified'
            or row.get('actual_validator_worker_attested') is not False
            or not module.is_file() or row.get('resident_module_sha256') != hashlib.sha256(module.read_bytes()).hexdigest()
            or not proof.is_file() or not proof.resolve().is_relative_to(release.resolve())
            or row.get('qualification_report_sha256') != hashlib.sha256(proof.read_bytes()).hexdigest()):
        raise ValueError('native_registration_not_qualified_for_release')
    pins = ('runtime_sha256', 'observer_sha256', 'implementation_sha256',
            'environment_manifest_sha256', 'bundle_manifest_sha256', 'weights', 'numpy', 'policy', 'cuda_plugin')
    value = copy.deepcopy(config)
    selected = 0
    for host in value.get('native_hosts', ['master']):
        settings = value.get('native_worker_overrides', {}).get(host, {}).get(
            'native_reference', value.get('worker', {}).get('native_reference', {}))
        old = by_identity.get(settings.get('registration_sha256'), {})
        if (settings.get('enabled') is not True or settings.get('resident') is not True
                or old.get('policy', {}).get('final_refinement', {}).get('platform') != 'CUDA'):
            continue
        if any(key not in old or old[key] != row.get(key) for key in pins):
            raise ValueError('native_registration_changes_scientific_runtime')
        replacement = copy.deepcopy(settings)
        replacement['registration_sha256'] = identity
        value.setdefault('native_worker_overrides', {}).setdefault(host, {})['native_reference'] = replacement
        selected += 1
    if not selected:
        raise ValueError('no_selected_cuda_resident_registration')
    return value


def handoff(base, sha, mode, release, qualification, *, novelty_runtimes=None, interleave_confirmations=False,
            native_refill_hosts=(), missing_cell_donors=False, native_screen=None, recovery_rollout_receipt=None,
            native_registration=None,
            clock_reader=current_clock,
            service_factory=Service, now=time.time, force=False):
    path = base / 'pr100-controller.json'; original = path.read_bytes(); config = json.loads(original)
    clock = clock_reader()
    migration = (novelty_runtimes is not None and any(
        config.get('hosts', {}).get(host, {}).get('novelty_runtime', {}).get('identity_sha256')
        != runtime.get('identity_sha256') for host, runtime in novelty_runtimes.items()))
    protect_partial = config.get('source_sha') != sha
    if not boundary(config, clock, now(), novelty_migration=migration, protect_partial_native=protect_partial):
        return {'status': 'waiting_for_clean_attempt', 'epoch': clock['epoch']}
    candidate = base / 'pr100-staging' / ('throughput-candidate-' + sha + '-' + mode + '.json')
    proposal = proposed_config(config, sha, clock['epoch'], mode, qualification, novelty_runtimes=novelty_runtimes,
                               interleave_confirmations=interleave_confirmations,
                               native_refill_hosts=native_refill_hosts, missing_cell_donors=missing_cell_donors,
                               native_screen=native_screen, force=force)
    proposal = recovery_rollout(proposal, recovery_rollout_receipt, sha)
    proposal = registered_native_rollout(proposal, release, native_registration)
    if ({k: v for k, v in proposal.items() if k != 'start_epoch'} ==
            {k: v for k, v in config.items() if k != 'start_epoch'}
            and config.get('start_epoch', clock['epoch'] + 1) <= clock['epoch']):
        service = service_factory(release, path, None)
        try:
            verify_started_service(service, release, path)
        except Exception as error:
            return recovery_required(sha, clock['epoch'], error, start_attempted=False)
        return {'status': 'already_configured', 'epoch': clock['epoch']}
    backup = base / 'pr100-staging' / ('throughput-backup-' + sha + '-' + mode + '-' + str(time.time_ns()))
    backup.mkdir(parents=True, exist_ok=False)
    service = service_factory(release, path, backup)
    atomic(candidate, proposal); service.validate(candidate)
    atomic(backup / 'config.json', original, raw=True)
    atomic(backup / 'controller.service', service.unit.read_bytes(), raw=True)
    staged_unit = backup / 'new-controller.service'; atomic(staged_unit, rendered_unit(release, path), raw=True)
    changed, start_attempted = False, False
    # Keep the short pause between attempts free of another slow chain RPC.
    # The fresh chain check under the lock below still precedes any install.
    if (path.read_bytes() != original or not boundary(config, clock, now(), novelty_migration=migration,
                                                    protect_partial_native=protect_partial)):
        return {'status': 'waiting_for_clean_attempt', 'epoch': clock['epoch']}
    # The prepared config must use the same current epoch as the final check.
    if proposal['start_epoch'] != clock['epoch']:
        return {'status': 'waiting_for_clean_attempt', 'epoch': clock['epoch']}
    try:
        service.stop()
        # Obtain the same lock as the controller after stop has completed.
        with (Path(config['state_root']) / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fresh = clock_reader()
            raced = (path.read_bytes() != original or fresh['epoch'] != clock['epoch']
                     or not boundary(config, fresh, now(), heartbeat_max_age=90, novelty_migration=migration,
                                     protect_partial_native=protect_partial))
            if not raced:
                # Do not rewrite continuation state or the publication artifact.
                changed = True
                os.replace(candidate, path)
                service.install(staged_unit)
        if raced:
            service.start()
            return {'status': 'boundary_race_old_service_restarted', 'epoch': fresh['epoch']}
        # A started controller may already own native work. Its nonblocking
        # lock cannot safely be held through startup. Do not roll it back here.
        start_attempted = True
        service.start(); verify_started_service(service, release, path)
        return {'status': 'activated', 'epoch': fresh['epoch'], 'native_hosts': proposal.get('native_hosts', ['master']),
                'source_sha': sha, 'state_preserved': True, 'publication_preserved': True}
    except Exception as error:
        if changed and start_attempted:
            # Never rewrite old config beneath a possibly live new process.
            return recovery_required(sha, clock['epoch'], error, start_attempted=True)
        if changed:
            # No new start occurred. Restore while stopped; a failed restore
            # must leave the service stopped for explicit recovery.
            try:
                service.install(backup / 'controller.service')
                atomic(path, original, raw=True)
            except Exception as restore_error:
                return {'status': 'rollback_failed_service_stopped', 'source_sha': sha,
                        'epoch': clock['epoch'], 'error_type': type(restore_error).__name__,
                        'recovery_required': True, 'new_service_start_attempted': False}
        service.start()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sha'); parser.add_argument('mode', choices=('master', 'both', 'preserve'))
    parser.add_argument('--staging-receipt', required=True, type=Path)
    parser.add_argument('--qualification-receipt', type=Path)
    parser.add_argument('--novelty-qualification-receipt', required=True, type=Path)
    parser.add_argument('--interleave-confirmations', action='store_true',
                        help='Opt in to CI-qualified confirmation-first mixed native waves')
    parser.add_argument('--native-refill-host', action='append', choices=('master', 'slave'), default=[],
                        help='Enable bounded slot refill only on this qualified native host')
    parser.add_argument('--native-screen', choices=('resident_cpu', 'cuda_refine'), default=None,
                        help='Enable the non-authoritative native developability pre-filter on the chosen backend')
    parser.add_argument('--missing-cell-donors', action='store_true',
                        help='Enable bounded missing-cell donors with unchanged validation and unseeded controls')
    parser.add_argument('--recovery-rollout-receipt', type=Path,
                        help='Source-bound measured screen, annotated bank and novelty-qualified edit rollout')
    parser.add_argument('--native-registration',
                        help='Explicit qualified CUDA resident replacement with identical sealed science')
    parser.add_argument('--wait-seconds', type=int, default=5400)
    args = parser.parse_args(argv)
    if args.native_screen is not None and args.interleave_confirmations:
        parser.error('--native-screen and --interleave-confirmations are mutually exclusive')
    if args.native_screen is not None and args.native_refill_host:
        parser.error('--native-screen and --native-refill-host are mutually exclusive')
    if not re.fullmatch('[0-9a-f]{40}', args.sha) or not 1 <= args.wait_seconds <= 7200:
        raise ValueError('invalid_arguments')
    release = BASE / 'pr100-releases' / args.sha
    report = BASE / 'pr100-staging' / ('throughput-activation-' + args.sha + '-' + args.mode + '.json')
    end = time.monotonic() + args.wait_seconds
    lock_path = BASE / 'pr100-staging/throughput-activation.lock'; lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while time.monotonic() < end:
                if not args.staging_receipt.is_file():
                    value = {'status': 'waiting_for_verified_staging'}
                else:
                    effective_mode = selected_mode(read(BASE / 'pr100-controller.json', {}), args.mode)
                    q = verify_prerequisites(args.sha, release, args.staging_receipt,
                        mode=effective_mode, qualification_path=args.qualification_receipt,
                        interleave_confirmations=args.interleave_confirmations,
                        native_refill_hosts=args.native_refill_host, missing_cell_donors=args.missing_cell_donors)
                    novelty = verify_novelty_qualification(release, args.novelty_qualification_receipt)
                    try:
                        value = handoff(BASE, args.sha, args.mode, release, q, novelty_runtimes=novelty,
                                        interleave_confirmations=args.interleave_confirmations,
                                        native_refill_hosts=args.native_refill_host,
                                        missing_cell_donors=args.missing_cell_donors,
                                        native_screen=args.native_screen,
                                        native_registration=args.native_registration,
                                        recovery_rollout_receipt=(read(args.recovery_rollout_receipt, {})
                                            if args.recovery_rollout_receipt else None),
                                        force=False)
                    except ClockUnavailable:
                        value = {'status': 'waiting_for_fresh_clock'}
                atomic(report, {**value, 'source_sha': args.sha, 'mode': args.mode, 'updated_at_unix': time.time()})
                if value['status'] in ('activated', 'already_configured'):
                    print(json.dumps(value)); return 0
                if value.get('recovery_required') is True:
                    print(json.dumps(value)); return 1
                time.sleep(5)
            atomic(report, {'status': 'timed_out', 'source_sha': args.sha, 'mode': args.mode})
            print(json.dumps({'status': 'timed_out'})); return 1
        except Exception as error:
            value = {'status': 'failed', 'error_type': type(error).__name__, 'source_sha': args.sha, 'mode': args.mode}
            atomic(report, value); print(json.dumps(value)); return 1


if __name__ == '__main__':
    raise SystemExit(main())
