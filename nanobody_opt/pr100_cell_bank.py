"""Challenge-indexed candidate bank for the PR100 nanobody epoch.

Only two scalars are revealed at epoch start (``allowed_v`` and
``allowed_cdr3_length``), so every expensive stage -- annotation, native
developability (~0.86 core-hours per accepted candidate), archive novelty and
target scoring -- can be paid for in advance and indexed by the 171 buckets of
``rules.challenge`` (9 germlines x 19 CDR3 lengths). This module owns the
storage and the use-time re-validation; it never measures anything itself and it
never re-implements a gate: every verdict comes from ``nanobody_opt.rules`` and
``nanobody_opt.pr100_live_selection``.

Three rules the audit makes non-negotiable
(``nova-reports/pr100-audit-20261026/7_developability_algorithm.md`` 2.1, 8 3.1):

* A stored developability measurement is a **compute saving, never a pass
  certificate**. A record is used as tier A only when it carries at least
  ``min_repeats`` measurements from the *pinned* runtime identity that are not
  older than the configured freshness window; otherwise it is downgraded to
  tier B and published as such.
* Scores are target-scoped. Annotations and native profiles survive a target
  rotation; score observations do not, so they live in their own list with the
  scorer identity beside them.
* Novelty is a moving target. Each record carries an archive watermark (sha256,
  row count and a prefix digest), and at use time only the rows appended since
  that watermark are searched. Anything that cannot be checked incrementally is
  reported as needing a full pass rather than silently trusted.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import tempfile
import time
from pathlib import Path

from nanobody_opt import rules, pr100_live_selection as selection

SCHEMA_VERSION = 1
MANIFEST_NAME = 'bank.json'
RECORD_KIND = 'pr100-cell-bank-candidate'
# Measurements older than this are no longer accepted as a tier A certificate.
DEFAULT_MAX_MEASUREMENT_AGE_SECONDS = 14 * 24 * 3600.0
DEFAULT_LIMIT = 13
# Per-bucket inventory target: 13 are consumed per draw, the margin covers
# novelty churn and the measured native gate flip rate.
DEFAULT_BUCKET_TARGET = 20
_HEX64 = re.compile(r'[0-9a-f]{64}')
_GENE = re.compile(r'[A-Za-z0-9*-]{3,32}')


# --- bucket algebra ---------------------------------------------------------------

def bucket_id(v_gene, cdr3_length):
    """Canonical bucket key ``"IGHV3-23:18"``."""
    gene = str(v_gene)
    if not _GENE.fullmatch(gene):
        raise ValueError('invalid v gene')
    length = cdr3_length
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError('cdr3_length must be a positive integer')
    return f'{gene}:{length}'


def challenge_bucket(challenge):
    return bucket_id(challenge['allowed_v'], challenge['allowed_cdr3_length'])


def bucket_probabilities(config):
    """P(bucket) for all 171 buckets, straight from the pinned germline config.

    ``rules.challenge`` picks a category uniformly, then a gene uniformly inside
    it, then a CDR3 length uniformly: the three singleton categories therefore
    carry 75% of the probability mass over 57 of the 171 buckets.
    """
    settings = config['germline_filtering']
    categories = settings['v_neighborhood_categories']
    lengths = settings['cdr3_length_range']
    span = list(range(int(lengths['min']), int(lengths['max']) + 1))
    if not categories or not span:
        raise ValueError('empty challenge space')
    out = {}
    for genes in categories.values():
        if not genes:
            raise ValueError('empty V category')
        weight = 1.0 / len(categories) / len(genes) / len(span)
        for gene in genes:
            for length in span:
                out[bucket_id(gene, length)] = weight
    return out


def synthetic_challenge(config, epoch, v_gene, cdr3_length, *, epoch_length=361,
                        attempts=200000, seed=None):
    """A real ``rules.challenge`` object for a chosen bucket, from a searched hash.

    The builder must not wait for an epoch-start hash to prepare a bucket, so we
    search block hashes until ``rules.challenge`` draws the requested bucket
    (expected 76 tries for a V1-V3 bucket, 456 for a V4 one). The result passes
    ``rules.challenge``/``rules.context`` byte-for-byte, which is what the worker
    phases verify on entry.

    The searched hash is **not** a chain hash. Nothing built this way may be
    published: ``pr100_evidence`` is rebuilt against the live challenge in
    :meth:`CellBank.select_for_epoch`, and the live epoch-start hash is still
    required for ``rules.verified_fleet``, the trial export and the Pi commit.
    """
    wanted = bucket_id(v_gene, cdr3_length)
    rng = random.Random(f'{wanted}|{epoch}' if seed is None else seed)
    for _ in range(int(attempts)):
        block_hash = '0x' + hashlib.sha256(rng.randbytes(32)).hexdigest()
        candidate = rules.challenge(config, epoch, block_hash, epoch_length)
        if challenge_bucket(candidate) == wanted:
            return candidate
    raise ValueError('no block hash produced the requested bucket')


# --- archive views ----------------------------------------------------------------

def archive_view(archive):
    """``{'sha256', 'rows', 'sequences'}`` from csv bytes, a path or a built view."""
    if isinstance(archive, dict) and 'sequences' in archive and 'sha256' in archive:
        return archive
    if isinstance(archive, (str, Path)):
        data = Path(archive).read_bytes()
    elif isinstance(archive, bytes):
        data = archive
    else:
        raise ValueError('archive must be csv bytes, a path or a built view')
    digest = hashlib.sha256(data).hexdigest()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8', 'replace')))
    if not reader.fieldnames or 'sequence' not in reader.fieldnames:
        raise ValueError('archive csv has no sequence column')
    sequences = [str(row.get('sequence') or '').strip().upper() for row in reader]
    return {'sha256': digest, 'rows': len(sequences), 'sequences': sequences}


def prefix_digest(sequences, rows):
    """Digest of the first ``rows`` archive sequences: makes append-only checkable."""
    rows = max(0, int(rows))
    body = '\n'.join(sequences[:rows])
    return hashlib.sha256(body.encode()).hexdigest()


def archive_delta(watermark, view):
    """Rows appended since ``watermark``, or a reason a full pass is required."""
    if not isinstance(watermark, dict):
        return None, 'novelty_missing'
    stored_rows, stored_sha = watermark.get('archive_rows'), watermark.get('archive_sha256')
    if (isinstance(stored_rows, bool) or not isinstance(stored_rows, int) or stored_rows < 0
            or not isinstance(stored_sha, str) or not _HEX64.fullmatch(stored_sha)):
        return None, 'novelty_archive_invalid'
    if stored_sha == view['sha256']:
        return [], ''
    if view['rows'] < stored_rows:
        # The archive shrank or was rewritten: the watermark says nothing.
        return None, 'archive_not_append_only'
    stored_prefix = watermark.get('archive_prefix_sha256')
    if isinstance(stored_prefix, str) and _HEX64.fullmatch(stored_prefix):
        if prefix_digest(view['sequences'], stored_rows) != stored_prefix:
            return None, 'archive_not_append_only'
    else:
        return None, 'novelty_archive_invalid'
    return view['sequences'][stored_rows:], ''


# --- records ----------------------------------------------------------------------

def measurement_runtime(*, filter_sha=None, annotation_identity=None, native_runtime=None,
                        host=None, measured_at_unix=None):
    """Identity of the runtime that produced a measurement set."""
    return {'filter_sha': filter_sha or rules.FILTER_SHA,
            'annotation_identity': annotation_identity,
            'native_runtime': native_runtime, 'host': host,
            'measured_at_unix': time.time() if measured_at_unix is None else float(measured_at_unix)}


def runtime_matches(stored, pinned):
    """True when every pinned runtime key is present and equal in ``stored``."""
    if not pinned:
        return True
    if not isinstance(stored, dict):
        return False
    return all(stored.get(key) == value for key, value in pinned.items())


def build_record(*, target, v_gene, cdr3_length, sequence, annotation, measurements,
                 runtime, novelty, provenance, scores=None, tier=None, repeats=None,
                 margin=None, now=None):
    """A schema-checked bank record. Raises rather than storing something unusable."""
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError('invalid sequence')
    sequence = sequence.strip().upper()
    measurements = list(measurements or [])
    if not measurements:
        raise ValueError('a bank record needs at least one measurement')
    if not isinstance(annotation, dict) or not isinstance(novelty, dict):
        raise ValueError('annotation and novelty evidence are required')
    if not isinstance(provenance, dict) or not provenance.get('source_sha'):
        raise ValueError('provenance must carry the source sha')
    settings = selection.margin_settings(margin)
    values = selection.psh_values(measurements)
    mean, sd = selection.mean_and_sd(values)
    stored_tier = tier
    if stored_tier is None:
        stored_tier = ('A' if mean is not None and len(measurements) >= settings['min_repeats']
                       and mean <= selection.margin_threshold(len(measurements), settings) else 'B')
    if stored_tier not in ('A', 'B'):
        raise ValueError('tier must be A or B')
    stamp = time.time() if now is None else float(now)
    return {'schema_version': SCHEMA_VERSION, 'record_kind': RECORD_KIND,
            'target': str(target), 'bucket': {'v_gene': str(v_gene),
                                              'cdr3_length': int(cdr3_length)},
            'bucket_id': bucket_id(v_gene, cdr3_length),
            'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
            'annotation': annotation, 'measurements': measurements,
            'measurement_runtime': dict(runtime), 'tier': stored_tier,
            'psh_values': values, 'psh_mean': mean, 'psh_sd': sd,
            'repeats': int(repeats if repeats is not None else len(measurements)),
            'margin': settings,
            'margin_threshold': selection.margin_threshold(len(measurements), settings),
            'novelty': novelty, 'scores': list(scores or []),
            'provenance': dict(provenance), 'updated_at_unix': stamp}


def score_observation(*, target, metrics, scorer_sha256, settings, settings_file_sha256,
                      observed_at_unix=None):
    for value in (scorer_sha256, settings_file_sha256):
        if not isinstance(value, str) or not _HEX64.fullmatch(value):
            raise ValueError('scorer and settings identities must be sha256 digests')
    if not isinstance(metrics, dict) or not metrics or not isinstance(settings, dict) or not settings:
        raise ValueError('a score observation needs metrics and scorer settings')
    return {'target': str(target), 'metrics': dict(metrics), 'scorer_sha256': scorer_sha256,
            'settings': dict(settings), 'settings_file_sha256': settings_file_sha256,
            'settings_sha256': rules.digest(settings),
            'observed_at_unix': time.time() if observed_at_unix is None else float(observed_at_unix)}


# --- storage ----------------------------------------------------------------------

class CellBank:
    """Append-only JSON-lines bank, one file per (target, bucket).

    Writes are a single atomic ``O_APPEND`` write of one line, so a crash can
    lose the tail of a line but never corrupt an earlier record; a truncated or
    unparsable last line is skipped on read. A later record for the same
    sequence hash supersedes the earlier one (the file stays the audit trail).
    """

    def __init__(self, root):
        self.root = Path(root)

    # -- layout
    def manifest_path(self):
        return self.root / MANIFEST_NAME

    def manifest(self):
        try:
            value = json.loads(self.manifest_path().read_text())
        except (OSError, ValueError):
            return None
        return value

    def ensure(self):
        """Create (or validate) the manifest; refuses a future schema version."""
        existing = self.manifest()
        if existing is not None:
            stored = existing.get('schema_version')
            if stored != SCHEMA_VERSION:
                raise ValueError(f'bank schema version {stored} is not {SCHEMA_VERSION}')
            return existing
        value = {'schema_version': SCHEMA_VERSION, 'record_kind': RECORD_KIND,
                 'created_at_unix': time.time()}
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.manifest_path(), value)
        return value

    def bucket_file(self, target, bucket):
        safe = str(bucket).replace(':', '__cdr3')
        if not re.fullmatch(r'[A-Za-z0-9*_-]{3,64}', safe) or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', str(target)):
            raise ValueError('unsafe bucket or target name')
        return self.root / 'records' / str(target) / (safe + '.jsonl')

    # -- writes
    def add(self, record):
        """Append one record; returns the stored line's path."""
        if record.get('record_kind') != RECORD_KIND or record.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('not a current bank record')
        self.ensure()
        path = self.bucket_file(record['target'], record['bucket_id'])
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, allow_nan=False) + '\n'
        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(handle, line.encode())
            os.fsync(handle)
        finally:
            os.close(handle)
        return path

    # -- reads
    def records(self, target, bucket=None):
        """Current records (last write per sequence hash) for one or all buckets."""
        root = self.root / 'records' / str(target)
        if bucket is not None:
            files = [self.bucket_file(target, bucket)]
        else:
            files = sorted(root.glob('*.jsonl')) if root.is_dir() else []
        current = {}
        for path in files:
            try:
                raw = path.read_text()
            except OSError:
                continue
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    continue  # torn tail line from an interrupted append
                if (value.get('record_kind') != RECORD_KIND
                        or value.get('schema_version') != SCHEMA_VERSION):
                    continue
                current[(value.get('bucket_id'), value.get('sequence_hash'))] = value
        return list(current.values())

    def bucket_status(self, target, *, config=None, per_bucket_target=DEFAULT_BUCKET_TARGET,
                      runtime_identity=None, max_measurement_age_seconds=None, now=None):
        """Inventory per bucket plus the probability-weighted coverage of the bank."""
        now = time.time() if now is None else float(now)
        counts = {}
        for record in self.records(target):
            key = record.get('bucket_id')
            entry = counts.setdefault(key, {'bucket_id': key, 'total': 0, 'tier_a': 0,
                                            'tier_b': 0, 'scored': 0})
            entry['total'] += 1
            tier = usable_tier(record, runtime_identity=runtime_identity, now=now,
                               max_measurement_age_seconds=max_measurement_age_seconds)['tier']
            entry['tier_a' if tier == 'A' else 'tier_b'] += 1
            if any(s.get('target') == target for s in record.get('scores') or []):
                entry['scored'] += 1
        status = {'target': target, 'schema_version': SCHEMA_VERSION, 'buckets': counts,
                  'per_bucket_target': int(per_bucket_target),
                  'non_empty_buckets': len(counts)}
        if config is not None:
            weights = bucket_probabilities(config)
            status['possible_buckets'] = len(weights)
            status['probability_any'] = sum(
                w for key, w in weights.items() if counts.get(key, {}).get('total', 0) > 0)
            status['probability_full'] = sum(
                w for key, w in weights.items()
                if counts.get(key, {}).get('scored', 0) >= DEFAULT_LIMIT)
            status['missing'] = build_priority(config, status,
                                               per_bucket_target=per_bucket_target)
        return status

    # -- use time
    def select_for_epoch(self, challenge, archive, *, config=None, limit=DEFAULT_LIMIT,
                         runtime_identity=None, max_measurement_age_seconds=None,
                         allow_marginal=True, duplicate_reason=None, now=None,
                         require_score=True):
        """A ready-to-publish tiered candidate list for the live challenge.

        Everything cheap is re-checked here and nothing cached is trusted:

        1. ``rules.annotation_reason`` is re-run against the **live** challenge,
           so a stored record whose germline or CDR3 length does not match this
           epoch is dropped rather than published.
        2. ``rules.developability_reason`` is re-run on every stored repeat, and
           the tier is recomputed: tier A survives only with enough fresh
           repeats from the pinned runtime identity (see :func:`usable_tier`).
        3. Novelty is re-checked incrementally against the rows appended since
           the record's watermark; a record whose watermark cannot be used is
           dropped with ``novelty_recheck_required``.
        4. ``pr100_evidence`` is rebuilt around the live challenge and passed
           through ``rules.validate_evidence``; a record that fails is dropped.

        The returned ``validity`` dict has the same shape the controller already
        consumes from the native phase, so the publication path (and therefore
        ``rules.verified_fleet`` and the monotone-quality rule) is unchanged.
        """
        now = time.time() if now is None else float(now)
        view = archive_view(archive)
        epoch = int(challenge['epoch'])
        config = rules.pinned_config(epoch) if config is None else config
        expected = rules.challenge(config, epoch, challenge['epoch_start_block_hash'],
                                   challenge['epoch_length'])
        if expected != challenge:
            raise ValueError('challenge does not match rules')
        requirements = config['nanobody_requirements']
        target = expected['target']
        bucket = challenge_bucket(expected)
        selected, rejected = [], []

        def reject(record, reason):
            rejected.append({'sequence_hash': record.get('sequence_hash'), 'reason': reason})

        for record in self.records(target, bucket):
            if record.get('target') != target:
                reject(record, 'target_mismatch')
                continue
            sequence = record['sequence']
            reason = rules.annotation_reason(record.get('annotation'), requirements,
                                             sequence=sequence, epoch_challenge=expected)
            if reason:
                reject(record, reason)
                continue
            verdict = usable_tier(record, runtime_identity=runtime_identity, now=now,
                                  max_measurement_age_seconds=max_measurement_age_seconds,
                                  requirements=requirements)
            if verdict['tier'] is None:
                reject(record, verdict['reason'] or 'developability_failed')
                continue
            if verdict['tier'] == 'B' and not allow_marginal:
                reject(record, 'margin_missed')
                continue
            new_rows, novelty_reason = archive_delta(record.get('novelty'), view)
            if novelty_reason:
                reject(record, 'novelty_recheck_required')
                continue
            if sequence in new_rows:
                reject(record, 'novelty_not_passed')
                continue
            if duplicate_reason is not None:
                # The two-tier top-50 gate moves with the archive and cannot be
                # cached; the caller supplies the live check.
                reason = duplicate_reason(sequence, view)
                if reason:
                    reject(record, reason or 'layer_1_duplicate_gate_failed')
                    continue
            score = _latest_score(record, target)
            if require_score and score is None:
                reject(record, 'score_missing')
                continue
            novelty = {**{k: v for k, v in (record.get('novelty') or {}).items()
                          if k in ('validator_novel', 'exact_historical', 'coarse_candidates',
                                   'maximum_identity')},
                       'rechecked_archive_sha256': view['sha256'],
                       'rechecked_rows': view['rows'],
                       'incremental_rows_checked': len(new_rows)}
            if novelty.get('validator_novel') is not True or novelty.get('exact_historical') is True:
                reject(record, 'novelty_not_passed')
                continue
            evidence = {'challenge': expected, 'filter_sha': rules.FILTER_SHA,
                        'annotation': record['annotation'],
                        'developability': verdict['measurements'][0]}
            try:
                rules.validate_evidence(evidence, sequence, expected, config)
            except (ValueError, TypeError, KeyError) as error:
                reject(record, str(error) if str(error) in rules.HANDOFF_REJECTION_CODES
                       else 'pr100_evidence_invalid')
                continue
            selected.append({'sequence': sequence, 'sequence_hash': record['sequence_hash'],
                             'source': 'cell_bank', 'pr100_annotation': record['annotation'],
                             'pr100_evidence': evidence, 'novelty': novelty,
                             'layer2_passed': True, 'tier': verdict['tier'],
                             'margin_safe': verdict['tier'] == 'A',
                             'stored_tier': record.get('tier'),
                             'downgraded': verdict['tier'] != record.get('tier'),
                             'measurements': verdict['measurements'],
                             'psh_values': verdict['psh_values'], 'psh_mean': verdict['psh_mean'],
                             'psh_sd': verdict['psh_sd'], 'repeats': verdict['repeats'],
                             'margin_threshold': verdict['margin_threshold'],
                             'bank_score': score})
        ordered = selection.order_candidates(selected, allow_marginal=allow_marginal)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('limit must be a positive integer')
        ordered = ordered[:limit]
        validity = {'challenge': expected, 'target': target, 'epoch': epoch,
                    'bucket_id': bucket, 'source': 'cell_bank',
                    'archive_sha256': view['sha256'], 'archive_checked_at_unix': now,
                    'archive_rows': view['rows'], 'novelty_checked': True,
                    'novelty_mode': 'incremental_watermark', 'status': 'complete',
                    'input_count': len(selected) + len(rejected), 'completed': len(selected),
                    'candidates': ordered, 'rejected': rejected,
                    'tier_counts': {tier: sum(1 for r in ordered if r['tier'] == tier)
                                    for tier in ('A', 'B')},
                    'downgraded_count': sum(1 for r in ordered if r['downgraded'])}
        return validity


def _latest_score(record, target):
    scored = [s for s in record.get('scores') or [] if s.get('target') == target]
    return max(scored, key=lambda s: s.get('observed_at_unix') or 0.0) if scored else None


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def usable_tier(record, *, runtime_identity=None, now=None, max_measurement_age_seconds=None,
                requirements=None):
    """Re-derive the tier from the stored repeats. A cache is never a certificate.

    Measurements that do not come from the pinned runtime identity, or that are
    older than the freshness window, cannot support tier A: the candidate is
    downgraded to tier B (still publishable -- an idle slot forfeits a ticket for
    nothing, see docs/PR100_LIVE_CONTRACT.md 4) rather than promoted on a cached
    lucky draw.
    """
    now = time.time() if now is None else float(now)
    age_limit = (DEFAULT_MAX_MEASUREMENT_AGE_SECONDS if max_measurement_age_seconds is None
                 else float(max_measurement_age_seconds))
    measurements = list(record.get('measurements') or [])
    margin = selection.margin_settings(record.get('margin'))
    if requirements is None:
        requirements = rules.pinned_config(rules.ACTIVATION_EPOCH)['nanobody_requirements']
    verdict = selection.decide(measurements, requirements, sequence=record.get('sequence'),
                               repeats=len(measurements) or 1, margin=margin)
    if verdict['tier'] is None:
        return verdict
    runtime = record.get('measurement_runtime') or {}
    measured_at = runtime.get('measured_at_unix')
    fresh = (isinstance(measured_at, (int, float)) and not isinstance(measured_at, bool)
             and now - float(measured_at) <= age_limit)
    pinned = runtime_matches(runtime, runtime_identity)
    if verdict['tier'] == 'A' and not (fresh and pinned):
        verdict = {**verdict, 'tier': 'B', 'margin_safe': False,
                   'reason': 'stale_measurement' if not fresh else 'runtime_identity_mismatch'}
    return verdict


# --- build planning ---------------------------------------------------------------

def build_priority(config, status, *, per_bucket_target=DEFAULT_BUCKET_TARGET):
    """Buckets still short of ``per_bucket_target``, ordered by probability mass.

    The 57 V1-V3 buckets carry 75% of epochs, so they come first; ties break on
    the shortfall and then on the bucket id for a stable, resumable order.
    """
    weights = bucket_probabilities(config)
    counts = status.get('buckets', {}) if isinstance(status, dict) else {}
    rows = []
    for key, weight in weights.items():
        have = int((counts.get(key) or {}).get('total', 0))
        shortfall = int(per_bucket_target) - have
        if shortfall <= 0:
            continue
        gene, length = key.split(':')
        rows.append({'bucket_id': key, 'v_gene': gene, 'cdr3_length': int(length),
                     'probability': weight, 'have': have, 'shortfall': shortfall})
    rows.sort(key=lambda r: (-r['probability'], -r['shortfall'], r['bucket_id']))
    return rows


def budget_estimate(rows, *, native_cpu_seconds=55.2, native_pass_rate=0.0179,
                    gpu_seconds=16.9, repeats=3):
    """Audit cost model for a build plan (8_epoch_algorithm.md 1.3 / 3.1).

    ``native_cpu_seconds`` and ``native_pass_rate`` are the measured census
    numbers; the repeat-and-margin rule adds ``repeats - 1`` further profiles for
    the ~2% that survive the first pass.
    """
    wanted = sum(int(r['shortfall']) for r in rows)
    if native_pass_rate <= 0:
        raise ValueError('native_pass_rate must be positive')
    first_pass = wanted / native_pass_rate
    profiles = first_pass + wanted * max(0, int(repeats) - 1)
    return {'candidates': wanted, 'native_profiles': profiles,
            'native_core_hours': profiles * float(native_cpu_seconds) / 3600.0,
            'gpu_hours': wanted * float(gpu_seconds) / 3600.0,
            'buckets': len(rows)}
