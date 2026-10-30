"""Stable historical-field ranking for already validated live-trial candidates.

This changes neither eligibility nor the measured scoring protocol. A caller
must explicitly supply a hash-bound public field frozen before the selected
epoch. The live controller freezes a past-only field from the completed
public-ledger boards and falls back to its within-trial ranking whenever no
usable field can be frozen; the field is a ranking prior, never a gate.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
import re


@dataclass(frozen=True)
class FrozenField:
    target: str
    epoch: int
    reference_epochs: tuple[int, ...]
    source_sha256: str
    scorer_sha256: str
    settings_sha256: str
    settings_file_sha256: str
    row_count: int
    columns: tuple[tuple[str, str, tuple[float, ...]], ...]

    def score(self, metrics):
        total = 0
        for key, direction, values in self.columns:
            value = metrics.get(key)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError('reference ranking requires complete finite metrics')
            better = (len(values) - bisect_right(values, value) if direction == 'max'
                      else bisect_left(values, value))
            total += 1 + better
        return float(total)

    def selection(self, metrics):
        score = self.score(metrics)
        return {'score': score, 'score_p90': score,
                'method': 'one_pass_frozen_public_field_dense_rank_sum',
                'reference_scope': 'historical_public_validator_field',
                'reference_count': self.row_count,
                'reference_epochs': list(self.reference_epochs),
                'reference_sha256': self.source_sha256,
                'reference_target': self.target, 'reference_for_epoch': self.epoch,
                'reference_latest_epoch_age': self.epoch - max(self.reference_epochs),
                'reference_source': 'public_validator_observations',
                'multipass_measured': False}


def prepare(raw, *, expected_sha256, target, epoch, metrics, scorer_sha256,
            settings_sha256, settings_file_sha256, maximum_latest_epoch_age=2):
    """Compile exactly the supplied immutable field; no network or DB access."""
    if (not isinstance(raw, bytes) or len(raw) > 8 * 1024 * 1024
            or not isinstance(expected_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None
            or hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError('reference field bytes changed')
    if type(epoch) is not int or epoch < 0 or not isinstance(target, str) or not target:
        raise ValueError('invalid reference challenge')
    if any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None
           for value in (scorer_sha256, settings_sha256, settings_file_sha256)):
        raise ValueError('reference ranking scorer identity is missing')
    if type(maximum_latest_epoch_age) is not int or not 1 <= maximum_latest_epoch_age <= 10:
        raise ValueError('invalid reference freshness bound')
    if (not isinstance(metrics, dict) or not metrics
            or any(not isinstance(k, str) or v not in ('min', 'max') for k, v in metrics.items())):
        raise ValueError('invalid reference metric directions')
    field = json.loads(raw)
    if not isinstance(field, dict) or set(field) != {'target', 'epochs', 'rows'} or field['target'] != target:
        raise ValueError('reference field target or schema mismatch')
    epochs = field['epochs']
    if (not isinstance(epochs, list) or not 1 <= len(epochs) <= 10
            or any(type(e) is not int or not 0 <= e < epoch for e in epochs)
            or len(set(epochs)) != len(epochs)):
        raise ValueError('reference field must contain distinct past epochs')
    if epoch - max(epochs) > maximum_latest_epoch_age:
        raise ValueError('reference field is stale')
    rows = field['rows']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError('reference field has no bounded observations')
    for row in rows:
        if (not isinstance(row, dict) or set(row) != set(metrics)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in row.values())):
            raise ValueError('reference field has incomplete or nonfinite metrics')
    columns = tuple((key, direction, tuple(sorted({row[key] for row in rows})))
                    for key, direction in metrics.items())
    return FrozenField(target, epoch, tuple(epochs), expected_sha256, scorer_sha256,
                       settings_sha256, settings_file_sha256, len(rows), columns)


def serialize(field):
    """Canonical sort-keyed bytes for one collected ``{target, epochs, rows}`` dict."""
    return json.dumps(field, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def freeze(field, *, target, epoch, metrics, scorer_sha256, settings_sha256,
           settings_file_sha256, maximum_latest_epoch_age=2):
    """Hash-bound :class:`FrozenField` from an already-collected field dict.

    The caller collects only target-matched, strictly-past observations.
    ``prepare`` re-validates the hash, challenge, freshness and metric
    completeness, so an unusable field raises instead of silently changing a
    ranking. This is the single freeze entry point used by the live controller.
    """
    raw = serialize(field)
    return prepare(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
                   target=target, epoch=epoch, metrics=metrics,
                   scorer_sha256=scorer_sha256, settings_sha256=settings_sha256,
                   settings_file_sha256=settings_file_sha256,
                   maximum_latest_epoch_age=maximum_latest_epoch_age)
