"""Process-local memoization of already authenticated, unchanged native receipts.

The marker is a Python type, never a JSON field. Every reuse hashes the complete
measurement and the current approval documents. Mutation or registry/contract
changes fall back to the original full validator. Serialized evidence always
loses the marker and is independently validated by its next consumer.
"""
from __future__ import annotations

import hashlib
import json


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    # JSON would encode a tuple like a list and integer keys like strings.
    # Those changes can affect the full validator, so never memoize them.
    pending, seen = [value], set()
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind is dict or (item is value and kind is _VerifiedMeasurement):
            if any(type(key) is not str for key in item):
                raise TypeError('non_json_evidence_key')
            if id(item) not in seen:
                seen.add(id(item)); pending.extend(item.values())
        elif kind is list:
            if id(item) not in seen:
                seen.add(id(item)); pending.extend(item)
        elif item is not None and kind not in (str, int, float, bool):
            raise TypeError('non_json_evidence_value')
    return hashlib.sha256(encoded).digest()


def _approval():
    from nanobody_opt import pr100_native_reference as native
    return (native.authoritative_measurement,
            hashlib.sha256(native.REGISTRY.read_bytes()).digest(),
            _digest(native.reference.contract()))


class _VerifiedMeasurement(dict):
    def __init__(self, value):
        from nanobody_opt import pr100_native_reference as native
        before = _approval()
        content = _digest(value)
        identity = native.measurement_identity(value)
        if identity is None or before != _approval() or content != _digest(value):
            raise ValueError('non_authoritative_measurement')
        super().__init__(value)
        self._content = content
        self._approval = before
        self._identity = dict(identity)

    def unchanged(self):
        try:
            return self._approval == _approval() and self._content == _digest(self)
        except (OSError, ValueError, TypeError, AttributeError):
            return False


def verified(value):
    """Authenticate once before retaining a draw in this worker's history."""
    if type(value) is _VerifiedMeasurement and value.unchanged():
        return value
    try:
        _approval()
    except (AttributeError, OSError):
        # An alternate authentication implementation without readable approval
        # documents stays on its full validation path and receives no marker.
        from nanobody_opt import pr100_native_reference as native
        if not native.authoritative_measurement(value):
            raise ValueError('non_authoritative_measurement')
        return value
    return _VerifiedMeasurement(value)


def authoritative_measurement(value):
    from nanobody_opt import pr100_native_reference as native
    if type(value) is _VerifiedMeasurement and value.unchanged():
        return True
    return native.authoritative_measurement(value)


def measurement_identity(value):
    from nanobody_opt import pr100_native_reference as native
    if type(value) is _VerifiedMeasurement and value.unchanged():
        return dict(value._identity)
    return native.measurement_identity(value)
