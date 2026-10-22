"""Bugs 7/8: predicate parity only; no search engine, HF, IgBLAST or AbNumber."""
from itertools import product
from types import SimpleNamespace

from nanobody_opt.check_validator_novelty import duplicate_reason
from sn68_source import pinned_validator_predicate


def test_duplicate_reason_matches_validator_is_duplicate_on_boundary_corpus():
    """Bugs 7/8. Shipping legacy novelty helper vs immutable validator ref."""
    oracle = pinned_validator_predicate()
    # Cartesian product exercises every boundary with all CDR configurations,
    # rather than accidentally hiding a CDR threshold behind Tier 1 identity.
    cdrs = [None, {}, *({'CDR1': a, 'CDR2': b, 'CDR3': c}
            for a, b, c in product((.89, .90, .91), (.89, .90, .91),
                                   (.79, .80, .81, .84, .85, .86)))]
    identities = (.89, .90, .91, .94, .95, .96)
    for identity, cdr in product(identities, cdrs):
        expected, reason = oracle(SimpleNamespace(identity=identity, cdr_similarity=cdr))
        actual = duplicate_reason(identity, cdr)
        assert (actual is not None) == expected, (identity, cdr, actual, reason)
        assert actual == (reason if expected else None), (identity, cdr)
    # Incident denominator fixtures explicitly isolate both identity and CDR3.
    for identity, cdr, expected in (
        (14 / 15, None, True), (60 / 120, None, False),
        (.5, {'CDR3': 14 / 15}, True), (.5, {'CDR3': 60 / 120}, False),
    ):
        duplicate, reason = oracle(SimpleNamespace(identity=identity, cdr_similarity=cdr))
        assert duplicate is expected
        assert duplicate_reason(identity, cdr) == (reason if duplicate else None)
