"""Pure, dependency-free parent/donor gates ported from ``nanobody_opt_v2``.

The PR100 controller generates de-novo scaffolds and chooses reference donors
with ``pr100_reference_donors.select`` (score-free, cell-ordered). The v2
producer additionally gated its *parents* before seeding edit lanes with
checks that PR100 has no equivalent of:

* ``cdr_developability_in_range`` -- the metanano CDR-only "Red Region" bounds
  (total CDR length / CDR3 length / CDR3 compactness). A framework edit
  preserves the CDRs, so a parent outside these bounds can only seed doomed
  single/framework-edit children; it must be repaired via CDR3 edits first.
  Ported from ``nanobody_opt_v2/select_elite_parents.py``
  (``DEVELOPABILITY_CDR_RED_REGION``, ``_cdr_developability_in_range``).

* ``same_cdr3_family`` -- CDR3 clonotype grouping within a length-normalised
  Levenshtein distance. v2 uses this to cycle distinct CDR3 families when
  backfilling, so one leading clonotype cannot monopolise the seed pool
  (``nanobody_opt_v2/select_elite_parents.py::_same_cdr3_family`` and
  ``nanobody_opt_v2/run_live_producer.py::_historical_child_fallback``).

Both are pure functions of strings/dicts (no abnumber, no network, no DB), so
they are safe to import from the sequence/annotation environment that the PR100
worker's ``sequence_phase`` already runs in.

None of these functions are wired into the live PR100 path yet: they are the
mechanical core of the "parent selection / competitor lane" port. Wiring them
into ``pr100_reference_donors.select`` (family diversity) or into donor
evidence (red-region flagging) is a separate, design-reviewed change -- see the
port plan.
"""
from __future__ import annotations

from typing import Any, Callable

#: CDR-only developability "Red Region" bounds (metanano/filters/developability.py,
#: July 2025). Framework edits preserve the CDRs, so a parent whose L/L3/C is
#: out of range can only produce doomed framework-edit children; it must be
#: repaired via CDR3 edits instead of taking normal single-edit seed seats.
#: Ported verbatim from nanobody_opt_v2/select_elite_parents.py.
CDR_DEVELOPABILITY_RED_REGION = {
    "total_cdr_length": (20.0, 39.0),
    "cdr3_length": (5.0, 23.0),
    "cdr3_compactness": (0.56, 1.61),
}


def cdr_developability_in_range(profile: dict[str, Any] | None) -> bool:
    """True when the CDR-only red flags (L/L3/C) are all inside their ranges.

    These three are inherited unchanged by framework edits, so they are the
    only ones a parent must satisfy to be a safe single-edit/multi-edit
    framework anchor. PSH/PPC/PNC are full-sequence surface metrics and are
    deliberately NOT enforced here -- they are re-checked on the child at the
    native developability gate.

    ``None`` or a malformed/partial profile fails closed (out-of-range), matching
    the v2 selector's behaviour: an unmeasured parent must not silently enter a
    CDR3-preserving lane.
    """
    if not isinstance(profile, dict):
        return False
    try:
        total_cdr_length = float(profile["total_cdr_length"])
        cdr3_length = float(profile["cdr3_length"])
        cdr3_compactness = float(profile["cdr3_compactness"])
    except (KeyError, TypeError, ValueError):
        return False
    lo, hi = CDR_DEVELOPABILITY_RED_REGION["total_cdr_length"]
    if not lo <= total_cdr_length <= hi:
        return False
    lo, hi = CDR_DEVELOPABILITY_RED_REGION["cdr3_length"]
    if not lo <= cdr3_length <= hi:
        return False
    lo, hi = CDR_DEVELOPABILITY_RED_REGION["cdr3_compactness"]
    return lo <= cdr3_compactness <= hi


def same_cdr3_family(left: str, right: str, *, fraction: float = 0.15) -> bool:
    """Group CDR3s within ``fraction`` length-normalised edit distance.

    Never use whole-VHH identity: length-normalised Levenshtein distance
    handles insertions/deletions as well as substitutions. Fresh validator
    nonduplicate status supplies novelty headroom; this additional separation
    keeps reserved families distinct. Ported from
    ``nanobody_opt_v2/select_elite_parents.py::_same_cdr3_family``.
    """
    if not left or not right:
        return False
    if fraction < 0:
        raise ValueError("fraction must be nonnegative")
    previous = list(range(len(right) + 1))
    for index, residue in enumerate(left, 1):
        current = [index]
        for column, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (residue != other)))
        previous = current
    return previous[-1] <= fraction * max(len(left), len(right))


def _family_buckets(
    items: list[dict[str, Any]],
    *,
    cdr3_of: Callable[[dict[str, Any]], str],
    fraction: float,
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Group ``items`` into first-seen CDR3-family buckets.

    Returns ``(order, buckets)`` where ``order`` is the first-seen family key
    list and ``buckets[key]`` preserves the input order of that family's items.
    An empty/missing CDR3 forms one shared unknown family, so every item lands
    in exactly one bucket.
    """
    order: list[str] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        cdr3 = (cdr3_of(item) or "").strip()
        key = next((k for k in order if same_cdr3_family(k, cdr3, fraction=fraction)),
                   cdr3)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(item)
    return order, buckets


def family_round_robin(
    items: list[dict[str, Any]],
    *,
    cdr3_of: Callable[[dict[str, Any]], str],
    fraction: float = 0.15,
) -> list[dict[str, Any]]:
    """Cycle distinct CDR3 families before revisiting a family.

    Mirrors ``nanobody_opt_v2/run_live_producer.py::_historical_child_fallback``:
    ``[family[i] for i in ... for family in families.values() if i < len(family)]``
    -- a round robin over per-family buckets in first-seen family order, so the
    leading family cannot fill the head of the pool on its own. Within a family
    the original relative order is preserved.

    ``cdr3_of`` extracts the CDR3 string from an item (e.g. an
    ``extract_cdrs`` wrapper or ``regions.find_regions`` slice).
    """
    order, buckets = _family_buckets(items, cdr3_of=cdr3_of, fraction=fraction)
    width = max((len(bucket) for bucket in buckets.values()), default=0)
    return [buckets[key][i] for i in range(width) for key in order
            if i < len(buckets[key])]


def family_count(
    items: list[dict[str, Any]],
    *,
    cdr3_of: Callable[[dict[str, Any]], str],
    fraction: float = 0.15,
) -> int:
    """Number of distinct CDR3 families among ``items`` (see ``same_cdr3_family``)."""
    order, _ = _family_buckets(items, cdr3_of=cdr3_of, fraction=fraction)
    return len(order)
