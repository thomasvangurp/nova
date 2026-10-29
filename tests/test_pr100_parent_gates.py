"""Unit tests for the v2 parent/donor gates ported into pr100_parent_gates.py.

These are pure functions: no abnumber, no network, no DB. The fixtures use
synthetic CDR3 strings so the tests also run without any scientific stack.
"""
import pytest

from nanobody_opt import pr100_parent_gates as gates


# ---------------------------------------------------------------------------
# cdr_developability_in_range -- CDR-only metanano red-region bounds.
# ---------------------------------------------------------------------------
IN_RANGE = {"total_cdr_length": 30.0, "cdr3_length": 14.0, "cdr3_compactness": 1.0}


def test_red_region_constant_matches_v2_values():
    assert gates.CDR_DEVELOPABILITY_RED_REGION == {
        "total_cdr_length": (20.0, 39.0),
        "cdr3_length": (5.0, 23.0),
        "cdr3_compactness": (0.56, 1.61),
    }


def test_in_range_profile_passes():
    assert gates.cdr_developability_in_range(IN_RANGE) is True


@pytest.mark.parametrize("field,bad", [
    ("total_cdr_length", 19.9),
    ("total_cdr_length", 39.1),
    ("cdr3_length", 4.9),
    ("cdr3_length", 23.1),
    ("cdr3_compactness", 0.559),
    ("cdr3_compactness", 1.611),
])
def test_out_of_range_red_flag_fails(field, bad):
    profile = dict(IN_RANGE)
    profile[field] = bad
    assert gates.cdr_developability_in_range(profile) is False


@pytest.mark.parametrize("boundary", [
    {"total_cdr_length": 20.0, "cdr3_length": 5.0, "cdr3_compactness": 0.56},
    {"total_cdr_length": 39.0, "cdr3_length": 23.0, "cdr3_compactness": 1.61},
])
def test_red_region_bounds_are_inclusive(boundary):
    assert gates.cdr_developability_in_range(boundary) is True


@pytest.mark.parametrize("profile", [
    None,
    "not-a-dict",
    {"total_cdr_length": 30.0, "cdr3_length": 14.0},          # missing compactness
    {"total_cdr_length": "abc", "cdr3_length": 14.0, "cdr3_compactness": 1.0},
    {"total_cdr_length": 30.0, "cdr3_length": None, "cdr3_compactness": 1.0},
])
def test_malformed_profile_fails_closed(profile):
    assert gates.cdr_developability_in_range(profile) is False


# ---------------------------------------------------------------------------
# same_cdr3_family -- length-normalised Levenshtein grouping.
# ---------------------------------------------------------------------------
def test_identical_cdr3s_are_same_family():
    assert gates.same_cdr3_family("VFLGAHPCPLE", "VFLGAHPCPLE") is True


def test_empty_cdr3_is_never_same_family():
    assert gates.same_cdr3_family("", "") is False
    assert gates.same_cdr3_family("A", "") is False
    assert gates.same_cdr3_family("", "A") is False


def test_one_edit_within_fraction_groups_for_long_cdr3():
    # Two 18-residue CDR3s one edit apart: 1 <= 0.15 * 18 = 2.7.
    left = "ABCDEFGHIJKLMNOPQR"
    right = "ABCDEFGHIJKLMNOPQS"
    assert gates.same_cdr3_family(left, right) is True


def test_short_cdr3_requires_exact_match_at_default_fraction():
    # A 5-residue CDR3 one edit apart: 1 <= 0.15 * 5 = 0.75 is false.
    assert gates.same_cdr3_family("RAFDY", "RHFDY") is False


def test_zero_fraction_requires_identity():
    assert gates.same_cdr3_family("RAFDY", "RAFDY", fraction=0.0) is True
    assert gates.same_cdr3_family("RAFDY", "RAFDX", fraction=0.0) is False


def test_negative_fraction_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        gates.same_cdr3_family("A", "A", fraction=-0.1)


def test_insertion_deletion_is_handled():
    # One deletion keeps the distance at 1, so a long family still groups.
    assert gates.same_cdr3_family("ABCDEFGHIJKLMNOPQR",
                                  "ABCDEFGHIJKLMNOPQ") is True


# ---------------------------------------------------------------------------
# family_round_robin / family_count -- diversity helpers over item dicts.
# ---------------------------------------------------------------------------
def _item(name):
    return {"name": name}


def _cdr3_of(item):
    return item["name"].split("-", 1)[0]


def test_round_robin_cycles_families_before_revisiting():
    items = [_item(f) for f in ("A-1", "A-2", "A-3", "B-1", "B-2", "C-1")]
    ordered = gates.family_round_robin(items, cdr3_of=_cdr3_of)
    assert [i["name"] for i in ordered] == ["A-1", "B-1", "C-1", "A-2", "B-2", "A-3"]


def test_round_robin_preserves_within_family_order():
    items = [_item(f) for f in ("B-1", "A-1", "B-2", "A-2")]
    ordered = gates.family_round_robin(items, cdr3_of=_cdr3_of)
    assert [i["name"] for i in ordered] == ["B-1", "A-1", "B-2", "A-2"]


def test_round_robin_uses_first_seen_family_key():
    # A-2 and A-1 are the same family; the first-seen key is "A" via A-1.
    items = [_item(f) for f in ("A-1", "A-2", "B-1")]
    assert gates.family_count(items, cdr3_of=_cdr3_of) == 2


def test_round_robin_empty_input_is_empty():
    assert gates.family_round_robin([], cdr3_of=_cdr3_of) == []


def test_round_robin_single_family_keeps_input_order():
    items = [_item(f) for f in ("A-2", "A-1", "A-3")]
    ordered = gates.family_round_robin(items, cdr3_of=_cdr3_of)
    assert [i["name"] for i in ordered] == ["A-2", "A-1", "A-3"]


def test_missing_cdr3_forms_one_unknown_family():
    items = [_item("A-1"), _item("A-2"), _item("")]
    ordered = gates.family_round_robin(items, cdr3_of=lambda i: i["name"].split("-", 1)[0])
    # "" and "A" are not the same family, so "" interleaves as its own bucket.
    assert [i["name"] for i in ordered] == ["A-1", "", "A-2"]


def test_family_count_empty_is_zero():
    assert gates.family_count([], cdr3_of=_cdr3_of) == 0


def test_family_count_groups_near_duplicates():
    items = [_item(f) for f in ("A", "A", "A-X", "B", "B")]
    # A and A-X are one edit apart: with fraction=0 they split, so A/A group
    # and A-X/B stay distinct.
    assert gates.family_count(items, cdr3_of=lambda i: i["name"], fraction=0.0) == 3
