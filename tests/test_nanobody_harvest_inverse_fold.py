import pytest

from nanobody_opt.harvest_inverse_fold import choose_designed_sequence


def test_selects_vhh_polymer_by_teacher_length():
    assert choose_designed_sequence(
        ["AAAA AAAA", "ACD\nEFG"], teacher="QQQQQQ"
    ) == "ACDEFG"


def test_rejects_ambiguous_vhh_polymer():
    with pytest.raises(ValueError, match="expected one"):
        choose_designed_sequence(["AAAA", "CCCC"], teacher="QQQQ")
