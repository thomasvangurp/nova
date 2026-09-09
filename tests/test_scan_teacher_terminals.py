from nanobody_opt.scan_teacher_terminals import scan_variants, terminal_positions


REQUIREMENTS = {
    "min_sequence_length": 4,
    "max_sequence_length": 20,
    "max_homopolymer_run": 5,
    "max_di_repeat_pairs": 5,
    "min_cysteines": 0,
    "cys_pair_min_separation": 1,
    "cys_pair_max_separation": 20,
    "reject_signal_peptides": False,
    "sp_window": 4,
    "sp_hydro_min_in_window": 4,
    "sp_scan_prefix": 4,
}


def test_terminal_positions_do_not_overlap():
    assert terminal_positions(10, 2) == [0, 1, 8, 9]


def test_scan_excludes_teacher_history_and_failed_diversity():
    teacher = "ACDEFGHI"
    historical = {"DCDEFGHI"}

    def diversity(sequence: str):
        return sequence != "ECDEFGHI", 0.4

    cases = scan_variants(
        teacher,
        width=1,
        existing=historical,
        requirements=REQUIREMENTS,
        passes_top50=diversity,
    )
    sequences = {case["sequence"] for case in cases}
    assert teacher not in sequences
    assert "DCDEFGHI" not in sequences
    assert "ECDEFGHI" not in sequences
    assert all(case["max_jaccard_to_top50"] == 0.4 for case in cases)
    assert all(case["mutation"][1:-1] in {"1", "8"} for case in cases)
