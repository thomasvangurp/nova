from nanobody_opt.competitor_position_lane import (
    competitor_position_candidates,
    learn_hot_positions,
    position_restricted_candidates,
)


def test_learn_hot_positions_ranks_by_frequency_and_respects_the_epoch_cutoff():
    submissions = [
        (100, 1, [1]),
        (101, 1, [1]),
        (102, 1, [3]),
        (103, 1, [1]),
        (104, 1, [121]),  # strictly before cutoff, counts
        (105, 1, [3]),    # at/after cutoff, excluded
        (106, 1, [3]),    # at/after cutoff, excluded
    ]
    assert learn_hot_positions(submissions, before_epoch=105, top_n=3) == [1, 3, 121]


def test_learn_hot_positions_excludes_multi_edit_and_identical_rows():
    submissions = [
        (100, 0, []),           # identical resubmission -- excluded
        (101, 2, [1, 3]),       # double edit -- excluded (ambiguous)
        (102, 1, [7]),
        (103, 1, [7]),
    ]
    assert learn_hot_positions(submissions, before_epoch=200, top_n=5) == [7]


def test_position_restricted_candidates_only_touches_listed_positions():
    parent = "AAAAAAAAAA"
    allowed = [["A", "C"], ["A"], ["A"], ["A", "D"], *([["A"]] * 6)]

    candidates = position_restricted_candidates(parent, allowed=allowed, positions=[1, 4])

    assert candidates == {
        "CAAAAAAAAA": 1,
        "AAADAAAAAA": 4,
    }


def test_position_restricted_candidates_skips_out_of_range_and_unsupported_positions():
    parent = "AAA"
    allowed = [["A"], ["A"], ["A"]]

    candidates = position_restricted_candidates(parent, allowed=allowed, positions=[1, 99])

    assert candidates == {}


def test_position_restricted_candidates_rejects_disallowed_residues():
    parent = "AAA"
    allowed = [["A", "B"], ["A"], ["A"]]  # 'B' is not a valid amino acid

    candidates = position_restricted_candidates(parent, allowed=allowed, positions=[1])

    assert candidates == {}


def test_competitor_position_candidates_pools_across_parents_first_wins_on_collision():
    allowed = [["A", "C", "G"], ["A"], ["T"]]
    # both parents can reach "CAT" by substituting position 1 to 'C'
    parents = ["AAT", "GAT"]

    results = competitor_position_candidates(parents, allowed=allowed, positions=[1])

    assert "CAT" in results
    assert results["CAT"]["parent"] == "AAT"
    assert results["CAT"]["position"] == 1
    # each parent also contributes the OTHER parent's original as a candidate
    assert set(results) == {"CAT", "GAT", "AAT"}
