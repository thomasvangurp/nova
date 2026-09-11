from nanobody_opt import metric_outliers as mo


def _field(hbonds, pae):
    return [
        {"plip_hbonds_refolded": h, "min_design_to_target_pae": p}
        for h, p in zip(hbonds, pae, strict=True)
    ]


METRICS = {"plip_hbonds_refolded": "max", "min_design_to_target_pae": "min"}


def test_worst_metric_drives_the_verdict():
    field = mo.field_distribution(_field([2, 4, 4, 6, 8], [2.2, 2.3, 2.3, 2.4, 2.5]),
                                  METRICS)
    # excellent on hbonds, near-last on PAE: the PAE decides
    fraction, culprit = mo.worst_percentile(
        {"plip_hbonds_refolded": 8.0, "min_design_to_target_pae": 2.5}, field, METRICS
    )
    assert culprit == "min_design_to_target_pae"
    assert fraction >= 0.8


def test_a_balanced_candidate_survives():
    field = mo.field_distribution(_field([2, 4, 4, 6, 8], [2.2, 2.3, 2.3, 2.4, 2.5]),
                                  METRICS)
    fraction, _ = mo.worst_percentile(
        {"plip_hbonds_refolded": 6.0, "min_design_to_target_pae": 2.25}, field, METRICS
    )
    assert fraction <= 0.4


def test_the_epoch_25061_candidate_is_rejected():
    # Two hbonds against a field median of four, PAE near the bottom.  A field of
    # 48 puts 47th place at 0.96, so the live threshold of 0.85 rejects it; this
    # toy field of five caps the worst fraction at 0.8, hence 0.75 here.
    field = mo.field_distribution(_field([2, 4, 4, 5, 6], [2.29, 2.33, 2.34, 2.40, 2.43]),
                                  METRICS)
    kept, dropped = mo.without_catastrophic_outliers(
        [("BAD", {"plip_hbonds_refolded": 2.0, "min_design_to_target_pae": 2.43}),
         ("OK", {"plip_hbonds_refolded": 5.0, "min_design_to_target_pae": 2.30})],
        field, maximum_percentile=0.75, minimum_keep=1, metrics=METRICS,
    )
    assert kept == ["OK"]
    assert dropped and dropped[0][0] == "BAD"


def test_the_live_threshold_rejects_a_real_47th_of_48():
    # 47th of 48 is 0.958 of the field beating the candidate
    field = mo.field_distribution(
        _field([4] * 48, [2.30 + 0.002 * i for i in range(48)]), METRICS
    )
    fraction, culprit = mo.worst_percentile(
        {"plip_hbonds_refolded": 4.0, "min_design_to_target_pae": 2.30 + 0.002 * 46},
        field, METRICS,
    )
    assert culprit == "min_design_to_target_pae"
    assert fraction > 0.85


def test_the_guard_never_empties_the_shortlist():
    field = mo.field_distribution(_field([6, 7, 8], [2.1, 2.1, 2.2]), METRICS)
    everyone_bad = [
        (f"C{i}", {"plip_hbonds_refolded": 1.0, "min_design_to_target_pae": 2.9})
        for i in range(4)
    ]

    kept, _ = mo.without_catastrophic_outliers(
        everyone_bad, field, maximum_percentile=0.5, minimum_keep=2, metrics=METRICS
    )

    assert len(kept) == 2
