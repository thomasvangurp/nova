"""Decision-rule tests. PSH noise makes a single measurement at the cap worthless."""
import math

import pytest

from nanobody_opt import pr100_live_selection as selection


def measurement(psh, *, ok=True):
    return {'surface_hydrophobic_patches': psh, 'ok': ok}


def reason_for(value):
    """Stand-in gate: mirrors 'every other gate' without touching real thresholds."""
    return '' if value.get('ok') else 'developability_red_flags'


def decide(values, **kwargs):
    kwargs.setdefault('reason_for', reason_for)
    return selection.decide([measurement(v) for v in values], {}, **kwargs)


class TestThreshold:
    def test_k_scaled_rule_matches_audit_figures(self):
        # cap 89, z 1.3, sigma 4.4 -> ~85.0 at k=2 and ~85.7 at k=3 statistically,
        # so the hard margin (80 / 82) is what binds.
        assert selection.margin_threshold(2) == 80
        assert selection.margin_threshold(3) == 82

    def test_single_measurement_falls_back_to_the_strictest_hard_margin(self):
        # k=1 has no hard_margin entry, so the smallest (k=2, 80) binds. Tier A is
        # refused at k=1 regardless by min_repeats (see TestTiers).
        assert selection.margin_threshold(1) == 80

    def test_constants_are_configurable(self):
        # Statistical term at k=3 is 89 - 1.3*4.4/sqrt(3) = 85.70, so a hard
        # margin of 85 binds and a hard margin of 90 does not.
        assert selection.margin_threshold(3, {'hard_margin': {'3': 85}}) == 85
        assert selection.margin_threshold(3, {'hard_margin': {'3': 90}}) == pytest.approx(
            89 - 1.3 * 4.4 / math.sqrt(3))
        assert selection.margin_threshold(3, {'cap': 100, 'z': 0, 'hard_margin': {'3': 90}}) == 90

    def test_counts_above_the_table_reuse_the_largest_entry(self):
        assert selection.margin_threshold(5) == 82

    def test_lower_margin_tracks_the_validator_lower_boundary(self):
        assert selection.margin_lower_threshold(2) == pytest.approx(73.4 + 1.3 * 4.4 / math.sqrt(2))
        assert selection.margin_lower_threshold(3) == pytest.approx(73.4 + 1.3 * 4.4 / math.sqrt(3))

    @pytest.mark.parametrize('bad', [{'min_repeats': 1}, {'sigma': float('nan')},
                                     {'z': -1}, {'nonsense': 1}, {'hard_margin': {}}])
    def test_invalid_margin_parameters_refuse(self, bad):
        with pytest.raises(ValueError):
            selection.margin_settings(bad)


class TestTiers:
    def test_single_88_is_not_margin_safe(self):
        # The audit case: one measurement at 88 is noise hugging the cap.
        verdict = decide([88])
        assert verdict['tier'] != 'A'
        assert verdict['margin_safe'] is False

    def test_single_measurement_never_tier_a_even_when_low(self):
        verdict = decide([70])
        assert verdict['margin_safe'] is False
        assert verdict['reason'] == 'insufficient_repeats'

    def test_mean_81_over_three_repeats_is_margin_safe(self):
        verdict = decide([78, 81, 84])
        assert verdict['psh_mean'] == pytest.approx(81.0)
        assert verdict['tier'] == 'A' and verdict['margin_safe'] is True
        assert verdict['margin_threshold'] == 82

    def test_mean_85_over_three_repeats_is_not_tier_a(self):
        verdict = decide([83, 85, 87])
        assert verdict['psh_mean'] == pytest.approx(85.0)
        assert verdict['tier'] == 'B'
        assert verdict['margin_safe'] is False
        assert verdict['reason'] == 'margin_missed'

    @pytest.mark.parametrize('psh', [73.5184, 74.4958, 75.0])
    def test_repeated_lower_boundary_passes_are_not_margin_safe(self, psh):
        verdict = decide([psh, psh, psh])
        assert verdict['tier'] == 'B'
        assert verdict['margin_safe'] is False
        assert verdict['reason'] == 'margin_missed'
        assert verdict['margin_lower_threshold'] > psh

    def test_two_measurements_can_still_qualify_inside_both_margins(self):
        verdict = decide([78.0, 80.0])
        assert verdict['tier'] == 'A'
        assert verdict['psh_boundary_margin'] == pytest.approx(5.6)

    def test_any_repeat_failing_a_gate_rejects_outright(self):
        verdict = selection.decide(
            [measurement(70), measurement(71, ok=False), measurement(70)],
            {}, reason_for=reason_for)
        assert verdict['tier'] is None
        assert verdict['reason'] == 'developability_red_flags'

    def test_no_measurement_is_not_run_not_failed(self):
        verdict = decide([])
        assert verdict['reason'] == 'not_run' and verdict['tier'] is None

    def test_raw_measurements_mean_and_sd_are_recorded(self):
        verdict = decide([78, 81, 84])
        assert verdict['psh_values'] == [78, 81, 84]
        assert verdict['repeats'] == 3
        assert verdict['psh_sd'] == pytest.approx(3.0)
        assert len(verdict['measurements']) == 3

    def test_unusable_psh_is_not_silently_averaged(self):
        verdict = decide([78, None, 80])
        assert verdict['psh_mean'] is None
        assert verdict['reason'] == 'psh_unavailable'
        assert verdict['tier'] is None

    def test_real_gate_semantics_are_used_by_default(self):
        from nanobody_opt import rules
        requirements = {'psh_strict_rejection_threshold': 89}
        verdict = selection.decide([{'surface_hydrophobic_patches': 95}], requirements)
        # Delegates to rules.developability_reason rather than re-deciding.
        assert verdict['tier'] is None
        assert verdict['reason'] == rules.developability_reason(
            {'surface_hydrophobic_patches': 95}, requirements)


class TestOrdering:
    def rows(self):
        return [
            {'sequence_hash': 'b', 'tier': 'B', 'psh_mean': 83.0},
            {'sequence_hash': 'a', 'tier': 'A', 'psh_mean': 81.0},
            {'sequence_hash': 'c', 'tier': 'A', 'psh_mean': 75.0},
            {'sequence_hash': 'd', 'tier': None, 'psh_mean': 60.0},
        ]

    def test_tier_a_first_then_margin_order(self):
        ordered = selection.order_candidates(self.rows())
        assert [r['sequence_hash'] for r in ordered] == ['a', 'c', 'b']

    def test_rejected_rows_are_never_published(self):
        assert all(r['tier'] in ('A', 'B') for r in selection.order_candidates(self.rows()))

    def test_allow_marginal_false_drops_tier_b(self):
        ordered = selection.order_candidates(self.rows(), allow_marginal=False)
        assert [r['sequence_hash'] for r in ordered] == ['a', 'c']

    def test_prefix_is_monotonic_in_two_sided_margin_within_tier(self):
        ordered = selection.order_candidates(self.rows())
        margins = [selection.psh_margin(r['psh_mean']) for r in ordered if r['tier'] == 'A']
        assert margins == sorted(margins, reverse=True)

    def test_both_lower_and_upper_cap_huggers_lose_to_interior(self):
        rows = [{'sequence_hash': str(value), 'tier': 'B', 'psh_mean': value}
                for value in (73.5184, 88.5286, 81.2, None)]
        assert [r['psh_mean'] for r in selection.order_candidates(rows)] == [81.2, 88.5286, 73.5184, None]


class TestShard:
    def test_candidates_spread_across_every_gpu(self):
        shards = selection.shard([{'i': i} for i in range(5)], [0, 1])
        assert [r['i'] for r in shards[0]] == [0, 2, 4]
        assert [r['i'] for r in shards[1]] == [1, 3]

    def test_single_gpu_keeps_everything(self):
        shards = selection.shard([{'i': 0}, {'i': 1}], [3])
        assert len(shards[3]) == 2

    def test_no_eight_candidate_cap(self):
        shards = selection.shard([{'i': i} for i in range(40)], [0, 1, 2])
        assert sum(len(v) for v in shards.values()) == 40

    @pytest.mark.parametrize('gpus', [[], [0, 0], [-1], [True], 'zero'])
    def test_invalid_gpu_lists_refuse(self, gpus):
        with pytest.raises(ValueError):
            selection.shard([{'i': 0}], gpus)
