"""Transport/label regression tests; scientific evidence is a mocked boundary."""
import copy
import math

import pytest

from nanobody_opt import rules
from nanobody_opt.pr100_live_trial_export import build
from nanobody_opt.simulate_round import METRICS


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(rules, "validate_evidence", lambda *args: None)
    challenge = rules.challenge(rules.pinned_config(25956), 25956, "0x" + "a" * 64, 361)
    candidates = [{"sequence": "A" * 120, "layer2_passed": True, "novelty": {"validator_novel": True, "exact_historical": False, "coarse_candidates": 7, "maximum_identity": 0.81},
                   "pr100_evidence": {"mocked_scientific_boundary": True}},
                  {"sequence": "C" * 120, "layer2_passed": True, "novelty": {"validator_novel": True, "exact_historical": False, "coarse_candidates": 7, "maximum_identity": 0.81},
                   "pr100_evidence": {"mocked_scientific_boundary": True}}]
    validity = {"challenge": challenge, "target": rules.TARGET, "candidates": candidates,
                "archive_sha256": "c" * 64, "archive_checked_at_unix": 1.}
    scores = {"target": rules.TARGET, "per_nanobody_components": {"0": {
        row["sequence"]: {rules.TARGET: {key: 1. for key in METRICS}} for row in candidates}}}
    return validity, scores, {"fold_steps": 35, "diffusion_samples": 5}


def export(inputs):
    return build(*inputs, scorer_sha256="a" * 64, settings_file_sha256="b" * 64, now=2.)


def test_partial_fleet_is_honest_and_retains_ties(inputs):
    fleet = export(inputs)
    assert fleet["sequence_pool_size"] == 2
    assert fleet["publication_stage"] == "live_trial"
    assert fleet["producer"]["name"] == "pr100_live_collector"
    for checkpoint in fleet["sequence_pool_checkpoints"].values():
        assert checkpoint["selection"]["score"] == 10.
        assert checkpoint["exact"]["observation_count"] == 1
        measured = checkpoint["validation"]["boltzgen"]
        assert measured["exact"] is measured["runtime_matched"] is measured["validator_matched"] is False
        assert measured["provenance"]["settings_sha256"] == rules.digest(inputs[2])
        novelty = checkpoint["validation"]["layer_1"]["novelty"]
        assert novelty == {"validator_novel": True, "exact_historical": False,
                           "coarse_candidates": 7, "maximum_identity": 0.81,
                           "sequence_hash": rules.sequence_hash(checkpoint["sequences"][0]),
                           "target": rules.TARGET, "archive_sha256": inputs[0]["archive_sha256"]}


def test_valid_but_unscored_candidate_is_not_exported(inputs):
    inputs[1]["per_nanobody_components"]["0"].pop("C" * 120)
    assert export(inputs)["sequence_pool_size"] == 1


@pytest.mark.parametrize("fault", ["target", "novelty", "duplicate", "nonfinite", "incomplete", "empty"])
def test_bad_or_missing_evidence_refuses_export(inputs, fault):
    validity, scores, _ = inputs
    sequence = validity["candidates"][0]["sequence"]
    if fault == "target": scores["target"] = "P20809"
    elif fault == "novelty": validity["candidates"][0]["novelty"]["validator_novel"] = False
    elif fault == "duplicate": validity["candidates"].append(copy.deepcopy(validity["candidates"][0]))
    elif fault == "nonfinite": scores["per_nanobody_components"]["0"][sequence][rules.TARGET]["design_iiptm"] = math.nan
    elif fault == "incomplete": scores["per_nanobody_components"]["0"][sequence][rules.TARGET].pop("design_iiptm")
    elif fault == "empty": scores["per_nanobody_components"] = {}
    with pytest.raises(ValueError): export(inputs)


@pytest.mark.parametrize("archive", [None, "", "not-a-digest", "D" * 64])
def test_export_requires_actual_archive_identity(inputs, archive):
    inputs[0]["archive_sha256"] = archive
    with pytest.raises(ValueError, match="archive identity"):
        export(inputs)


def test_export_does_not_turn_historical_match_into_novelty_pass(inputs):
    inputs[0]["candidates"][0]["novelty"]["exact_historical"] = True
    with pytest.raises(ValueError, match="novelty_not_passed"):
        export(inputs)
