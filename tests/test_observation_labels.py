"""Label registry (ARCHITECTURE_REDESIGN P4/R3): a score_kind names one scale."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from nanobody_opt import observation_labels as L

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mp():
    return _load("multipass_all_rivals_t", ROOT / "tools" / "multipass_all_rivals.py")


def test_golden_outputs_match_the_validation_gate_builtins():
    """GOLDEN_OUTPUTS are the same numbers the scoring-change gate computes."""
    vsc = _load("validate_scoring_change_t", ROOT / "tools" / "validate_scoring_change.py")
    field = L.golden_field()
    for formula, fn in vsc.FIELD_BUILTINS.items():
        got = [float(x) for x in fn(field)]
        want = list(L.GOLDEN_OUTPUTS[formula])
        assert [repr(x) for x in got] == [repr(x) for x in want], formula


def test_multipass_writer_passes_its_formula_guard(mp):
    mp.assert_pass_label()
    assert L.LABELS[mp.PASS_SCORE_KIND].formula == mp.PASS_FORMULA_ID


def test_formula_change_under_the_same_label_is_refused(mp):
    # The 2026-10-09 failure: algo2 -> category3 -> sum10 under one label.
    with pytest.raises(L.LabelViolation):
        L.assert_formula(mp.PASS_SCORE_KIND, "sum10", mp.algo2_score)
    with pytest.raises(L.LabelViolation):
        L.assert_formula(mp.PASS_SCORE_KIND, "category3", mp.sum_of_ten)
    with pytest.raises(L.LabelViolation):
        L.assert_formula("multipass_sum11_pass", "sum10", mp.sum_of_ten)


def test_multipass_pass_rows_are_accepted(mp):
    rows = mp.pass_observations(
        epoch=25700, run=1, passes=10, ranks={"QVQL": 120000.0},
        raw={"QVQL": {"design_ptm": 0.9}}, structs={}, cohort_of={},
        field={"field": "hf_archive_full", "field_rows": 1, "field_sha256": "x"})
    L.check_observations(rows)  # no raise


def test_strict_label_refuses_wrong_writer_formula_and_closed_label():
    base = {"source": "public_multipass_rivals", "score_kind": "multipass_sum10_pass",
            "writer": "multipass_all_rivals", "formula": "sum10"}
    L.check_observations([base])
    with pytest.raises(L.LabelViolation):
        L.check_observations([{**base, "formula": "category3"}])
    with pytest.raises(L.LabelViolation):
        L.check_observations([{**base, "writer": "someone_else"}])
    with pytest.raises(L.LabelViolation):
        L.check_observations([{**base, "source": "live_exact_screen"}])
    legacy = {**base, "score_kind": "legacy_multipass_algo2_pass", "formula": "algo2"}
    with pytest.raises(L.LabelViolation):
        L.check_observations([{**legacy, "writer": "multipass_all_rivals"}])
    L.check_observations([{**legacy, "writer": L.RELABEL_WRITER}])


def test_old_multipass_write_shape_is_refused():
    """The pre-registry multipass write (producer label, no provenance) would
    put ~1e5 values among the producer's ~200 scale: refuse it."""
    old = {"source": "public_multipass_rivals",
           "score_kind": "local_validator_replay_rank_sum", "score": 120000.0}
    assert L.violations(old)  # source not registered for the producer label
    # warn mode: does not raise without NOVA_LABEL_ENFORCE
    L.check_observations([old])


def test_enforce_env_makes_warnings_hard(monkeypatch):
    monkeypatch.setenv("NOVA_LABEL_ENFORCE", "1")
    with pytest.raises(L.LabelViolation):
        L.check_observations([{"source": "x", "score_kind": "brand_new_kind"}])


def test_every_score_kind_literal_written_in_code_is_registered():
    pattern = re.compile(r'(?:"score_kind"\s*:\s*|SCORE_KIND\s*=\s*)"([a-z0-9_]+)"')
    found = set()
    for folder in ("nanobody_opt", "tools"):
        for path in (ROOT / folder).rglob("*.py"):
            found.update(pattern.findall(path.read_text(errors="ignore")))
    unregistered = sorted(k for k in found if k not in L.LABELS)
    assert not unregistered, (
        f"score_kind(s) {unregistered} are written but not registered in "
        "nanobody_opt/observation_labels.py")


def test_generated_doc_is_current():
    doc = (ROOT / "docs" / "OBSERVATION_LABELS.md").read_text()
    assert doc == L.markdown(), (
        "regenerate: python3 -m nanobody_opt.observation_labels --markdown "
        "> docs/OBSERVATION_LABELS.md")


def test_relabel_regimes_are_disjoint_and_registered():
    relabel = _load("relabel_multipass_legacy_t", ROOT / "tools" / "relabel_multipass_legacy.py")
    seen: set[int] = set()
    for formula, (kind, epochs, *_rest) in relabel.REGIMES.items():
        assert L.LABELS[kind].formula == formula
        assert not (seen & epochs)
        seen |= epochs
    assert relabel.fingerprint_ok("algo2", {"n": 500, "mean": 37000, "nonint": 230})[0]
    assert not relabel.fingerprint_ok("algo2", {"n": 500, "mean": 52000, "nonint": 0})[0]
    assert relabel.fingerprint_ok("category3", {"n": 500, "mean": 52500, "nonint": 0})[0]
    assert not relabel.fingerprint_ok("sum10", {"n": 500, "mean": 37000, "nonint": 230})[0]


def test_dashboard_never_joins_measurement_regimes(tmp_path):
    go = pytest.importorskip("plotly.graph_objects")  # noqa: F841
    from nanobody_dash import progress

    def actor(mean):
        return {"mean": mean, "stdev": 10.0, "fleet_size": 3, "scores": [mean]}

    payload = {}
    for epoch in range(25608, 25617):
        row = {"ours": actor(120000.0), "rivals": {"5GhREcsKxyz": actor(119000.0)},
               "sequences": {}}
        if epoch >= 25615:
            row["method"] = {"formula": "sum10",
                             "measurement": "local_fast_scorer_multipass_mean"}
        payload[str(epoch)] = row
    path = tmp_path / "multipass_comparison.json"
    path.write_text(json.dumps(payload))

    for fn in (progress.fig_multipass_vs_rival, progress.fig_multipass_strongest):
        fig = fn(path)
        us = fig.data[0]
        gap_x = [x for x, y in zip(us.x, us.y) if y is None]
        assert gap_x == [25612.5], fn.__name__  # 25613+ inferred, 25615+ explicit: same
        assert [s.x0 for s in fig.layout.shapes] == [25612.5]
        texts = [a.text for a in fig.layout.annotations]
        assert any("validator-published" in t for t in texts)
        assert any("multi-pass mean" in t for t in texts)
    dist = progress.fig_multipass_distribution(path)
    assert len(dist.layout.shapes) == 1
