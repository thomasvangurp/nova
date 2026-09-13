"""Donor grafts entering the producer's proposals.

Neighbourhood search explores one molecule's surroundings, so its pool is one
molecule: across epochs 25093-25095 every candidate sat within 0.93 identity of
every other and the validator scored three of our seven submissions. Grafts are
the scaffold diversity that pool cannot reach, and they must arrive before
screening so they earn their scores on the same evidence as anything else.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt.run_live_producer import GRAFT_PROPOSAL_TYPE, merge_graft_candidates

FIELDS = [
    "sequence", "parent_sequence", "proposal_type", "proposal_lane",
    "edit_distance", "parent_max_jaccard_to_top50", "parent_selection_score",
]
TEACHER = "KWALVESGGGLVKPGGSLRLSCAASGFTASRGWYGMSWFRQ"
NATIVE = "KWMLVESGGGLVKPGGSLRLSCAASGFTASRGWYGMSWFRQ"


def _candidates(path, sequences=(NATIVE,)):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sequence in sequences:
            row = {name: "" for name in FIELDS}
            row["sequence"] = sequence
            row["parent_selection_score"] = "38.0"
            writer.writerow(row)
    return path


def _grafts(path, sequences):
    path.write_text(json.dumps({
        "teacher_sequence": TEACHER,
        "cases": [
            {"sequence": s, "mutation_count_from_teacher": 17,
             "max_jaccard_to_top50": 0.41}
            for s in sequences
        ],
    }))
    return path


def test_grafts_join_the_proposals_without_disturbing_them(tmp_path):
    candidates = _candidates(tmp_path / "candidates.csv")
    grafts = _grafts(tmp_path / "grafts.json", ["DWTLVESGGGGVAPGGSLRLSCAGSGFTASRG"])

    added = merge_graft_candidates(candidates, grafts)

    rows = list(csv.DictReader(candidates.open(newline="")))
    assert added == 1
    assert [r["sequence"] for r in rows] == [
        NATIVE, "DWTLVESGGGGVAPGGSLRLSCAGSGFTASRG"
    ]
    assert rows[0]["parent_selection_score"] == "38.0", "native rows were rewritten"
    graft = rows[1]
    assert graft["proposal_type"] == GRAFT_PROPOSAL_TYPE
    assert graft["parent_sequence"] == TEACHER
    assert graft["edit_distance"] == "17"


def test_a_graft_already_proposed_does_not_take_a_second_row(tmp_path):
    """Screening cost is per proposal; the same molecule twice buys nothing."""
    candidates = _candidates(tmp_path / "candidates.csv", [NATIVE, "DWTL"])
    grafts = _grafts(tmp_path / "grafts.json", ["DWTL", "DWQL"])

    assert merge_graft_candidates(candidates, grafts) == 1
    rows = list(csv.DictReader(candidates.open(newline="")))
    assert [r["sequence"] for r in rows] == [NATIVE, "DWTL", "DWQL"]


def test_the_columns_the_screen_reads_are_preserved_exactly(tmp_path):
    candidates = _candidates(tmp_path / "candidates.csv")
    grafts = _grafts(tmp_path / "grafts.json", ["DWQL"])
    merge_graft_candidates(candidates, grafts)
    with candidates.open(newline="") as handle:
        assert csv.DictReader(handle).fieldnames == FIELDS


def test_a_missing_or_empty_graft_file_stops_the_iteration(tmp_path):
    candidates = _candidates(tmp_path / "candidates.csv")
    with pytest.raises(SystemExit, match="missing"):
        merge_graft_candidates(candidates, tmp_path / "absent.json")
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"cases": []}))
    with pytest.raises(SystemExit, match="no graft cases"):
        merge_graft_candidates(candidates, empty)


def test_a_graft_pool_already_built_for_this_epoch_is_reused(tmp_path, monkeypatch):
    """One graft build per epoch: the archive does not change under us mid-epoch
    and the build is not free."""
    from nanobody_opt import run_live_producer as producer

    epoch_root = tmp_path / "epoch1"
    epoch_root.mkdir()
    existing = _grafts(epoch_root / "donor_grafts.json", ["DWTL"])
    calls = []
    monkeypatch.setattr(producer, "_run", lambda command: calls.append(command))

    result = producer.refresh_donor_grafts(
        epoch_root=epoch_root, archive=tmp_path / "a.csv",
        config=tmp_path / "c.yaml", pixi=Path("/usr/local/bin/pixi"),
        manifest=tmp_path / "pixi.toml", teacher_rank=1, limit=60,
    )
    assert result == existing
    assert calls == [], "rebuilt a graft pool that already existed"


def test_a_graft_build_failure_costs_diversity_not_the_epoch(tmp_path, monkeypatch, capsys):
    """Grafts widen the pool. Losing them costs one epoch's diversity; stopping
    the producer costs the epoch itself."""
    import subprocess as sp

    from nanobody_opt import run_live_producer as producer

    epoch_root = tmp_path / "epoch1"
    epoch_root.mkdir()

    def explode(command):
        raise sp.CalledProcessError(1, command)

    monkeypatch.setattr(producer, "_run", explode)
    result = producer.refresh_donor_grafts(
        epoch_root=epoch_root, archive=tmp_path / "a.csv",
        config=tmp_path / "c.yaml", pixi=Path("/usr/local/bin/pixi"),
        manifest=tmp_path / "pixi.toml", teacher_rank=1, limit=60,
    )
    assert result is None
    assert "continuing with neighbourhood proposals alone" in capsys.readouterr().err


def test_grafts_are_merged_after_the_public_neighbourhood_filter():
    """Order is the whole fix.

    filter_public_neighborhood keeps proposals within
    --public-neighbourhood-distance substitutions of a published sequence -- 2
    in production -- and a graft is 16-18 mutations from its teacher by
    construction. Merging before the filter added sixty cases per epoch and
    stripped every one of them on the next line: epochs 25097 and 25098
    generated grafts and screened none.
    """
    import inspect

    from nanobody_opt import run_live_producer as producer

    source = inspect.getsource(producer)
    filtered = source.index("os.replace(filtered, candidates)")
    merged = source.index("merge_graft_candidates(candidates, grafts)")
    assert filtered < merged, "grafts merge before the filter that removes them"


def test_grafts_merge_even_when_the_neighbourhood_filter_is_disabled():
    """The merge must not be nested inside the filter's branch."""
    import inspect
    import textwrap

    from nanobody_opt import run_live_producer as producer

    source = inspect.getsource(producer)
    line = next(
        raw for raw in source.splitlines()
        if raw.strip() == "if grafts is not None:"
    )
    guard = next(
        raw for raw in source.splitlines()
        if "if args.public_neighbourhood_distance:" in raw
    )
    indent = len(line) - len(line.lstrip())
    guard_indent = len(guard) - len(guard.lstrip())
    assert indent <= guard_indent, (
        "the merge is nested inside the neighbourhood filter's branch"
    )
