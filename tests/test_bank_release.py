"""Releasing a bank pool that the fleet allocator can actually bind.

Splitting molecules across uids is not this program's job. The validator scores
a SMILES once and fans the value out to every uid that submitted it -- the Boltz
wrapper keys on ``{smiles: [(uid, mol_id), ...]}`` and then "distributes results
to all UIDs" -- so uid portfolios have to differ to be worth anything, and
compound_miner.fleet already builds them: a shared core of the best 12, then a
distinct tail of 8 and 4 spares per uid. What that allocator cannot do is invent
rows. A release carrying 60 molecules is not a smaller fleet, it is
``insufficient_round_fleet_pool`` and no compound submission at all.

rdkit is absent on a controller, so the entropy gate is stubbed; it is the same
function the compound branch ships and is exercised on the box.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _release_module():
    spec = importlib.util.spec_from_file_location(
        "_bank_release", ROOT / "tools" / "bank_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def release(monkeypatch):
    module = _release_module()
    monkeypatch.setattr(module, "atompair_entropy", lambda smiles: 0.31)
    return module


def _bank(path: Path, rows: int = 400, reaction: int = 3) -> Path:
    """A bank shaped like the real one, down to the columns.

    The column list is the live bank's, because the release has to carry
    `reaction`, `components` and both affinity terms: the fleet allocator
    rebuilds a Candidate from every row and cross-checks the identity fields
    against the identifier. validate_release_payload does not look at them, so
    a release missing them passed every check here and was refused by the
    publisher.

    The SMILES are real amino-alcohols rather than placeholders because the
    allocator fingerprints every candidate and refuses the whole fleet on one
    unparseable row.
    """
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE candidates(name TEXT PRIMARY KEY, smiles TEXT, "
        "reaction INT, components TEXT, inchikey TEXT, heavy_atoms INT, "
        "rotatable_bonds INT, affinity_probability_binary REAL, "
        "affinity_pred_value REAL, score REAL, scorer TEXT)")
    connection.execute(
        "CREATE TABLE rescored(name TEXT PRIMARY KEY, passes INT, "
        "mean_score REAL, sd_score REAL, seeds TEXT, single_pass_score REAL)")
    for index in range(rows):
        components = [56473 + index, 38787 + index, 87418 + index]
        name = "rxn:%d:%d:%d:%d" % (reaction, *components)
        connection.execute(
            "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                name, "C" * (index // 20 + 1) + "N" + "C" * (index % 20 + 1) + "O",
                reaction,
                json.dumps(components), f"KEY{index:08d}-AAA-B", 24, 4,
                0.30 + index * 1e-5, -3.11 - index * 1e-4, 0.09, "nim19"))
        connection.execute("INSERT INTO rescored VALUES(?,?,?,?,?,?)", (
            name, 3, 2.0 - index * 0.001, 0.004, "[68,69,70]", 0.09))
    connection.commit()
    connection.close()
    return path


def _run(module, arguments, monkeypatch):
    # These tests build a synthetic bank and have no copy of the 102k-row
    # Metanova/Submission-Archive, so they opt out of the historical gate
    # explicitly. Production must NOT: bank_release refuses to run without it,
    # because the validator applies it whether we checked or not.
    if "--archive" not in arguments and "--no-archive" not in arguments:
        arguments = arguments + ["--no-archive"]
    monkeypatch.setattr(sys, "argv", ["bank_release.py"] + arguments)
    return module.main()


def _release_for(module, tmp_path, monkeypatch, *extra):
    bank = _bank(tmp_path / "bank.sqlite")
    output = str(tmp_path / "result_rxn{reaction}.json")
    code = _run(module, ["--bank", str(bank), "--epoch", "25300",
                         "--output", output, *extra], monkeypatch)
    return code, Path(output.format(reaction=3))


def test_the_sizing_rule_is_the_one_the_allocator_enforces(release):
    """A mirrored constant that drifts is a release the publisher will refuse.

    required_exact_pool_size lives in the miner tree, which the bank box does
    not have, so it is mirrored here and pinned against the original whenever
    both are checked out.
    """
    contract = ROOT.parent / "nova-compound-fleet" / "compound_miner" / "fleet_contract.py"
    if not contract.is_file():
        pytest.skip("the compound fleet tree is not checked out beside this one")
    namespace: dict = {}
    exec(compile(contract.read_text(), str(contract), "exec"), namespace)
    for fleet in (2, 5, 10, 12):
        assert release.required_pool_size(fleet, 20, 40) == \
            namespace["required_exact_pool_size"](fleet)


def test_a_ten_uid_release_carries_the_whole_allocation_pool(release, tmp_path,
                                                             monkeypatch):
    code, path = _release_for(release, tmp_path, monkeypatch, "--fleet", "10")
    assert code == 0
    payload = json.loads(path.read_text())
    assert len(payload["molecules"]) == 132
    assert payload["release"]["exact_count"] == 132
    assert payload["release"]["selected_count"] == 20
    assert payload["release"]["fleet_size"] == 10


def test_one_uid_still_gets_the_sixty_row_release_it_always_did(release, tmp_path,
                                                                monkeypatch):
    """The single-uid path predates the fleet and must not have moved."""
    code, path = _release_for(release, tmp_path, monkeypatch)
    assert code == 0
    payload = json.loads(path.read_text())
    assert len(payload["molecules"]) == 60
    assert payload["release"]["epoch"] == 25300


def test_the_pool_is_ordered_best_first(release, tmp_path, monkeypatch):
    """The allocator takes the shared core off the top, so the order is the
    difference between every uid holding our best 12 and none of them doing."""
    _, path = _release_for(release, tmp_path, monkeypatch, "--fleet", "10")
    scores = [m["score"] for m in json.loads(path.read_text())["molecules"]]
    assert scores == sorted(scores, reverse=True)


def test_the_declared_selection_score_matches_its_own_molecules(release, tmp_path,
                                                                monkeypatch):
    """validate_release_payload recomputes this to 1e-12 and refuses on drift."""
    _, path = _release_for(release, tmp_path, monkeypatch, "--fleet", "10")
    payload = json.loads(path.read_text())
    declared = payload["release"]["selection_score"]
    observed = sum(m["score"] for m in payload["molecules"][:20])
    assert abs(declared - observed) < 1e-12


def test_a_bank_too_thin_for_the_fleet_is_refused(release, tmp_path, monkeypatch):
    """Emitting a short pool produces a file the publisher silently declines to
    allocate from, which reads on the box as a healthy release and no commit."""
    bank = _bank(tmp_path / "bank.sqlite", rows=100)
    assert _run(release, ["--bank", str(bank), "--epoch", "25300", "--fleet", "10",
                          "--output", str(tmp_path / "r{reaction}.json")],
                monkeypatch) == 1


def test_a_release_below_the_entropy_gate_is_refused(tmp_path, monkeypatch):
    """A skipped uid is indistinguishable from a crashed one on the chain."""
    module = _release_module()
    monkeypatch.setattr(module, "atompair_entropy", lambda smiles: 0.2462)
    code, _ = _release_for(module, tmp_path, monkeypatch, "--fleet", "10")
    assert code == 1


def test_only_rescored_molecules_are_eligible(release, tmp_path, monkeypatch):
    """Single-pass scores carry the winner's curse; selecting on them costs
    0.0817 against the three-pass mean."""
    bank = tmp_path / "bank.sqlite"
    _bank(bank)
    connection = sqlite3.connect(bank)
    connection.execute("DROP TABLE rescored")
    connection.commit()
    connection.close()
    assert _run(release, ["--bank", str(bank), "--epoch", "25300",
                          "--output", str(tmp_path / "r{reaction}.json")],
                monkeypatch) == 1


REAL_BANK = ROOT / "data" / "bank_master.sqlite"


def test_the_fixture_schema_is_the_real_bank_schema(tmp_path):
    """Every column the release reads has to exist in the bank it reads from.

    The fixture is hand-written, so it can drift into agreeing with a producer
    the real bank would not support.
    """
    if not REAL_BANK.is_file():
        pytest.skip("no local copy of the bank")
    real = sqlite3.connect(f"file:{REAL_BANK}?mode=ro", uri=True)
    expected = [row[1] for row in real.execute("PRAGMA table_info(candidates)")]
    real.close()
    fixture = sqlite3.connect(_bank(tmp_path / "bank.sqlite", rows=1))
    observed = [row[1] for row in fixture.execute("PRAGMA table_info(candidates)")]
    fixture.close()
    assert observed == expected


FLEET_TREE = ROOT.parent / "nova-compound-fleet"


def _fleet_allocator():
    """Import the publisher's allocator out of the compound fleet worktree.

    Both trees ship a package called ``compound_miner`` and only one of them has
    ``fleet``, so a plain import picks up whichever the rest of the suite
    imported first: these tests passed alone and failed in the full run. The
    name is evicted for the duration of the import and put back afterwards.
    """
    if not (FLEET_TREE / "compound_miner" / "fleet.py").is_file():
        pytest.skip("the compound fleet tree is not checked out beside this one")
    pytest.importorskip("rdkit")
    saved = {name: module for name, module in sys.modules.items()
             if name == "compound_miner" or name.startswith("compound_miner.")}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(FLEET_TREE))
    try:
        from compound_miner.fleet import (  # noqa: PLC0415
            FleetMember, allocate_compound_fleet_from_release,
        )
        return FleetMember, allocate_compound_fleet_from_release
    finally:
        sys.path.remove(str(FLEET_TREE))
        for name in [n for n in sys.modules
                     if n == "compound_miner" or n.startswith("compound_miner.")]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_the_live_allocator_binds_the_release_to_ten_uids(release, tmp_path,
                                                          monkeypatch):
    """The check that caught the missing identity fields.

    Running the publisher's own allocator is the point: our own mirror of its
    expectations agreed with itself while the real one raised KeyError on the
    first row.
    """
    FleetMember, allocate_compound_fleet_from_release = _fleet_allocator()
    code, path = _release_for(release, tmp_path, monkeypatch, "--fleet", "10")
    assert code == 0
    crew = [FleetMember(uid=10 + i, hotkey="5H" + str(i) * 40) for i in range(10)]
    # The synthetic bank's molecules are too alike to clear the real entropy
    # gate; the gate itself is covered separately and on the box.
    artifact = allocate_compound_fleet_from_release(
        json.loads(path.read_text()), crew, minimum_entropy=0.0)
    portfolios = {s["uid"]: [m["name"] for m in s["molecules"]]
                  for s in artifact["submissions"]}
    assert len(portfolios) == 10
    assert all(len(names) == 20 for names in portfolios.values())
    assert len({tuple(sorted(names)) for names in portfolios.values()}) == 10
    # The shared core is 12, so any two uids overlap in exactly 12 and differ
    # in the other 8. That is what makes k uids worth more than one.
    overlaps = {len(set(portfolios[a]) & set(portfolios[b]))
                for a in portfolios for b in portfolios if a < b}
    assert overlaps == {12}


def test_the_top_uid_receives_the_releases_own_selection(release, tmp_path,
                                                         monkeypatch):
    """The fleet must not cost us the best portfolio it was built from."""
    FleetMember, allocate_compound_fleet_from_release = _fleet_allocator()
    code, path = _release_for(release, tmp_path, monkeypatch, "--fleet", "10")
    assert code == 0
    payload = json.loads(path.read_text())
    artifact = allocate_compound_fleet_from_release(
        payload, [FleetMember(uid=10 + i, hotkey="5H" + str(i) * 40)
                  for i in range(10)], minimum_entropy=0.0)
    best = max(s["selection_score"] for s in artifact["submissions"])
    assert abs(best - payload["release"]["selection_score"]) < 1e-9


def test_the_historical_gate_matches_the_validator_constants():
    # Read from neurons/validator/molecule_validity.py and config/config.yaml,
    # not inferred: Morgan radius=2 fpSize=2048, refuse at >= 0.5. Our own
    # repo's validity.py carries 0.7, which is why this is pinned here.
    import importlib.util, sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("br_const", root / "tools/bank_release.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["br_const"] = mod
    spec.loader.exec_module(mod)
    assert mod.MAX_SIMILARITY_TO_HISTORICAL == 0.5
    assert mod.HISTORY_RADIUS == 2 and mod.HISTORY_BITS == 2048


def test_one_flagged_molecule_is_enough_to_refuse_it():
    # molecule_validity breaks on the FIRST failure and then skips the whole
    # submission, so the filter must drop a molecule on its single worst
    # neighbour, never on an average.
    import importlib.util, sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("br_gate", root / "tools/bank_release.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["br_gate"] = mod
    spec.loader.exec_module(mod)
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        import pytest; pytest.skip("rdkit not installed here")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    history = [gen.GetFingerprint(Chem.MolFromSmiles(aspirin))]
    flagged = mod.too_close_to_history([aspirin, "CCCCCCCCCCO"], history)
    assert 0 in flagged and flagged[0] == 1.0
    assert 1 not in flagged


def test_an_unparseable_molecule_is_refused_not_admitted():
    import importlib.util, sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("br_bad", root / "tools/bank_release.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["br_bad"] = mod
    spec.loader.exec_module(mod)
    try:
        import rdkit  # noqa: F401
    except ImportError:
        import pytest; pytest.skip("rdkit not installed here")
    assert mod.too_close_to_history(["not-a-smiles"], []) == {0: 1.0}


def _gate_module():
    import importlib.util, sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("br_refresh", root / "tools/bank_release.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["br_refresh"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_failed_refresh_keeps_the_cached_archive(tmp_path):
    # A network blip must cost a slightly stale filter, never the epoch.
    mod = _gate_module()
    archive = tmp_path / "arch.csv"
    archive.write_text("SMILES\nCCO\n")
    status = mod.refresh_archive(archive, url="http://127.0.0.1:1/nope.csv", timeout=2)
    assert status["refreshed"] is False
    assert status["using_cached"] is True
    assert archive.read_text() == "SMILES\nCCO\n"      # untouched
    assert not (tmp_path / "arch.csv.download").exists()


def test_a_truncated_download_does_not_replace_a_good_archive(tmp_path, monkeypatch):
    # An error page or half a file must not overwrite the archive: that would
    # DISABLE the gate rather than fail, which is the worst outcome.
    mod = _gate_module()
    archive = tmp_path / "arch.csv"
    archive.write_text("SMILES\n" + "\n".join(["CCO"] * 5000) + "\n")
    before = archive.read_text()

    class FakeResponse:
        def read(self, *a): return b""
        def __enter__(self): return io.BytesIO(b"SMILES\nCCO\n")
        def __exit__(self, *a): return False

    import io, urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    status = mod.refresh_archive(archive, url="http://example/x.csv")
    assert status["refreshed"] is False
    assert "truncated" in status["why"]
    assert archive.read_text() == before


def test_every_validator_gate_runs_at_selection_not_only_preflight():
    """Selection must refuse what the validator refuses, for ALL its rules.

    A pre-flight that fails closed protects us from shipping junk but costs the
    whole epoch. Measured on the live bank, 1 of 394 rescored molecules fails
    BRENK -- one unlucky draw from an empty epoch. Selection now applies heavy
    atoms, BRENK, banned atoms, rotatable bonds, Boltz safety and similarity.
    """
    mod = _gate_module()
    body = (Path(__file__).resolve().parent.parent / "tools/bank_release.py").read_text()
    start = body.index("def refused_by_validator")
    fragment = body[start:body.index("def too_close_to_history")]
    for gate in ("min_heavy_atoms", "HasMatch", "banned_atom_types",
                 "NumRotatableBonds", "boltz", "MAX_SIMILARITY_TO_HISTORICAL"):
        assert gate in fragment, f"selection does not apply {gate}"


def test_selection_degrades_to_similarity_when_the_validator_tree_is_absent():
    # The extra gates need the cloned validator. Without it we still apply the
    # similarity filter and say so, rather than refusing to release at all --
    # the pre-flight remains the backstop.
    mod = _gate_module()
    assert mod.validator_helpers(Path("/nonexistent/validator")) is None
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        import pytest; pytest.skip("rdkit not installed here")
    assert mod.refused_by_validator(["CCO"], [], None) == {}


def test_the_rescore_filters_before_spending_the_gpu():
    """63% of the last rescore was spent on molecules that cannot ship."""
    body = (Path(__file__).resolve().parent.parent / "tools/bank_rescore.py").read_text()
    assert "refused_by_validator" in body
    assert "overfetch" in body
    # The gate must come BEFORE the shortlist is cut, or it saves nothing.
    assert body.index("refused_by_validator") < body.index("rows = candidates[:arguments.shortlist]")


def test_the_builder_refuses_invalid_molecules_before_scoring_them():
    """Never PROPOSE what the validator will refuse.

    23% of the 58,754-molecule bank can never be submitted, and it is worst
    exactly where we select: 35% of the top 500 survives against 83% of the
    tail, because high scores come from scaffolds everyone already submitted.
    Each refused molecule is GPU time spent on something unshippable.
    """
    body = (Path(__file__).resolve().parent.parent / "tools/bank_builder.py").read_text()
    assert "load_validator_gate" in body
    assert "refused_by_validator" in body
    # The gate must precede scoring, or it saves no compute at all.
    assert body.index("refused_by_validator([c.smiles for c in batch]") < \
           body.index("scored_records = score_with_nim(batch)")


def test_the_builder_still_runs_without_the_archive():
    # A builder box without the CSV must degrade, not refuse: release and
    # pre-flight remain the backstops.
    body = (Path(__file__).resolve().parent.parent / "tools/bank_builder.py").read_text()
    assert "validator_gate_absent" in body
    assert "if validator_gate is not None:" in body
