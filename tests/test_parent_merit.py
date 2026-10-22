import json
import sqlite3

import pandas as pd

from sn68_source import import_shipping


def parent(seq, *, family="GAHPCPL", repair=False, old_score=1):
    return {"sequence": seq, "parent_cdr3": family, "validator_final_score": old_score,
            "parent_validator_duplicate": repair, "parent_top50_identity_in_basin": repair,
            "parent_cdr_developability_in_range": True, "parent_multipass_mean": old_score}


def evidence(scores):
    return {"scores": scores, "depths": {s: 20 for s in scores}, "minimum_passes": 10,
            "field_rows": 400, "before_epoch": 72}


def test_current_merit_beats_old_scores_family_and_coldkey_reservations(monkeypatch):
    selector = import_shipping("nanobody_opt_v2.select_elite_parents", monkeypatch)
    pool = [parent("OLD", family="UNRELATED", old_score=100),
            parent("RIVAL_A", old_score=730), parent("RIVAL_B", old_score=760),
            parent("UNMEASURED", family="OTHER", old_score=1)]
    selected = selector._merit_parents(pool, evidence({"OLD": 1400, "RIVAL_A": 742, "RIVAL_B": 771}), limit=2)
    assert [p["sequence"] for p in selected] == ["RIVAL_A", "RIVAL_B"]
    assert [p["selection_score"] for p in selected] == [742, 771]
    assert all(p["selection_score_kind"] == "current_multipass_mean" for p in selected)


def test_a_better_parent_from_another_owner_keeps_first_place(monkeypatch):
    selector = import_shipping("nanobody_opt_v2.select_elite_parents", monkeypatch)
    pool = [parent("RIVAL"), parent("BETTER")]
    selected = selector._merit_parents(pool, evidence({"RIVAL": 742, "BETTER": 700}), limit=2)
    assert [p["sequence"] for p in selected] == ["BETTER", "RIVAL"]


def test_proposal_cap_does_not_fill_from_one_parent_before_visiting_the_rest(monkeypatch):
    lane = import_shipping("nanobody_opt.competitor_lane", monkeypatch)
    items = [{"parent": p, "sequence": p+str(i)} for p in ["A", "B", "C"] for i in range(3)]
    ordered = lane._interleave_parent_edits(items, priority=["A", "B", "C"],
                                            scores={"A": 900, "B": 700, "C": 800})
    assert [x["sequence"] for x in ordered[:4]] == ["B0", "C0", "A0", "B1"]


def test_repair_parents_keep_gate_routing_and_capacity_cap(monkeypatch):
    selector = import_shipping("nanobody_opt_v2.select_elite_parents", monkeypatch)
    pool = [parent(str(i), repair=i < 4) for i in range(8)]
    selected = selector._merit_parents(pool, evidence({str(i): 700+i for i in range(8)}), limit=4)
    assert [p["sequence"] for p in selected] == ["0", "4", "5", "6"]
    assert selected[0]["parent_developability_repair"]
    assert selected[0]["parent_validator_duplicate"]


def test_parent_evidence_ranks_raw_folds_on_one_field_and_excludes_aggregate_rows(monkeypatch, tmp_path):
    selector = import_shipping("nanobody_opt_v2.select_elite_parents", monkeypatch)
    rank = import_shipping("nanobody_opt.rank_exact", monkeypatch)
    database = tmp_path / "db.sqlite"
    with sqlite3.connect(database) as c:
        c.executescript("""
            CREATE TABLE nodes (id INTEGER PRIMARY KEY, target TEXT, sequence TEXT);
            CREATE INDEX node_seq ON nodes(target,sequence);
            CREATE TABLE observations (node_id INTEGER, observation_id TEXT, source TEXT,
                score_kind TEXT, observation_count INTEGER, metrics_json TEXT);
            CREATE INDEX obs_node ON observations(node_id);
        """)
        for i, (seq, depth) in enumerate([("GOOD", 10), ("THIN", 9), ("OUTSIDE", 20)]):
            c.execute("INSERT INTO nodes VALUES (?, 'TEST', ?)", (i, seq))
            metrics = {m: 1.0 if direction == "min" else 9.0 for m, direction in rank.METRICS.items()}
            for j in range(depth):
                c.execute("INSERT INTO observations VALUES (?, ?, ?, ?, 1, ?)",
                          (i, f"{seq}-{j}", "public_multipass_rivals", "multipass_sum10_pass", json.dumps(metrics)))
            c.execute("INSERT INTO observations VALUES (?, ?, ?, ?, 20, ?)",
                      (i, seq+"aggregate", "live_exact_screen", "selection", json.dumps(metrics)))
    frame = pd.DataFrame({m: [4.0] for m in rank.METRICS})
    seen = []
    def fields(db, **kw):
        seen.append(kw)
        return [(e, frame) for e in range(62, 72)]
    monkeypatch.setattr(rank, "fetch_recent_fields_from_db", fields)
    result = selector._current_parent_evidence([parent("GOOD"), parent("THIN")],
                                              database=database, target="TEST", before_epoch=72)
    assert result["scores"] == {"GOOD": 10.0}
    assert result["depths"] == {"GOOD": 10, "THIN": 9}
    assert seen == [{"target": "TEST", "before_epoch": 72, "count": 10}]


def test_history_uses_mean_of_pass_ranks_not_rank_of_mean_metrics(monkeypatch, tmp_path):
    lane = import_shipping("nanobody_opt.competitor_lane", monkeypatch)
    rank = import_shipping("nanobody_opt.rank_exact", monkeypatch)
    database_module = import_shipping("nanobody_opt.db", monkeypatch)
    database = tmp_path / "history.sqlite"
    with sqlite3.connect(database) as c:
        c.executescript("""
            CREATE TABLE nodes (id INTEGER PRIMARY KEY, target TEXT, sequence TEXT);
            CREATE TABLE observations (node_id INTEGER, observation_id TEXT, source TEXT,
                epoch INTEGER, observation_count INTEGER, metrics_json TEXT);
        """)
        for i, (seq, values) in enumerate([("A", [0]*9+[100]), ("B", [2]*10), ("THIN", [0]*9)]):
            c.execute("INSERT INTO nodes VALUES (?, 'TEST', ?)", (i, seq))
            for j, value in enumerate(values):
                metrics = {m: value if direction == "min" else -value for m, direction in rank.METRICS.items()}
                c.execute("INSERT INTO observations VALUES (?, ?, 'local_exact_raw', 70, 1, ?)",
                          (i, f"{seq}-{j}", json.dumps(metrics)))
            c.execute("INSERT INTO observations VALUES (?, ?, 'local_exact_raw', 70, 20, ?)",
                      (i, seq+"aggregate", json.dumps({m: 999 for m in rank.METRICS})))
    frame = pd.DataFrame({m: list(range(10)) if direction == "min" else list(range(0, -10, -1))
                          for m, direction in rank.METRICS.items()})
    monkeypatch.setattr(rank, "fetch_recent_fields_from_db", lambda *a, **kw: [(e, frame) for e in range(62, 72)])
    monkeypatch.setattr(database_module, "connect_local", lambda p: sqlite3.connect(p))
    scores = lane.fetch_elite_history_parents(database, target="TEST", before_epoch=72, min_passes=10)
    assert list(scores) == ["A", "B"]
    assert scores == {"A": 20.0, "B": 30.0}


def test_generation_keeps_fresh_bank_scores_and_visits_each_parent(monkeypatch, tmp_path):
    lane = import_shipping("nanobody_opt.competitor_lane", monkeypatch)
    a, b = "A" * 127, "C" * 127
    monkeypatch.setattr(lane, "_load_pool", lambda _: [a, b])
    monkeypatch.setattr(lane, "multipass_scores", lambda *a_, **kw: {a: 10, b: 20})
    result = lane.generate(tmp_path / "archive.csv", [1], 2, priority=[a, b],
                           mutation_database=tmp_path / "history.sqlite", epoch=72,
                           priority_scores={a: 900, b: 700})
    assert [c["parent"] for c in result["candidates"]] == [b, a]
    assert [c["parent_multipass_mean"] for c in result["candidates"]] == [700, 900]


def test_selector_cli_uses_current_merit_instead_of_the_legacy_selection_path(monkeypatch, tmp_path):
    selector = import_shipping("nanobody_opt_v2.select_elite_parents", monkeypatch)
    old, best = "A" * 127, "C" * 127
    payload = {"success": True, "data": [{"coldkey": "OLD_OWNER", "nanobodies": [
        {"sequence": old, "protein_name": "TEST", "final_score": 1,
         **{metric: 1 for metric in selector.METRICS}},
        {"sequence": best, "protein_name": "TEST", "final_score": 999,
         **{metric: 1 for metric in selector.METRICS}},
    ]}]}
    reservoir, output = tmp_path / "history.json", tmp_path / "parents.json"
    reservoir.write_text('{"candidates": []}')
    parser = selector._parser()
    args = parser.parse_args(['--target', 'TEST', '--before-epoch', '72', '--scan-limit', '1',
        '--output', str(output), '--historical-reservoir', str(reservoir),
        '--observation-database', str(tmp_path / 'observations.sqlite')])
    monkeypatch.setattr(parser, 'parse_args', lambda: args)
    monkeypatch.setattr(selector, '_parser', lambda: parser)
    monkeypatch.setattr(selector, '_fetch', lambda root, epoch: (epoch, payload))
    monkeypatch.setattr(selector, 'update_observation_store', lambda *a, **kw: None)
    measured = {**evidence({old: 1400, best: 700}), 'field_epochs': list(range(62, 72))}
    monkeypatch.setattr(selector, '_current_parent_evidence', lambda *a, **kw: measured)
    monkeypatch.setattr(selector, '_validator_parent_pool', lambda pool, **kw: [
        {**p, 'parent_validator_duplicate': False, 'parent_top50_identity_in_basin': False} for p in pool])
    monkeypatch.setattr(selector, '_developability_parent_pool', lambda pool, **kw: [
        {**p, 'parent_cdr_developability_in_range': True} for p in pool])
    selector.main()
    bank = json.loads(output.read_text())
    assert bank['sequences'] == [best, old]
    assert bank['parent_ranking_policy'] == 'current_field_multipass_merit'
    audit = json.loads(output.with_suffix('.ranking.json').read_text())
    assert audit['field_epochs'] == list(range(62, 72))
    assert [p['mean'] for p in audit['parents']] == [700, 1400]
