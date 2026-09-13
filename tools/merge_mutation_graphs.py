#!/usr/bin/env python3
"""Content-identity merge of one nanobody mutation graph SQLite into another.

Both files use the same schema (schema_version 2).  Node ``id`` values are NOT
comparable across files -- they are per-database autoincrement surrogates, and
in practice every id in a rebuilt database collides with an unrelated molecule
in the restored one.  The only meaningful identities are:

    nodes         (target, sequence)          -- UNIQUE in the schema
    observations  observation_id, which is content-derived
                  ("<source>:epoch<N>:<SEQUENCE>"), cross-checked against the
                  content key (target, sequence, source, score_kind, epoch,
                  score)

The merge therefore remaps every source node id onto a destination id and
inserts only rows that are new by content.  It never uses INSERT OR REPLACE and
never overwrites a destination row.  Any id collision that is not a verified
content-identical duplicate aborts the whole transaction.

Usage:
    merge_mutation_graphs.py --dest COPY.sqlite --src SRC.sqlite [--apply]

Without --apply the merge runs inside a transaction that is rolled back, so it
doubles as a dry run.  Always run against a copy, verify, then swap.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

# Tables that carry graph structure.  The rebuilt database has none of these;
# if a future source does, refuse rather than guess at id remapping semantics.
STRUCTURE_TABLES = ("edges", "lineage_edges", "mutation_effects", "proposal_scores")
COUNTED_TABLES = ("nodes", "observations", "edges", "lineage_edges",
                  "mutation_effects", "proposal_scores", "metadata")


def counts(con: sqlite3.Connection, schema: str) -> dict[str, int]:
    return {t: con.execute('select count(*) from %s."%s"' % (schema, t)).fetchone()[0]
            for t in COUNTED_TABLES}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="destination copy, written in place")
    ap.add_argument("--src", required=True, help="source database, opened read-only")
    ap.add_argument("--apply", action="store_true", help="commit (default: roll back)")
    args = ap.parse_args()

    con = sqlite3.connect(args.dest)
    con.execute("pragma foreign_keys=ON")
    con.execute("attach database ? as src", ("file:%s?mode=ro" % args.src,))

    before = counts(con, "main")
    print("before:", before)
    print("source:", counts(con, "src"))

    # Schema equality is a precondition; a differing schema means the identity
    # rules below may not hold.
    def schema_of(s: str) -> list[tuple[str, str]]:
        return [(n, " ".join(q.split()))
                for n, q in con.execute(
                    "select name,sql from %s.sqlite_master "
                    "where type='table' order by name" % s)]

    if schema_of("main") != schema_of("src"):
        print("ABORT: table schemas differ between destination and source")
        return 2
    print("schema check: identical")

    for t in STRUCTURE_TABLES:
        n = con.execute('select count(*) from src."%s"' % t).fetchone()[0]
        if n:
            print("ABORT: source has %d rows in %s; this tool only merges "
                  "nodes+observations and cannot remap structural ids" % (n, t))
            return 2
    print("structure check: source has no edges/lineage/effects/proposal_scores")

    con.execute("begin immediate")

    # ---- nodes: map by (target, sequence), allocate fresh ids for new ones ----
    dest_by_seq = {(t, s): (i, rv, ln) for i, t, s, ln, rv in con.execute(
        "select id,target,sequence,length,residue_vector from main.nodes")}
    next_id = (con.execute("select coalesce(max(id),0) from main.nodes").fetchone()[0]) + 1

    id_map: dict[int, int] = {}
    new_nodes = []
    reused = 0
    for sid, target, seq, length, rv in con.execute(
            "select id,target,sequence,length,residue_vector from src.nodes"):
        hit = dest_by_seq.get((target, seq))
        if hit is not None:
            did, drv, dlen = hit
            if drv != rv or dlen != length:
                con.rollback()
                print("ABORT: node (%s, %s...) matches destination id %d by "
                      "sequence but residue_vector/length differ -- encodings "
                      "are not compatible" % (target, seq[:24], did))
                return 2
            id_map[sid] = did
            reused += 1
        else:
            id_map[sid] = next_id
            new_nodes.append((next_id, target, seq, length, rv))
            next_id += 1

    con.executemany(
        "insert into main.nodes(id,target,sequence,length,residue_vector) "
        "values(?,?,?,?,?)", new_nodes)
    print("nodes: %d matched existing by (target,sequence), %d inserted with "
          "remapped ids" % (reused, len(new_nodes)))

    # ---- observations: dedup on observation_id, cross-checked by content ----
    dest_obs = {r[0]: r[1:] for r in con.execute(
        "select observation_id,node_id,epoch,source,score_kind,score,"
        "observation_count,metrics_json,pose_json from main.observations")}
    dest_content = set(con.execute(
        "select n.target,n.sequence,o.source,o.score_kind,o.epoch,o.score "
        "from main.observations o join main.nodes n on n.id=o.node_id"))

    to_insert, dup_id, dup_content, conflicts = [], 0, 0, []
    for row in con.execute(
            "select o.observation_id,o.node_id,o.epoch,o.source,o.score_kind,"
            "o.score,o.observation_count,o.metrics_json,o.pose_json,"
            "o.provenance_json,n.target,n.sequence "
            "from src.observations o join src.nodes n on n.id=o.node_id"):
        (oid, snode, epoch, source, kind, score, ocount, metrics, pose,
         prov, target, seq) = row
        mapped = id_map[snode]

        existing = dest_obs.get(oid)
        if existing is not None:
            (dnode, depoch, dsource, dkind, dscore, docount, dmetrics,
             dpose) = existing
            same = (dnode == mapped and depoch == epoch and dsource == source
                    and dkind == kind and dscore == score
                    and docount == ocount and dmetrics == metrics
                    and dpose == pose)
            if same:
                dup_id += 1
                continue
            # Same content-derived id, different payload: a real conflict.
            # Keep the destination row untouched and refuse the merge.
            conflicts.append((oid, existing, (mapped, epoch, source, kind,
                                              score, ocount)))
            continue

        if (target, seq, source, kind, epoch, score) in dest_content:
            # Same fact already recorded under a different observation_id
            # (the id prefix changed between producer versions).
            dup_content += 1
            continue

        to_insert.append((oid, mapped, epoch, source, kind, score, ocount,
                          metrics, pose, prov))

    if conflicts:
        con.rollback()
        print("ABORT: %d observation_id collisions with differing payloads; "
              "destination left untouched" % len(conflicts))
        for c in conflicts[:5]:
            print("   ", c)
        return 2

    con.executemany(
        "insert into main.observations(observation_id,node_id,epoch,source,"
        "score_kind,score,observation_count,metrics_json,pose_json,"
        "provenance_json) values(?,?,?,?,?,?,?,?,?,?)", to_insert)
    print("observations: %d inserted, %d skipped as identical observation_id, "
          "%d skipped as duplicate content under another id"
          % (len(to_insert), dup_id, dup_content))

    # ---- invariants ----
    fk = con.execute("pragma foreign_key_check").fetchall()
    if fk:
        con.rollback()
        print("ABORT: foreign_key_check reported %d violations" % len(fk))
        return 2

    orphan = con.execute(
        "select count(*) from main.observations o "
        "left join main.nodes n on n.id=o.node_id where n.id is null").fetchone()[0]
    if orphan:
        con.rollback()
        print("ABORT: %d observations reference a missing node" % orphan)
        return 2

    after = counts(con, "main")
    for t, v in before.items():
        if after[t] < v:
            con.rollback()
            print("ABORT: %s shrank %d -> %d" % (t, v, after[t]))
            return 2

    if args.apply:
        con.commit()
        print("COMMITTED")
    else:
        con.rollback()
        print("ROLLED BACK (dry run; pass --apply to commit)")
    print("after: ", after)
    print("delta: ", {t: after[t] - before[t] for t in before})
    return 0


if __name__ == "__main__":
    sys.exit(main())
