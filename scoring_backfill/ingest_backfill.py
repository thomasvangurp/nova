#!/usr/bin/env python3
"""Compute local_validator_replay_rank_sum from 8x scorer results and upsert
into the mutation DB. Runs on the production box (single writer)."""
import sqlite3, json, math, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = {"design_iiptm":"max","design_ptm":"max","design_to_target_iptm":"max",
           "min_design_to_target_pae":"min","interaction_pae":"min",
           "plip_hbonds_refolded":"max","plip_saltbridge_refolded":"max",
           "delta_sasa_refolded":"max","liability_score":"min","liability_num_violations":"min"}
METRIC_ORDER = list(METRICS.keys())
TARGET = "P20809"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PASSES = 8

def load_fields(archive_path, recent=300, field_size=49, count=64, seed=0):
    df = pd.read_csv(archive_path)
    reference = df.tail(recent).reset_index(drop=True)
    if len(reference) < field_size:
        raise ValueError("archive smaller than field_size")
    rng = np.random.default_rng(seed)
    fields = []
    for _ in range(count):
        sel = reference.iloc[rng.choice(len(reference), field_size, replace=False)]
        fields.append({m: np.unique(pd.to_numeric(sel[m]).to_numpy(dtype=float)) for m in METRIC_ORDER})
    return fields

def replay_distribution(metrics, fields):
    scores = []
    for field in fields:
        total = 0.0
        for m in METRIC_ORDER:
            v = float(metrics[m])
            comp = field[m]
            if METRICS[m] == "min":
                total += 1 + int(np.count_nonzero(comp < v))
            else:
                total += 1 + int(np.count_nonzero(comp > v))
        scores.append(total)
    return scores

def upsert_observation(conn, obs):
    seq = obs["sequence"]; target = obs["target"]
    vector = bytes(AA_INDEX[aa] for aa in seq)
    conn.execute("INSERT INTO nodes(target, sequence, length, residue_vector) VALUES(?,?,?,?) "
                 "ON CONFLICT(target, sequence) DO UPDATE SET length=excluded.length, residue_vector=excluded.residue_vector",
                 (target, seq, len(seq), vector))
    node_id = conn.execute("SELECT id FROM nodes WHERE target=? AND sequence=?", (target, seq)).fetchone()[0]
    provenance = {k: v for k, v in obs.items() if k not in
                  {"observation_id","sequence","target","epoch","source","score_kind","score",
                   "observation_count","metrics","pose_stability","structure"}}
    conn.execute("""INSERT INTO observations(observation_id,node_id,epoch,source,score_kind,score,
                    observation_count,metrics_json,pose_json,provenance_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(observation_id) DO UPDATE SET node_id=excluded.node_id, epoch=excluded.epoch,
                    source=excluded.source, score_kind=excluded.score_kind, score=excluded.score,
                    observation_count=excluded.observation_count, metrics_json=excluded.metrics_json,
                    pose_json=excluded.pose_json, provenance_json=excluded.provenance_json""",
                 (obs["observation_id"], node_id, obs["epoch"], obs["source"], obs["score_kind"],
                  obs["score"], obs["observation_count"],
                  json.dumps(obs["metrics"], sort_keys=True), None,
                  json.dumps(provenance, sort_keys=True)))
    return node_id

def main():
    global PASSES
    results_dir = Path(sys.argv[1])
    archive_path = Path(sys.argv[2])
    db_path = Path(sys.argv[3])
    source = sys.argv[4] if len(sys.argv) > 4 else "backfill_8x"
    if len(sys.argv) > 5:
        PASSES = int(sys.argv[5])   # deepened banks (e.g. 20x) ingest at their own depth

    fields = load_fields(archive_path)
    print(f"fields={len(fields)}", flush=True)
    conn = sqlite3.connect(str(db_path), timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    # Merge every per-box + steal result file by hash (union of passes), so a
    # child whose 8 passes are split across its home box and a work-stealer is
    # still counted complete. The stale old-shard box*.json files are skipped.
    merged = {}
    for rp in sorted(results_dir.glob("*.json")):
        if rp.name.startswith("box"):
            continue
        try:
            results = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for h, entry in results.items():
            m = merged.setdefault(h, {"sequence": entry.get("sequence"),
                                      "epoch": entry.get("epoch"), "passes": {}})
            for i in range(1, PASSES + 1):
                if f"p{i}" in entry:
                    m["passes"][i] = entry[f"p{i}"]
            if entry.get("sequence") and not m["sequence"]:
                m["sequence"] = entry["sequence"]
            if entry.get("epoch") is not None and m["epoch"] is None:
                m["epoch"] = entry["epoch"]
    written = 0; skipped = 0
    for h, m in merged.items():
        if len(m["passes"]) < PASSES or not m["sequence"]:
            skipped += 1; continue
        seq = m["sequence"]; epoch = m["epoch"]
        passes = [m["passes"][i] for i in range(1, PASSES + 1)]
        # validate all 10 metrics present + finite
        ok = True
        for p in passes:
            if not isinstance(p, dict) or not set(METRIC_ORDER).issubset(p):
                ok = False; break
            for mm in METRIC_ORDER:
                if not math.isfinite(float(p[mm])):
                    ok = False; break
        if not ok:
            skipped += 1; continue
        rank_matrix = np.array([replay_distribution(p, fields) for p in passes])
        score_dist = rank_matrix.reshape(-1)
        score_mean = float(np.mean(score_dist))
        score_median = float(np.median(score_dist))
        score_p90 = float(np.quantile(score_dist, 0.9))
        aggregate = {mm: float(np.mean([p[mm] for p in passes])) for mm in METRIC_ORDER}
        obs = {"observation_id": f"{source}:epoch{epoch}:{seq}",
               "sequence": seq, "target": TARGET, "epoch": epoch,
               "source": source, "score_kind": "local_validator_replay_rank_sum",
               "score": score_mean, "observation_count": PASSES,
               "metrics": aggregate, "score_median": score_median, "score_p90": score_p90}
        upsert_observation(conn, obs)
        written += 1
        if written % 500 == 0:
            conn.commit()
    conn.commit()
    conn.close()
    print(f"WROTE={written} SKIPPED={skipped}", flush=True)

if __name__ == "__main__":
    main()
