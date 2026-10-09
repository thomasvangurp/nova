#!/usr/bin/env python3
"""Per-epoch refresh: extend the ``hf_field_20x_rank_sum`` pool with NEW sequences.

The parent pool is the flat top-50 by 20x HF-field rank-sum across three
sources.  This script grows that pool over time.  Each run it recomputes the
three source sets, finds the sequences that are NOT yet scored at 20x
(``hf_field_20x_rank_sum``), 20x-scores just those, and imports them.

Sources (all 20x, one shared 21k-58k rank-sum scale, lower = better):

  our_top50_20x           our top-50 by the mixed replay rank-sum read
  public_multipass_rivals every rival coldkey's submissions this epoch
                          (leaderboard API, coldkey != ours)
  hf_archive_top200       rank_binders top-200 over the raw HF archive

Idempotent and cheap to re-run: scoring only covers sequences missing from the
pool, so overlapping runs / restarts are safe.  Run it during the quiet moment
between epochs -- the loop wrapper refuses to start while the one-shot backfill
jobs (backfill_topup / score_top50_32) are still holding the GPUs.

Run on the slave (bgen env, over the tailscale MySQL relay)::

    SN68_MYSQL_HOST=127.0.0.1 SN68_MYSQL_PORT=3306 \
      /root/.pixi/envs/bgen/bin/python /root/hf_field_refresh.py --gpus 0,1,2,3,4,5,6,7
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

TARGET = "P20809"
SCORE_KIND = "hf_field_20x_rank_sum"
N_PASSES = 20
OUR_COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"
API = "https://compound-api.metanova-labs.ai"

ARCHIVE = "/root/P20809_nanobodies.csv"
CONFIG = "/root/config/config.yaml"
SCORER = "/root/nanobody_opt/score_boltzgen_fast_adapter.py"
OUT_ROOT = Path("/root/hf_field_refresh")

METRICS = ["design_iiptm", "design_ptm", "design_to_target_iptm",
           "min_design_to_target_pae", "interaction_pae",
           "plip_hbonds_refolded", "plip_saltbridge_refolded",
           "delta_sasa_refolded", "liability_score"]
WEIGHTS = {"design_iiptm": 1.0, "design_ptm": 2.0, "min_design_to_target_pae": 1.0,
           "plip_hbonds_refolded": 2.0, "plip_saltbridge_refolded": 2.0,
           "delta_sasa_refolded": 2.0}
DIRS = {"design_iiptm": "max", "design_ptm": "max", "min_design_to_target_pae": "min",
        "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
        "delta_sasa_refolded": "max"}

# Validator's actual score is the SUM of the ten metric dense ranks
# (rho=1.000 vs final_score), not the Algorithm-2 minmax (~0.56).
SUM_OF_TEN_DIRS = {
    "design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min", "interaction_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max", "liability_score": "min",
    "liability_num_violations": "min",
}


def sum_of_ten(field: pd.DataFrame) -> pd.Series:
    """Sum of the ten metric dense ranks (lower = better)."""
    total = pd.Series(0.0, index=field.index)
    for metric, direction in SUM_OF_TEN_DIRS.items():
        if metric not in field.columns:
            continue
        total += field[metric].rank(method="dense", ascending=(direction == "min"))
    return total

# Our top-50: the exact mixed replay read `_live_top50_from_db` used (no
# fillers), so "our own new top-50" means the same stable quality read.
OUR_TOP50_SQL = """
    SELECT n.sequence, AVG(o.score) AS mean_score
    FROM observations o JOIN nodes n ON o.node_id = n.id
    WHERE n.target = %s
      AND o.score_kind IN ('local_validator_replay_rank_sum',
                           'local_exact_validator_replay_rank_sum')
    GROUP BY n.sequence
    HAVING COUNT(*) >= 8
    ORDER BY mean_score ASC
    LIMIT 50
"""


def _rank_binders():
    import importlib.util
    src = Path("/root/utils/minmax_weighted_rank.py")
    spec = importlib.util.spec_from_file_location("_rb", src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.rank_binders


def algo2_score(field: pd.DataFrame) -> pd.Series:
    work = field[field["liability_num_violations"] <= 50].copy()
    rc = {}
    for metric, weight in WEIGHTS.items():
        if metric not in work.columns:
            continue
        rc[metric] = work[metric].rank(ascending=(DIRS[metric] == "min"),
                                       method="average") / weight
    return pd.DataFrame(rc, index=work.index).max(axis=1)


def latest_completed_epoch(start: int, lookback: int = 60) -> int | None:
    """Walk backward from ``start`` until the leaderboard API returns 200."""
    for epoch in range(start, start - lookback, -1):
        try:
            req = urllib.request.Request(
                f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
                headers={"User-Agent": "nova-hf-field-refresh/1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            return epoch
        except Exception:
            continue
    return None


def fetch_rivals(epoch: int) -> set[str]:
    try:
        req = urllib.request.Request(
            f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
            headers={"User-Agent": "nova-hf-field-refresh/1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except Exception:
        return set()
    out: set[str] = set()
    for submission in payload.get("data") or []:
        if str(submission.get("coldkey") or "").strip() == OUR_COLDKEY:
            continue
        for nb in submission.get("nanobodies") or []:
            if nb.get("protein_name") != TARGET:
                continue
            seq = str(nb.get("sequence") or "").strip().upper()
            if seq:
                out.add(seq)
    return out


def fetch_hf_top200(archive: pd.DataFrame) -> set[str]:
    rb = _rank_binders()
    top = rb(archive, k=200, max_liability_violations=50)
    return {str(s).strip().upper() for s in top["sequence"].tolist()}


def fetch_db_rivals(conn) -> set[str]:
    """Every rival sequence we have ever scored (cumulative, all epochs)."""
    rows = conn.execute(
        """
        SELECT DISTINCT n.sequence
        FROM observations o JOIN nodes n ON o.node_id = n.id
        WHERE o.source IN ('public_multipass_rivals', 'slave_public_multipass_rivals')
        """,
    ).fetchall()
    return {str(r[0]).strip().upper() for r in rows}


def fetch_our_top50(conn) -> set[str]:
    rows = conn.execute(OUR_TOP50_SQL, (TARGET,)).fetchall()
    return {str(r[0]).strip().upper() for r in rows}


def already_scored(conn) -> set[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT n.sequence
        FROM observations o JOIN nodes n ON o.node_id = n.id
        WHERE o.score_kind = %s
        """,
        (SCORE_KIND,),
    ).fetchall()
    return {str(r[0]).strip().upper() for r in rows}


def score_pass(sequences: list[str], run: int, archive: pd.DataFrame,
               out_root: Path, gpus: str) -> dict[str, float]:
    inp = out_root / f"input_{run}.json"
    out_dir = out_root / f"score{run}"
    inp.write_text(json.dumps({"sequences": [{"sequence": s} for s in sequences]}))
    subprocess.run(
        [sys.executable, "-m", "nanobody_opt.score_boltzgen_waves",
         "--sequences", str(inp), "--config", CONFIG, "--target", TARGET,
         "--output-dir", str(out_dir), "--gpus", gpus,
         "--wave-size", str(len(sequences)), "--predict-batch-size", "2",
         "--limit", str(len(sequences)), "--scorer", SCORER,
         "--capture-structures"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    payload = json.loads((out_dir / "exact_all.json").read_text())
    wave = payload["per_nanobody_components"]["0"]
    rows = {}
    for seq, comp in wave.items():
        m = comp.get(TARGET, {})
        rows[seq] = {k: float(v) for k, v in m.items()
                     if k in METRICS + ["liability_num_violations"]
                     and isinstance(v, (int, float))}
    rival = pd.DataFrame(rows).T
    arch_s = archive.set_index("sequence")[METRICS + ["liability_num_violations"]]
    field = pd.concat([arch_s, rival])
    scores = sum_of_ten(field)  # sum-of-10 (lower = better)
    out = {}
    for s in rival.index:
        mask = scores.index == s
        if mask.any():
            out[s] = float(scores[mask].iloc[-1])
    return out


def _retry_db(fn, *args, retries=6, delay=12, **kwargs):
    """Retry on the transient tailscale-relay 'Lost connection' errors."""
    import time as _time
    from MySQLdb import OperationalError

    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except OperationalError as exc:
            print(f"  DB retry {attempt}/{retries}: {exc}", flush=True)
            if attempt == retries:
                raise
            _time.sleep(delay)
    raise RuntimeError("unreachable")


def score_and_import(source: str, seqs: set[str], archive: pd.DataFrame,
                     gpus: str) -> int:
    from nanobody_opt.mutation_network import update_observation_store

    seqs = sorted(seqs)
    if not seqs:
        print(f"  {source}: nothing new", flush=True)
        return 0
    print(f"  {source}: scoring {len(seqs)} new sequences at 20x", flush=True)
    out_root = OUT_ROOT / source
    out_root.mkdir(parents=True, exist_ok=True)
    per_pass: list[dict[str, float]] = []
    for run in range(1, N_PASSES + 1):
        per_pass.append(score_pass(seqs, run, archive, out_root, gpus))
        print(f"    pass {run}/{N_PASSES} done", flush=True)
    common = sorted(set.intersection(*(set(p) for p in per_pass)))
    obs = []
    for s in common:
        mean = sum(p[s] for p in per_pass) / N_PASSES
        obs.append({
            "observation_id": f"hf20x:{s}",
            "sequence": s,
            "target": TARGET,
            "source": source,
            "score_kind": SCORE_KIND,
            "score": float(mean),
            "observation_count": N_PASSES,
        })
    written = _retry_db(update_observation_store, Path("/root/.sn68_mysql_creds"),
                        obs, commit_every=1000, commit_interval=5.0)
    print(f"  {source}: imported {written}", flush=True)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--epoch", type=int, default=None,
                    help="explicit epoch; default = latest completed")
    args = ap.parse_args()

    sys.path.insert(0, "/root")
    from nanobody_opt import db

    archive = pd.read_csv(ARCHIVE)
    archive["sequence"] = archive["sequence"].astype(str).str.strip().str.upper()

    if args.epoch is None:
        # Walk back from the highest epoch the DB has ever seen (the producer's
        # current epoch) to the newest COMPLETED epoch on the leaderboard API.
        with db.connect() as conn:
            max_epoch = conn.execute(
                "SELECT COALESCE(MAX(epoch), 0) FROM observations"
            ).fetchone()[0]
        args.epoch = latest_completed_epoch(int(max_epoch) + 2)
        if args.epoch is None:
            print("no completed epoch found", flush=True)
            return 1
    print(f"epoch={args.epoch}", flush=True)

    with db.connect() as conn:
        scored = already_scored(conn)
        our = fetch_our_top50(conn)
        db_rivals = fetch_db_rivals(conn)

    leaderboard_rivals = fetch_rivals(args.epoch)
    # Union of the latest leaderboard rivals and every rival we have ever
    # scored, so a sequence skipped by an earlier backfill snapshot still gets
    # caught and 20x-scored here.
    rivals = leaderboard_rivals | db_rivals
    hf = fetch_hf_top200(archive)

    print(f"pool already scored={len(scored)}; new: ours={len(our - scored)} "
          f"rivals={len(rivals - scored)} hf={len(hf - scored)}", flush=True)

    total = 0
    total += score_and_import("our_top50_20x", our - scored, archive, args.gpus)
    total += score_and_import("public_multipass_rivals", rivals - scored,
                              archive, args.gpus)
    total += score_and_import("hf_archive_top200", hf - scored, archive, args.gpus)
    print(f"REFRESH_DONE imported={total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
