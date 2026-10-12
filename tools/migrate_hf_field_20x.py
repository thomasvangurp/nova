#!/usr/bin/env python3
"""Salvage the orphaned master 20x HF-field data (ARCHITECTURE_REDESIGN P3/P4/R1).

Background. ``hf_field_refresh.py`` on master 20-pass scored rivals, the HF
top-200 and our top-50 (2026-10-08/09) and stored ONE row per sequence: the
20-pass mean of a field-relative formula, ``observation_id='hf20x:<seq>'``,
``epoch`` NULL, no ``metrics_json``, no provenance. Nothing ever read it, and
the run was killed on 2026-10-09 ~23:47Z (P0). Those rows cannot be
re-derived: they hold no raw metrics. But the raw per-pass BoltzGen metrics
of most of the runs are still on master disk, in
``/root/hf_field_refresh/<source>/score<k>/wave_000_exact*.json``. The
scorer's wave cache archived each older run's pass under
``*.legacy-pre-dual-sample-<uuid>.json`` instead of deleting it. That name
is misleading: every file here carries ``scorer_protocol=validator-dual-sample-v1``.

This tool has two phases. Both default to a dry run.

``--apply-passes``
    Writes every recovered pass as one ``multipass_sum10_pass`` row:
    source ``public_multipass_rivals``, writer ``migrate_hf_field_20x``, raw
    ``metrics_json``, and sum-of-10 re-derived against ``--archive`` (field
    fingerprint recorded). The ``epoch`` is the epoch the refresh run scored
    for, from ``refresh.log`` -- except for the killed 25616 re-score batch
    (every rival ever seen, 3 of 20 passes): tagging those 2,908 sequences
    25616 would pull ~2,750 old rivals into the top-up reader's 16-epoch
    "current rival field" window. Each of its passes instead gets the
    sequence's own latest epoch <= the run epoch (its last full refresh run
    or its last ``public_multipass_rivals`` row); see :func:`rescore_epochs`.
    ``cohort`` is derived per sequence (see
    :func:`ours_set`). It uses the same scorer, protocol and per-pass semantics
    as the dashboard multipass rows, so the same label applies (checked by
    ``--dry-run``'s paired comparison). The ids are
    ``hf20x_pass:<source>:epoch<E>:<seq>:pass<k>``, which never collide with the
    dashboard's ``public_multipass:`` ids. The writes are idempotent upserts.

``--apply-relabel``
    Moves the per-sequence mean rows to explicit legacy labels. Scores are
    untouched. Provenance is added (formula, evidence, and where the raw
    passes went):

    * ``hf_field_20x_rank_sum`` -> ``legacy_hf_field_algo2_20x_mean``
      (algo2: stored == algo2 recomputed from the disk passes, per run);
    * ``hf_field_sum10_20x_rank_sum`` -> ``legacy_hf_field_sum10_20x_mean``
      (sum10, our top-50 only; stored == sum10 recomputed, rho=1.000).

The dry run also simulates ``hf_topup.ranked_topup`` with and without the
migrated passes against the newest producer iteration's ``hf_archive.csv``.
The simulation is read-only. It is the before/after evidence.

Usage (master, metanano env, central-DB creds)::

    /root/metanano_python.sh /root/migrate_hf_field_20x.py              # dry run + simulation
    /root/metanano_python.sh /root/migrate_hf_field_20x.py --apply-passes --apply-relabel
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _root in (HERE, HERE.parent, Path("/root")):
    if (_root / "nanobody_opt").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
_MP_PATH = next(c / "multipass_all_rivals.py"
                for c in (HERE, HERE.parent / "tools", Path("/root"))
                if (c / "multipass_all_rivals.py").is_file())
_spec = importlib.util.spec_from_file_location("multipass_all_rivals", _MP_PATH)
mp = importlib.util.module_from_spec(_spec)
sys.modules["multipass_all_rivals"] = mp
_spec.loader.exec_module(mp)

from nanobody_opt import observation_labels  # noqa: E402

WRITER = "migrate_hf_field_20x"
TARGET = "P20809"
SOURCE = "public_multipass_rivals"
PROTOCOL = "validator-dual-sample-v1"
OLD_ALGO2 = "hf_field_20x_rank_sum"
OLD_SUM10 = "hf_field_sum10_20x_rank_sum"
NEW_ALGO2 = "legacy_hf_field_algo2_20x_mean"
NEW_SUM10 = "legacy_hf_field_sum10_20x_mean"
#: refresh source dir -> cohort when the sequence is not ours
SOURCE_COHORT = {"public_multipass_rivals": "rival", "hf_archive_top200": "hf_top200",
                 "our_top50_20x": "ours"}
#: algo2 was the refresh formula until the sum10 deploy on master (backup
#: /root/hf_field_refresh.py.pre-sum10-deploy-20261009 = algo2 build, unchanged
#: since 10-08T23:46Z; the sum10 build is dated 10-09T21:13Z). Runs from
#: epoch 25616 on (started 21:14:22Z) used sum10.
SUM10_FROM_EPOCH = 25616


# --------------------------------------------------------------------------
# Disk runs
# --------------------------------------------------------------------------
def parse_refresh_log(path: Path) -> list[dict]:
    """Ordered ``{epoch, source, n, imported}`` for every scoring step."""
    steps: list[dict] = []
    epoch = None
    for line in path.read_text().splitlines():
        m = re.match(r"epoch=(\d+)", line.strip())
        if m:
            epoch = int(m.group(1))
            continue
        m = re.match(r"\s*(\S+): scoring (\d+) new sequences", line)
        if m:
            steps.append({"epoch": epoch, "source": m.group(1), "n": int(m.group(2)),
                          "imported": False})
            continue
        m = re.match(r"\s*(\S+): imported (\d+)", line)
        if m and steps and steps[-1]["source"] == m.group(1):
            steps[-1]["imported"] = True
    return steps


def load_disk_runs(root: Path) -> list[dict]:
    """Every distinct scored batch on disk: ``{source, n, passes: {k: (path,
    raw)}, t0, t1}``, oldest first. ``exact_all.json`` duplicates its
    ``wave_000_exact.json`` and is skipped; a pass seen twice keeps one file."""
    groups: dict[tuple, dict] = {}
    for src in sorted(p for p in root.iterdir() if p.is_dir()):
        for score_dir in src.glob("score*"):
            if not score_dir.name[5:].isdigit():
                continue
            k = int(score_dir.name[5:])
            for f in sorted(score_dir.glob("wave_000_exact*.json")):
                payload = json.loads(f.read_text())
                if payload.get("scorer_protocol") != PROTOCOL:
                    raise SystemExit(f"{f}: scorer_protocol "
                                     f"{payload.get('scorer_protocol')!r} != {PROTOCOL}")
                raw = mp.raw_metrics_from_exact_all(payload)
                if not raw or int(payload.get("n", -1)) != len(raw):
                    raise SystemExit(f"{f}: incomplete wave (n={payload.get('n')}, "
                                     f"parsed={len(raw)})")
                digest = hashlib.sha1("\n".join(sorted(raw)).encode()).hexdigest()[:10]
                g = groups.setdefault((src.name, digest), {
                    "source": src.name, "n": len(raw), "passes": {},
                    "t0": f.stat().st_mtime, "t1": f.stat().st_mtime, "payloads": {}})
                g["t0"] = min(g["t0"], f.stat().st_mtime)
                g["t1"] = max(g["t1"], f.stat().st_mtime)
                if k not in g["passes"]:
                    g["passes"][k] = f
                    g["payloads"][k] = payload
    return sorted(groups.values(), key=lambda g: g["t0"])


def assign_epochs(runs: list[dict], steps: list[dict]) -> None:
    """Match each disk batch to its refresh.log step (same source and size,
    in order). Refuses on any batch it cannot place."""
    pending = collections.defaultdict(list)
    for step in steps:
        pending[(step["source"], step["n"])].append(step)
    for run in runs:
        queue = pending.get((run["source"], run["n"]))
        if not queue:
            raise SystemExit(f"disk batch {run['source']} n={run['n']} has no "
                             "refresh.log step; refusing to guess its epoch")
        step = queue.pop(0)
        run["epoch"] = step["epoch"]
        run["imported"] = step["imported"]


def is_rescore(run: dict) -> bool:
    """The killed full re-score batch: never imported, fewer than 20 passes."""
    return len(run["passes"]) < 20 and not run["imported"]


def rescore_epochs(conn, runs: list[dict]) -> dict[str, tuple[int, str]]:
    """``{seq: (epoch, basis)}`` for every sequence of a re-score batch: the
    latest epoch <= that run's epoch at which the sequence was really in the
    rival field (an earlier full refresh run, or a DB pass row)."""
    out: dict[str, tuple[int, str]] = {}
    for run in runs:
        if not is_rescore(run):
            continue
        seqs = set(mp.raw_metrics_from_exact_all(next(iter(run["payloads"].values()))))
        best = {s: (0, "") for s in seqs}
        for other in runs:
            if other is run or is_rescore(other) or other["epoch"] > run["epoch"]:
                continue
            for s in seqs & set(mp.raw_metrics_from_exact_all(
                    next(iter(other["payloads"].values())))):
                if other["epoch"] > best[s][0]:
                    best[s] = (other["epoch"], f"refresh run {other['source']}@{other['epoch']}")
        ordered = sorted(seqs)
        for i in range(0, len(ordered), 300):
            chunk = ordered[i:i + 300]
            rows = conn.execute(
                "SELECT n.sequence, MAX(o.epoch) FROM observations o JOIN nodes n ON n.id=o.node_id "
                "WHERE o.source=%s AND o.epoch IS NOT NULL AND o.epoch<=%s "
                f"AND n.sequence IN ({','.join(['%s'] * len(chunk))}) GROUP BY n.sequence",
                (SOURCE, run["epoch"], *chunk)).fetchall()
            for seq, epoch in rows:
                seq = str(seq).strip().upper()
                if seq in best and epoch is not None and int(epoch) > best[seq][0]:
                    best[seq] = (int(epoch), f"latest {SOURCE} row epoch")
        for s, (epoch, basis) in best.items():
            out[s] = (epoch, basis) if epoch else (run["epoch"], "run epoch (no prior epoch)")
    return out


# --------------------------------------------------------------------------
# Cohort
# --------------------------------------------------------------------------
def ours_set(conn, extra: set[str]) -> tuple[set[str], dict]:
    """Our sequences: provenance cohort=ours on any pass row, our coldkey on
    any published leaderboard since the first multipass epoch, the Pi's
    intended drops, and ``extra`` (the our_top50_20x batch)."""
    ours = set(extra)
    evidence = {"our_top50_20x": len(extra)}
    rows = conn.execute(
        "SELECT DISTINCT n.sequence FROM observations o JOIN nodes n ON n.id=o.node_id "
        "WHERE o.source=%s AND JSON_UNQUOTE(JSON_EXTRACT(o.provenance_json,'$.cohort'))='ours'",
        (SOURCE,)).fetchall()
    db_ours = {str(r[0]).strip().upper() for r in rows}
    evidence["db_cohort_ours"] = len(db_ours)
    ours |= db_ours
    lo, hi = conn.execute("SELECT MIN(epoch), MAX(epoch) FROM observations "
                          "WHERE source=%s", (SOURCE,)).fetchone()
    board_ours: set[str] = set()
    missing = 0
    for epoch in range(int(lo), int(hi) + 1):
        board = mp.fetch_leaderboard(epoch)
        if board is None:
            missing += 1
        for ck, seq in board or []:
            if ck == mp.OUR_COLDKEY:
                board_ours.add(seq)
        board_ours |= {seq for _, seq in mp.fetch_our_intended(epoch)}
    evidence.update({"leaderboard_or_intended_ours": len(board_ours),
                     "leaderboard_epochs": f"{lo}-{hi}", "leaderboard_unavailable": missing})
    ours |= board_ours
    evidence["total"] = len(ours)
    return ours, evidence


# --------------------------------------------------------------------------
# Verification of the stored means
# --------------------------------------------------------------------------
def recompute_means(run: dict, archive: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """``{seq: (algo2 mean, sum10 mean)}`` over the run's passes, with the
    refresh's own field (archive + the whole batch)."""
    arch = archive.set_index("sequence")[mp.METRICS_NEEDED]
    a2 = collections.defaultdict(list)
    s10 = collections.defaultdict(list)
    for payload in run["payloads"].values():
        raw = mp.raw_metrics_from_exact_all(payload)
        batch = pd.DataFrame({s: {m: v.get(m) for m in mp.METRICS_NEEDED}
                              for s, v in raw.items()}).T
        field = pd.concat([arch, batch])
        field = field[~field.index.duplicated(keep="last")]
        fa, fs = mp.algo2_score(field), mp.sum_of_ten(field)
        for s in batch.index:
            if s in fa.index and np.isfinite(fa[s]):
                a2[s].append(float(fa[s]))
            s10[s].append(float(fs[s]))
    n = len(run["payloads"])
    return {s: (float(np.mean(a2[s])) if len(a2[s]) == n else float("nan"),
                float(np.mean(s10[s]))) for s in s10}


def spearman(x, y) -> float:
    x, y = pd.Series(x).rank(), pd.Series(y).rank()
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------
def build_observations(runs: list[dict], archive: pd.DataFrame, field: dict,
                       ours: set[str], stored: dict[str, tuple],
                       rebased: dict[str, tuple[int, str]]) -> list[dict]:
    now = int(time.time())
    out: list[dict] = []
    for run in runs:
        for k, payload in sorted(run["payloads"].items()):
            ranks, _structs, raw = mp.rank_pass_payload(payload, archive)
            path = run["passes"][k]
            for seq, score in ranks.items():
                old = stored.get(seq)
                cohort = "ours" if seq in ours else SOURCE_COHORT[run["source"]]
                if is_rescore(run):
                    epoch, basis = rebased[seq]
                else:
                    epoch, basis = run["epoch"], "refresh run epoch (refresh.log)"
                out.append({
                    # the id keeps the RUN epoch: unique per (run, seq, pass)
                    "observation_id": f"hf20x_pass:{run['source']}:epoch{run['epoch']}:{seq}:pass{k}",
                    "sequence": seq, "target": TARGET, "epoch": epoch,
                    "epoch_basis": basis,
                    "source": SOURCE, "score_kind": mp.PASS_SCORE_KIND,
                    "score": float(score), "observation_count": 1,
                    "metrics": raw.get(seq) or {},
                    # provenance
                    "writer": WRITER, "formula": mp.PASS_FORMULA_ID, "cohort": cohort,
                    "pass_index": k, "passes": len(run["payloads"]),
                    **field,
                    "scorer_protocol": PROTOCOL,
                    "migrated_from": "master hf_field_refresh disk pass",
                    "migrated_from_path": str(path),
                    "migrated_from_mtime": int(path.stat().st_mtime),
                    "migrated_at": now,
                    "hf20x_source": run["source"], "hf20x_run_epoch": run["epoch"],
                    "hf20x_run_imported": bool(run["imported"]),
                    "original_observation_id": f"hf20x:{seq}",
                    "original_score_kind": old[0] if old else None,
                    "original_score": old[1] if old else None,
                })
    return out


def passes_frame(observations: list[dict]) -> pd.DataFrame:
    recs = []
    for o in observations:
        rec = {m: o["metrics"].get(m) for m in mp.METRICS_NEEDED}
        rec.update(sequence=o["sequence"], epoch=o["epoch"], cohort=o["cohort"])
        recs.append(rec)
    return pd.DataFrame.from_records(recs)


# --------------------------------------------------------------------------
# Before/after simulation of the live reader
# --------------------------------------------------------------------------
def simulate(conn, new_obs: list[dict], eval_archive: Path, live_index: Path | None
             ) -> dict:
    from nanobody_opt import hf_topup as H

    frame = pd.read_csv(eval_archive)
    hf_top = H.hf_archive_top(frame, H.HF_TOP_K)
    validator_top = hf_top[:H.VALIDATOR_TOP_N]
    base = H.load_topup_passes(conn, hf_top)
    latest = conn.execute("SELECT MAX(epoch) FROM observations WHERE source=%s "
                          "AND score_kind=%s", (H.PASS_SOURCE, H.PASS_SCORE_KIND)).fetchone()[0]
    new = passes_frame(new_obs)
    in_reader = new[(new["epoch"] > int(latest) - H.RIVAL_EPOCH_WINDOW)
                    | new["sequence"].isin(set(hf_top))]
    primary: list[str] = []
    filler: list[str] = []
    if live_index and live_index.is_file():
        d = json.loads(live_index.read_text())
        topup_seqs = set((d.get("topup") or {}).get("sequences") or [])
        rest = [s for s in d["live_sequences"] if s not in topup_seqs]
        primary, filler = rest[:50], rest[50:]
    mlen = collections.Counter(len(s) for s in primary).most_common(1)[0][0] if primary else None

    def ranked(passes: pd.DataFrame) -> dict:
        key = ["sequence", *H.SUM_OF_TEN_DIRS]
        passes = passes.drop_duplicates(subset=key).reset_index(drop=True)
        r = H.rank_topup_passes(passes, frame)
        keep = set(H.eligible_parents([x["sequence"] for x in r], validator_top,
                                      H.MIN_PARENT_DISTANCE))
        elig = [x for x in r if x["sequence"] in keep]
        usable = [x for x in elig if (mlen is None or len(x["sequence"]) == mlen)
                  and x["sequence"] not in filler and x["sequence"] not in primary]
        slots = usable[:H.DEFAULT_SLOTS]
        return {"passes_read": int(len(passes)), "ranked": len(r), "eligible": len(elig),
                "hf_top200_eligible": sum(1 for x in elig if x["sequence"] in set(hf_top)),
                "slot_cutoff": slots[-1]["derived_sum10_mean"] if slots else None,
                "slots": [(x["sequence"], round(x["derived_sum10_mean"], 1), x["passes"],
                           x["cohort"]) for x in slots],
                "_elig": elig}

    before = ranked(base)
    after = ranked(pd.concat([base, in_reader], ignore_index=True))
    cf = ranked(pd.concat([base, new], ignore_index=True))
    migrated = set(new["sequence"])
    base_seqs = set(base["sequence"]) if not base.empty else set()
    out = {"eval_archive": str(eval_archive), "latest_pass_epoch": int(latest),
           "reader_window": f"epoch > {int(latest) - H.RIVAL_EPOCH_WINDOW} or in HF top-200",
           "migrated_rows": int(len(new)), "migrated_rows_in_reader": int(len(in_reader)),
           "migrated_sequences": len(migrated),
           "migrated_sequences_in_reader": int(in_reader["sequence"].nunique()),
           "migrated_hf_top200_now": len(migrated & set(hf_top)),
           "migrated_already_in_reader": len(migrated & base_seqs)}
    for name, res in (("before", before), ("after", after), ("counterfactual_no_window", cf)):
        elig = res.pop("_elig")
        new_only = [x for x in elig if x["sequence"] in migrated
                    and x["sequence"] not in base_seqs]
        res["newly_ranked_from_migration"] = len(new_only)
        res["best_new"] = [(x["sequence"][:16], round(x["derived_sum10_mean"], 1),
                            x["passes"], x["cohort"], elig.index(x) + 1)
                           for x in new_only[:5]]
        out[name] = res
    return out


def paired_check(conn, new_obs: list[dict], archive: pd.DataFrame) -> dict:
    """Same sequence, dashboard pass rows vs migrated master passes: per-metric
    mean difference, in units of the dashboard's per-pass SD (scorer check)."""
    from nanobody_opt import hf_topup as H
    new = passes_frame(new_obs)
    seqs = sorted(set(new["sequence"]))
    rows = []
    for i in range(0, len(seqs), 200):
        chunk = seqs[i:i + 200]
        rows += conn.execute(
            "SELECT n.sequence, o.epoch, o.metrics_json, o.provenance_json FROM observations o "
            "JOIN nodes n ON n.id=o.node_id WHERE o.source=%s AND o.score_kind=%s "
            "AND o.metrics_json IS NOT NULL AND o.metrics_json<>'{}' "
            "AND JSON_UNQUOTE(JSON_EXTRACT(o.provenance_json,'$.writer'))<>%s "
            f"AND n.sequence IN ({','.join(['%s'] * len(chunk))})",
            (SOURCE, mp.PASS_SCORE_KIND, WRITER, *chunk)).fetchall()
    dash = H._passes_frame(rows)
    if dash.empty:
        return {"overlap_sequences": 0}
    both = sorted(set(dash["sequence"]) & set(new["sequence"]))
    out = {"overlap_sequences": len(both), "dashboard_rows": int(len(dash))}
    z = {}
    for m in mp.METRICS_NEEDED:
        d = dash[dash.sequence.isin(both)].groupby("sequence")[m].agg(["mean", "std"])
        n = new[new.sequence.isin(both)].groupby("sequence")[m].mean()
        diff = (n - d["mean"]) / d["std"].replace(0, np.nan)
        z[m] = round(float(np.nanmedian(diff)), 3)
    out["median_z_master_minus_dashboard"] = z
    dash["derived"] = H.sum10_against_field(dash, archive)
    new["derived"] = H.sum10_against_field(new, archive)
    dm = dash[dash.sequence.isin(both)].groupby("sequence")["derived"].mean()
    nm = new[new.sequence.isin(both)].groupby("sequence")["derived"].mean()
    out["derived_sum10_mean_rho"] = round(spearman(dm[both], nm[both]), 4)
    out["derived_sum10_median_rel_diff"] = round(float(np.median((nm[both] - dm[both]) / dm[both])), 4)
    return out


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("/root/hf_field_refresh"))
    parser.add_argument("--log", type=Path,
                        default=Path("/root/hf_field_refresh_logs/refresh.log"))
    parser.add_argument("--archive", type=Path, default=Path("/root/P20809_nanobodies.csv"),
                        help="field the stored sum10 score is ranked against")
    parser.add_argument("--eval-archive", type=Path, default=None,
                        help="archive for the reader simulation (default: newest "
                             "producer iteration's hf_archive.csv)")
    parser.add_argument("--apply-passes", action="store_true")
    parser.add_argument("--apply-relabel", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--include-ours", action="store_true",
                        help="also import cohort=ours passes (default: no; see below)")
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    from nanobody_opt import db
    mp.assert_pass_label()
    observation_labels.check_observations([
        {"score_kind": mp.PASS_SCORE_KIND, "source": SOURCE, "writer": WRITER,
         "formula": mp.PASS_FORMULA_ID},
        {"score_kind": NEW_ALGO2, "source": SOURCE, "writer": WRITER, "formula": "algo2"},
        {"score_kind": NEW_SUM10, "source": "our_top50_20x", "writer": WRITER,
         "formula": "sum10"},
    ])

    archive = pd.read_csv(args.archive).drop_duplicates("sequence", keep="last")
    archive["sequence"] = archive["sequence"].astype(str).str.strip().str.upper()
    field = mp.field_fingerprint(args.archive, archive)
    report: dict = {"field_rows": field["field_rows"], "field_sha256": field["field_sha256"]}

    runs = load_disk_runs(args.root)
    assign_epochs(runs, parse_refresh_log(args.log))
    report["disk_runs"] = [{"source": r["source"], "epoch": r["epoch"], "n": r["n"],
                            "passes": len(r["passes"]), "imported": r["imported"],
                            "written": time.strftime("%m-%dT%H:%MZ", time.gmtime(r["t0"]))}
                           for r in runs]

    with db.connect() as conn:
        stored_rows = conn.execute(
            "SELECT o.observation_id, o.score_kind, o.source, n.sequence, o.score, "
            "o.metrics_json FROM observations o JOIN nodes n ON n.id=o.node_id "
            "WHERE o.score_kind IN (%s, %s)", (OLD_ALGO2, OLD_SUM10)).fetchall()
    stored = {str(r[3]).strip().upper(): (r[1], float(r[4]), r[2], str(r[0]),
                                          bool(r[5] and r[5] not in ("{}", "")))
              for r in stored_rows}
    census = collections.Counter((r[1], r[2]) for r in stored_rows)
    report["db_rows"] = {f"{k}|{s}": n for (k, s), n in sorted(census.items())}
    report["db_rows_with_metrics"] = sum(1 for v in stored.values() if v[4])

    # 1. Formula of every stored mean that has its run's passes on disk.
    verify: dict[str, dict] = {}
    covered: dict[str, str] = {}   # seq -> evidence string
    for run in runs:
        if len(run["passes"]) < 20:
            continue
        means = recompute_means(run, archive)
        pairs = [(stored[s][1], a, b) for s, (a, b) in means.items()
                 if s in stored and stored[s][2] == run["source"]]
        if not pairs:
            continue
        v = np.array(pairs)
        ok = np.isfinite(v[:, 1])
        formula = "sum10" if run["epoch"] >= SUM10_FROM_EPOCH else "algo2"
        rec = {"n": len(v), "expected": formula,
               "algo2_rho": round(spearman(v[ok, 0], v[ok, 1]), 4),
               "algo2_med_rel": round(float(np.median(np.abs(v[ok, 0] - v[ok, 1]) / v[ok, 0])), 4),
               "sum10_rho": round(spearman(v[:, 0], v[:, 2]), 4),
               "sum10_med_rel": round(float(np.median(np.abs(v[:, 0] - v[:, 2]) / v[:, 0])), 4)}
        verify[f"{run['source']}@{run['epoch']}"] = rec
        for s in means:
            if s in stored and stored[s][2] == run["source"]:
                covered[s] = (f"stored == {formula} recomputed from the run's 20 disk "
                              f"passes ({run['source']} epoch {run['epoch']}): "
                              f"{formula}_rho={rec[formula + '_rho']}, "
                              f"{formula}_med_rel={rec[formula + '_med_rel']}")
    report["verification"] = verify
    report["stored_rows_verified_on_disk"] = len(covered)

    with db.connect() as conn:
        ours_extra = {s for r in runs if r["source"] == "our_top50_20x" for s in
                      mp.raw_metrics_from_exact_all(next(iter(r["payloads"].values())))}
        ours, ours_ev = ours_set(conn, ours_extra)
        rebased = rescore_epochs(conn, runs)
    report["ours_evidence"] = ours_ev
    report["rescore_epoch_basis"] = dict(collections.Counter(
        b.split("@")[0] for _e, b in rebased.values()))

    observations = build_observations(runs, archive, field, ours, stored, rebased)
    on_disk = {o["sequence"] for o in observations}
    # cohort=ours is NOT imported by default. The top-up reader drops it
    # (ours are ranked on the producer scale), so it gains nothing. It also
    # costs something: MutationNetwork still pools every score_kind into a
    # node's robust score (ARCHITECTURE_REDESIGN new finding 1, unfixed), and
    # 17 of the our_top50_20x sequences have producer-scale rows and no
    # validator row, so 20 extra ~1.2e5 rows would move their live node score.
    # Their raw passes stay on master disk.
    if not args.include_ours:
        skipped = [o for o in observations if o["cohort"] == "ours"]
        observations = [o for o in observations if o["cohort"] != "ours"]
        report["skipped_ours"] = {"rows": len(skipped),
                                  "sequences": len({o["sequence"] for o in skipped})}
    by = collections.Counter((o["hf20x_source"], o["cohort"]) for o in observations)
    report["migrated_rows"] = {f"{s}|{c}": n for (s, c), n in sorted(by.items())}
    seqs = collections.defaultdict(set)
    for o in observations:
        seqs[o["cohort"]].add(o["sequence"])
    depth = collections.Counter(o["sequence"] for o in observations)
    report["migrated_sequences"] = {c: len(v) for c, v in seqs.items()}
    report["migrated_depth_hist"] = dict(sorted(collections.Counter(depth.values()).items()))
    report["stored_rows_with_no_raw_anywhere"] = dict(collections.Counter(
        f"{v[0]}|{v[2]}" for s, v in stored.items() if s not in on_disk))

    with db.connect() as conn:
        report["paired_scorer_check"] = paired_check(conn, observations, archive)
        if not args.skip_simulation:
            eval_archive = args.eval_archive
            live_index = None
            if eval_archive is None:
                cands = sorted(glob.glob("/root/live/epoch*/producer/iteration_*/hf_archive.csv"),
                               key=lambda p: Path(p).stat().st_mtime)
                eval_archive = Path(cands[-1])
            live_index = eval_archive.with_name("top50_index.json")
            report["simulation"] = simulate(conn, observations, eval_archive, live_index)

    text = json.dumps(report, indent=1, default=str)
    print(text, flush=True)
    if args.report:
        args.report.write_text(text)

    if args.apply_passes:
        from nanobody_opt.mutation_network import update_observation_store
        written = 0
        for start in range(0, len(observations), args.batch):
            batch = observations[start:start + args.batch]
            for attempt in range(1, 6):
                try:
                    written += update_observation_store(
                        Path("/root/artifacts/nanobody_mutations.sqlite"), batch)
                    break
                except Exception as error:  # noqa: BLE001
                    print(f"  batch {start}: attempt {attempt} {type(error).__name__}: "
                          f"{error}", flush=True)
                    if attempt == 5 or not db.is_deadlock(error):
                        raise
                    time.sleep(5 * attempt)
            if start % 5000 == 0:
                print(f"  passes written {written}/{len(observations)}", flush=True)
        print(f"PASSES_DONE written={written}", flush=True)

    if args.apply_relabel:
        now = int(time.time())
        moved = 0
        plan = collections.defaultdict(list)
        for seq, (kind, score, source, oid, _has) in stored.items():
            new_kind = NEW_ALGO2 if kind == OLD_ALGO2 else NEW_SUM10
            formula = "algo2" if kind == OLD_ALGO2 else "sum10"
            evidence = covered.get(seq) or (
                "no raw passes left (not in DB, not on master disk); formula from the "
                "writer build (algo2 until the 10-09T21:13Z sum10 deploy) and the "
                "value band of the row's score")
            # (no per-row value in the patch: one UPDATE per evidence group,
            # not one connect per row)
            if seq in depth:
                to = f"multipass_sum10_pass hf20x_pass:*:{seq}:pass*"
            elif seq in ours:
                to = ("not imported (cohort ours): raw passes kept on master disk "
                      "under /root/hf_field_refresh/")
            else:
                to = None
            plan[(new_kind, kind, formula, evidence, to)].append(oid)
        for (new_kind, kind, formula, evidence, to), ids in plan.items():
            patch = json.dumps({"relabeled_from": kind, "relabeled_at": now,
                                "writer": WRITER, "original_writer": "hf_field_refresh / "
                                "import_hf_field_20x", "formula": formula,
                                "formula_evidence": evidence, "passes": 20,
                                "aggregate": "20-pass mean", "field": "hf_archive_full "
                                "snapshot at write time (unrecorded)",
                                "raw_passes_migrated_to": to})
            for start in range(0, len(ids), args.batch):
                chunk = ids[start:start + args.batch]
                marks = ",".join(["%s"] * len(chunk))
                for attempt in range(1, 6):
                    try:
                        with db.connect() as conn:
                            conn.execute(
                                "UPDATE observations SET score_kind=%s, provenance_json="
                                "JSON_MERGE_PATCH(COALESCE(NULLIF(provenance_json,''),'{}'), %s) "
                                f"WHERE observation_id IN ({marks}) AND score_kind=%s",
                                (new_kind, patch, *chunk, kind))
                            conn.commit()
                        break
                    except Exception as error:  # noqa: BLE001
                        if not db.is_deadlock(error) or attempt == 5:
                            raise
                        time.sleep(2 * attempt)
                moved += len(chunk)
        with db.connect() as conn:
            left = conn.execute("SELECT COUNT(*) FROM observations WHERE score_kind IN "
                                "(%s, %s)", (OLD_ALGO2, OLD_SUM10)).fetchone()[0]
        print(f"RELABEL_DONE moved={moved} left_under_old_labels={left}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
