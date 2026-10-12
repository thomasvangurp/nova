"""HF/rival parent top-up: one pass scorer, read by the producer by rank only.

ARCHITECTURE_REDESIGN R1. Until 2026-10-09 the live producer's parent pool
(``top50_index.json`` ``live_sequences``, built by ``build_top50_index``) came
only from our own generation history. The HF-archive top-200 and the rival
sequences were scored by a second full GPU scorer (``hf_field_refresh.py``,
20x) that nothing read, and that pipeline is off (P0/P3).

This module is the shared half of the replacement:

* **Writer side** (``tools/multipass_all_rivals.py`` on the dashboard box):
  every epoch, on top of the leaderboard batch it already scores, it tops up
  the HF-archive top-200 parent candidates to ``DEFAULT_DEPTH`` passes. It
  scores only the passes each sequence is *missing* (rows already stored under
  ``multipass_sum10_pass`` with raw metrics count), within a per-run budget
  of sequence-passes. See :func:`plan_topup`.
* **Reader side** (``build_top50_index`` on master, every producer epoch):
  :func:`ranked_topup` derives a sum-of-10 for every stored pass of the HF
  top-200 and recent rival sequences against *this epoch's* HF archive, from
  the stored raw metrics (R2: formula at read time, no GPU), and ranks them
  within their own list. :func:`merge_by_rank` then gives that list reserved
  slots in the producer-scale top-50 **by rank position**.

**Scale safety (R3).** The two lists are on different scales: the producer
replay rank-sum (~20-370, ``REPLAY_SCALE_SOURCES``) and a pass rank-sum vs the
full HF archive (~1e5). Their raw scores are never compared, pooled, averaged
or sorted together. :func:`merge_by_rank` takes two *ordered sequence lists*
and no scores at all, so it cannot compare them.
"""
from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "P20809"

#: The label and source the dashboard pass scorer writes (observation_labels).
PASS_SCORE_KIND = "multipass_sum10_pass"
PASS_SOURCE = "public_multipass_rivals"
#: provenance.cohort for top-up rows (R3's ``cohort`` vocabulary).
TOPUP_COHORT = "hf_top200"

#: Passes per HF candidate the top-up aims for. The same depth the dashboard
#: already runs for the leaderboard field; NOT the old 20x full re-fold.
DEFAULT_DEPTH = 10
#: How many of the HF archive's rank_binders order are candidates.
HF_TOP_K = 200
#: The validator dedups against the rank_binders top-50 of the full archive
#: (whole-sequence 0.95). A parent closer than this to that top-50 only yields
#: children the validator rejects, so it is not a parent candidate.
VALIDATOR_TOP_N = 50
MIN_PARENT_DISTANCE = 6
#: Per-run cap on top-up sequence-passes (one sequence scored once = 1).
#: 500 = 50 sequences x 10 passes, about the size of one leaderboard batch,
#: so a top-up run at most doubles the dashboard's per-epoch GPU time.
DEFAULT_PASS_BUDGET = 500

#: Reserved parent slots for the top-up list, and their spacing: every
#: ``DEFAULT_STRIDE``-th position of the merged list (1-indexed) is the next
#: top-up candidate. The competitor lane truncates its single edits at 2000
#: (~266 per parent, no ranker), so only the first ~8 parents ever get edits:
#: stride 4 puts top-up #1 and #2 at positions 4 and 8.
DEFAULT_SLOTS = 10
DEFAULT_STRIDE = 4
#: Recent epochs of rival multipass rows the reader considers.
RIVAL_EPOCH_WINDOW = 16
#: A candidate needs this many usable passes to be ranked at all.
MIN_PASSES = 8
#: Operator kill switch, read on every build (no restart).
DISABLE_FILE = Path("/root/artifacts/DISABLE_TOP50_TOPUP")

# The validator's final_score: sum of the ten per-metric dense ranks.
# Same table as tools/multipass_all_rivals.SUM_OF_TEN_DIRS (a test pins them
# equal; R2's validator_formulas.py is the eventual single home).
SUM_OF_TEN_DIRS = {
    "design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min", "interaction_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max", "liability_score": "min",
    "liability_num_violations": "min",
}


# --------------------------------------------------------------------------
# HF archive ranking (moved from hf_field_refresh.fetch_hf_top200 and
# build_top50_index._validator_rank_binders; one copy)
# --------------------------------------------------------------------------
def load_rank_binders():
    """The validator's own ``utils/minmax_weighted_rank.rank_binders``, by path.

    Not ``from utils import rank_binders``: ``utils/__init__.py`` imports
    rdkit, which neither the metanano nor the dashboard env carries.
    """
    source = Path(__file__).resolve().parent.parent / "utils" / "minmax_weighted_rank.py"
    spec = importlib.util.spec_from_file_location("_nova_minmax_weighted_rank", source)
    if spec is None or spec.loader is None or not source.is_file():
        raise SystemExit(f"cannot load the validator's ranking from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rank_binders


def hf_archive_top(frame: pd.DataFrame, k: int = HF_TOP_K) -> list[str]:
    """The HF archive's top-``k`` by the validator's rank_binders, best first."""
    top = load_rank_binders()(frame, k=k, max_liability_violations=50).head(k)
    out: list[str] = []
    for seq in top["sequence"].tolist():
        s = str(seq).strip().upper()
        if s and s not in out:
            out.append(s)
    return out


def eligible_parents(candidates: Sequence[str], reference: Sequence[str],
                     min_distance: int = MIN_PARENT_DISTANCE,
                     length: int | None = None) -> list[str]:
    """Candidates (order kept) whose Hamming distance to every same-length
    ``reference`` sequence is >= ``min_distance``; when ``length`` is given,
    other lengths are dropped (the competitor lane uses the modal length)."""
    refs: dict[int, np.ndarray] = {}
    for s in reference:
        refs.setdefault(len(s), []).append(s)  # type: ignore[arg-type]
    arrays = {n: np.frombuffer("".join(v).encode(), dtype=np.uint8).reshape(len(v), n)
              for n, v in refs.items()}
    out = []
    for s in candidates:
        if length is not None and len(s) != length:
            continue
        arr = arrays.get(len(s))
        if arr is not None:
            seq = np.frombuffer(s.encode(), dtype=np.uint8)
            if int((arr != seq).sum(axis=1).min()) < min_distance:
                continue
        out.append(s)
    return out


# --------------------------------------------------------------------------
# Writer side: which passes are missing
# --------------------------------------------------------------------------
def plan_topup(candidates: Sequence[str], existing: dict[str, int], *,
               exclude: Iterable[str] = (), depth: int = DEFAULT_DEPTH,
               budget: int = DEFAULT_PASS_BUDGET) -> dict[str, int]:
    """``{sequence: passes_to_score}`` for the bounded top-up, best first.

    A sequence with ``existing[s]`` stored passes gets ``depth - existing[s]``
    more, never a re-fold from zero. Sequences in ``exclude`` (this epoch's
    leaderboard batch, already being scored) are skipped. Candidates are taken
    in order until ``budget`` sequence-passes are used; a candidate that does
    not fit is cut to the remaining budget, and the rest wait for later runs.
    """
    skip = set(exclude)
    plan: dict[str, int] = {}
    left = max(0, int(budget))
    for s in candidates:
        if left <= 0:
            break
        if s in skip or s in plan:
            continue
        need = int(depth) - int(existing.get(s, 0))
        if need <= 0:
            continue
        take = min(need, left)
        plan[s] = take
        left -= take
    return plan


def pass_batches(plan: dict[str, int]) -> list[list[str]]:
    """Top-up sequences per pass: pass ``r`` (0-based) scores every sequence
    that still needs more than ``r`` passes."""
    rounds = max(plan.values(), default=0)
    return [[s for s, k in plan.items() if k > r] for r in range(rounds)]


def _ph(conn) -> str:
    """SQL placeholder for ``conn``: '?' for master's local SQLite store."""
    from nanobody_opt import db
    return "?" if isinstance(conn, db.LocalConnection) else "%s"


def stored_pass_counts(conn, sequences: Sequence[str], *,
                       target: str = TARGET) -> dict[str, int]:
    """Stored ``multipass_sum10_pass`` rows WITH raw metrics, per sequence.

    Rows without metrics (relabelled 25613-25618) cannot be re-derived, so
    they do not count towards the depth.
    """
    counts: dict[str, int] = {}
    p = _ph(conn)
    seqs = list(dict.fromkeys(sequences))
    for i in range(0, len(seqs), 200):
        chunk = seqs[i:i + 200]
        rows = conn.execute(
            f"""
            SELECT n.sequence, COUNT(*)
            FROM nodes n JOIN observations o ON o.node_id = n.id
            WHERE n.target = {p} AND n.sequence IN ({",".join([p] * len(chunk))})
              AND o.source = {p} AND o.score_kind = {p}
              AND o.metrics_json IS NOT NULL AND o.metrics_json <> '{{}}'
            GROUP BY n.sequence
            """,
            (target, *chunk, PASS_SOURCE, PASS_SCORE_KIND),
        ).fetchall()
        for seq, n in rows:
            counts[str(seq).strip().upper()] = int(n)
    return counts


# --------------------------------------------------------------------------
# Reader side: derived sum-of-10 vs this epoch's field, ranked in its own list
# --------------------------------------------------------------------------
def sum10_against_field(metrics: pd.DataFrame, field: pd.DataFrame) -> pd.Series:
    """Sum-of-10 of each row of ``metrics`` against ``field``, one row at a time.

    Equals ``sum_of_ten(field + [row])[row]`` for every row (dense rank of the
    row's value among the field's distinct values plus itself), vectorised.
    Rows missing any of the ten metrics get NaN.
    """
    total = pd.Series(0.0, index=metrics.index)
    for metric, direction in SUM_OF_TEN_DIRS.items():
        if metric not in metrics.columns or metric not in field.columns:
            total[:] = np.nan
            continue
        values = np.unique(pd.to_numeric(field[metric], errors="coerce").dropna().to_numpy(float))
        x = pd.to_numeric(metrics[metric], errors="coerce").to_numpy(float)
        if direction == "max":
            # distinct field values strictly greater than x, then x itself
            better = len(values) - np.searchsorted(values, x, side="right")
        else:
            better = np.searchsorted(values, x, side="left")
        rank = better.astype(float) + 1.0
        rank[np.isnan(x)] = np.nan
        total = total + rank
    return total


def _passes_frame(rows: Iterable[tuple]) -> pd.DataFrame:
    """(sequence, epoch, metrics_json, provenance_json) rows -> one row per pass."""
    records = []
    for seq, epoch, metrics_json, provenance_json in rows:
        try:
            metrics = json.loads(metrics_json or "{}")
            prov = json.loads(provenance_json or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(metrics, dict) or not metrics:
            continue
        rec = {k: metrics.get(k) for k in SUM_OF_TEN_DIRS}
        rec["sequence"] = str(seq).strip().upper()
        rec["epoch"] = epoch
        rec["cohort"] = str((prov or {}).get("cohort") or "")
        records.append(rec)
    return pd.DataFrame.from_records(records)


def rank_topup_passes(passes: pd.DataFrame, field: pd.DataFrame, *,
                      exclude: Iterable[str] = (), min_passes: int = MIN_PASSES,
                      ) -> list[dict]:
    """Rank candidates by mean derived sum-of-10 (lower = better).

    ``passes`` has one row per stored pass (metrics + sequence + cohort). Our
    own cohort and ``exclude`` are dropped: our sequences are ranked on the
    producer's own scale, in the other list.
    """
    if passes.empty:
        return []
    skip = set(exclude)
    work = passes[(passes["cohort"] != "ours") & ~passes["sequence"].isin(skip)].copy()
    if work.empty:
        return []
    work["derived"] = sum10_against_field(work, field)
    work = work[np.isfinite(work["derived"])]
    grouped = work.groupby("sequence").agg(
        derived_mean=("derived", "mean"), passes=("derived", "size"),
        cohort=("cohort", lambda c: ",".join(sorted(set(c)))),
        last_epoch=("epoch", "max"),
    )
    grouped = grouped[grouped["passes"] >= int(min_passes)]
    grouped = grouped.sort_values(["derived_mean", "passes"], ascending=[True, False])
    return [
        {"sequence": seq, "derived_sum10_mean": float(r.derived_mean),
         "passes": int(r.passes), "cohort": r.cohort,
         "last_epoch": None if r.last_epoch is None or (isinstance(r.last_epoch, float) and math.isnan(r.last_epoch)) else int(r.last_epoch)}
        for seq, r in grouped.iterrows()
    ]


def load_topup_passes(conn, hf_candidates: Sequence[str], *,
                      window: int = RIVAL_EPOCH_WINDOW, target: str = TARGET
                      ) -> pd.DataFrame:
    """Every stored pass (with metrics) of the HF candidates, plus every pass
    of the last ``window`` multipass epochs (the rival leaderboard field)."""
    p = _ph(conn)
    base = f"""
        SELECT n.sequence, o.epoch, o.metrics_json, o.provenance_json
        FROM observations o JOIN nodes n ON o.node_id = n.id
        WHERE n.target = {p} AND o.source = {p} AND o.score_kind = {p}
          AND o.metrics_json IS NOT NULL AND o.metrics_json <> '{{}}'
    """
    rows: list[tuple] = []
    latest = conn.execute(
        f"SELECT MAX(o.epoch) FROM observations o WHERE o.source = {p} AND o.score_kind = {p}",
        (PASS_SOURCE, PASS_SCORE_KIND),
    ).fetchone()
    if latest and latest[0] is not None:
        rows += conn.execute(base + f" AND o.epoch > {p}",
                             (target, PASS_SOURCE, PASS_SCORE_KIND,
                              int(latest[0]) - int(window))).fetchall()
    seqs = list(dict.fromkeys(hf_candidates))
    for i in range(0, len(seqs), 200):
        chunk = seqs[i:i + 200]
        rows += conn.execute(
            base + f" AND n.sequence IN ({','.join([p] * len(chunk))})",
            (target, PASS_SOURCE, PASS_SCORE_KIND, *chunk)).fetchall()
    frame = _passes_frame(rows)
    if frame.empty:
        return frame
    # The two reads overlap (an HF candidate scored in the window): one row
    # per stored pass. A pass is identified by its metrics, not its epoch.
    key_cols = ["sequence", *SUM_OF_TEN_DIRS]
    return frame.drop_duplicates(subset=key_cols).reset_index(drop=True)


def ranked_topup(conn, archive: pd.DataFrame, *, exclude: Iterable[str] = (),
                 length: int | None = None, hf_top: Sequence[str] | None = None,
                 window: int = RIVAL_EPOCH_WINDOW, min_passes: int = MIN_PASSES,
                 ) -> tuple[list[dict], dict]:
    """The HF/rival top-up list for this epoch, best first, plus a summary.

    Ranked by derived sum-of-10 vs ``archive`` (this epoch's HF field) from
    stored raw metrics; filtered to parent-eligible sequences (>=
    ``MIN_PARENT_DISTANCE`` from the validator's archive top-50, modal
    ``length``). Scores here are only ever compared with each other.
    """
    if hf_top is None:
        hf_top = hf_archive_top(archive, HF_TOP_K)
    validator_top = list(hf_top[:VALIDATOR_TOP_N])
    passes = load_topup_passes(conn, hf_top, window=window)
    ranked = rank_topup_passes(passes, archive, exclude=exclude, min_passes=min_passes)
    keep = set(eligible_parents([r["sequence"] for r in ranked], validator_top,
                                MIN_PARENT_DISTANCE, length))
    hf_set = set(hf_top)
    out = []
    for r in ranked:
        if r["sequence"] in keep:
            r["in_hf_top200"] = r["sequence"] in hf_set
            out.append(r)
    summary = {
        "stored_passes_read": int(len(passes)),
        "ranked": len(ranked),
        "eligible": len(out),
        "hf_top200_eligible": sum(1 for r in out if r["in_hf_top200"]),
        "score": "mean derived sum10 vs this epoch's HF archive (read time, "
                 "from multipass_sum10_pass metrics_json); compared only within "
                 "this list",
    }
    return out, summary


# --------------------------------------------------------------------------
# The merge: rank positions only
# --------------------------------------------------------------------------
def merge_by_rank(primary: Sequence[str], topup: Sequence[str], *,
                  slots: int = DEFAULT_SLOTS, stride: int = DEFAULT_STRIDE,
                  ) -> tuple[list[str], list[int]]:
    """Interleave two independently ranked lists by position, never by score.

    Every ``stride``-th position (1-indexed) of the result is the next unused
    ``topup`` sequence, for at most ``slots`` of them; every other position is
    the next ``primary`` sequence. All of ``primary`` is kept in its order.
    Top-up sequences already in ``primary`` are skipped. Returns the merged
    list and the 0-based positions the top-up occupies.

    This function receives no scores: the two lists' raw values (producer
    replay ~20-370 vs HF-field pass ~1e5) cannot be compared here.
    """
    if stride < 2:
        raise ValueError("stride must be >= 2 so the primary list keeps a majority")
    seen = set(primary)
    extra = []
    for s in topup:
        if s not in seen and len(extra) < max(0, int(slots)):
            extra.append(s)
            seen.add(s)
    merged: list[str] = []
    positions: list[int] = []
    p = t = 0
    while p < len(primary) or t < len(extra):
        if t < len(extra) and (len(merged) + 1) % stride == 0:
            positions.append(len(merged))
            merged.append(extra[t])
            t += 1
        elif p < len(primary):
            merged.append(primary[p])
            p += 1
        else:  # primary exhausted: the remaining top-up goes at the end
            positions.append(len(merged))
            merged.append(extra[t])
            t += 1
    return merged, positions
