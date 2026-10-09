#!/usr/bin/env python3
"""Empirical gate for any change to a scoring / ranking / selection formula.

QUESTION ANSWERED
    "Does the PROPOSED formula rank candidates better than the INCUMBENT one,
     judged against what the validator actually published?"

RULE (docs/SCORING_CHANGE_VALIDATION.md)
    Any change to a scoring, ranking or selection formula that affects live
    candidate choice must pass this check (or an equivalent empirical check
    against real historical outcomes) BEFORE it is deployed to a live box.
    Exit code 0 is the ONLY result that supports deploying the proposal.

GROUND TRUTH
    The public leaderboard API, one file per epoch, saved to disk before it is
    parsed (never piped into a parser):
        https://compound-api.metanova-labs.ai/api/competitions/leaderboard/{epoch}/nanobodies?limit=10000
    Every published entry carries the ten raw BoltzGen metrics and the
    validator's final_score, so a candidate formula can be recomputed over the
    exact field the validator ranked against.

TWO OUTCOME MODES
    --outcome own        Each candidate is a published submission (default:
                         ours, by coldkey). Outcome = its own published
                         final_score. Use for "does formula X reproduce/predict
                         the validator's score" questions.
    --outcome neighbors  Each candidate is a published sequence (a would-be
                         parent). Outcome = the mean within-epoch percentile the
                         validator gave its published relatives 1..--max-edits
                         Hamming edits away (optionally --future-only). Use for
                         parent / seed selection questions.

SCORE SOURCES (--incumbent / --proposed; LOWER = BETTER unless suffixed @higher)
    builtin:sum10          sum of the ten per-metric dense ranks within the
                           epoch field (the validator's actual formula)
    builtin:category3      rank each of the three metric categories, sum the
                           category ranks (the 2026-10-09 "corrected" formula
                           that was a regression -- kept as a known-bad control)
    builtin:algo2          Algorithm-2 minmax over weighted metric ranks
    builtin:published      the candidate's own published final_score
    builtin:published_pct  its within-epoch percentile (0 = best)
    csv:PATH[:COLUMN]      per-sequence precomputed scores (needs a `sequence`
                           column; COLUMN defaults to `score`; .csv.gz ok;
                           duplicate sequences are averaged)
    py:MODULE:FUNCTION     FUNCTION(field: DataFrame) -> array-like, called once
                           per epoch on that epoch's published field (columns =
                           the ten metrics + sequence, uid, coldkey, ...)

STATISTIC
    Spearman rank correlation. The validator's score is itself a sum of dense
    ranks and live selection is an ordering (top-k), so only the ORDER a formula
    induces matters; candidate formulas also live on unrelated scales (Algorithm-2
    ~26k-60k vs sum-of-ten ~10-500), which Pearson would conflate with quality.
    With --outcome own the primary statistic is the pooled Spearman over all
    candidates (as used for the 2026-10-09 decision); the within-epoch
    (stratified) Spearman is printed alongside because only within-epoch order
    matters to the validator. Uncertainty comes from a PAIRED bootstrap of
    rho(proposed) - rho(incumbent): resampling epochs (own mode -- candidates in
    one epoch share a field) or candidates (neighbors mode).

VERDICT / EXIT CODE
    0  PROPOSED IS BETTER        95% CI of the difference is above zero in the
                                 primary slice, no slice is significantly worse,
                                 and the sign agrees in every slice.
    1  NO EVIDENCE OF IMPROVEMENT  the CI contains zero somewhere relevant. This
                                 is NOT "safe to deploy" and NOT "equivalent".
    2  PROPOSED IS WORSE         some slice's CI is entirely below zero.
    3  INSUFFICIENT DATA / ERROR too few candidates to say anything.

KNOWN-ANSWER CASES (run these to check the tool itself still works)
    --case formula-20261009   sum-of-ten vs the 3-category "corrected" formula
                              over our last 10 epochs. Expected: incumbent
                              rho=1.000, proposed rho~0.92, verdict WORSE.
    --case parent20x-20261009 single-pass published score vs 20x multipass
                              (Algorithm-2) score for parent ranking. Expected:
                              NO EVIDENCE OF IMPROVEMENT (CIs straddle zero,
                              sign flips across slices).

EXAMPLES
    python tools/validate_scoring_change.py --case formula-20261009
    python tools/validate_scoring_change.py --last 15 \\
        --incumbent builtin:sum10 --proposed py:my_module:my_formula
    python tools/validate_scoring_change.py --outcome neighbors --max-edits 2 \\
        --epochs 25000-25615 --incumbent builtin:published \\
        --proposed csv:scores.csv:mean20x \\
        --slice 'frontier=incumbent>=100' --json-out verdict.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://compound-api.metanova-labs.ai/api/competitions/leaderboard/{epoch}/nanobodies?limit=10000"
TARGET = "P20809"
OUR_COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"
DEFAULT_CACHE = Path(os.environ.get(
    "NOVA_LEADERBOARD_CACHE",
    Path.home() / ".cache" / "nova-nanobody" / "leaderboard_nanobodies"))
REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "scoring_validation"

# Validator metric directions (min = lower is better).
METRICS = {
    "design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min", "interaction_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max", "liability_score": "min",
    "liability_num_violations": "min",
}
CATEGORIES = {
    "confidence": ("design_iiptm", "design_ptm", "design_to_target_iptm",
                   "min_design_to_target_pae", "interaction_pae"),
    "physical_interaction": ("plip_hbonds_refolded", "plip_saltbridge_refolded",
                             "delta_sasa_refolded"),
    "developability": ("liability_score", "liability_num_violations"),
}
ALGO2_WEIGHTS = {
    "design_iiptm": 1.0, "design_ptm": 2.0, "min_design_to_target_pae": 1.0,
    "plip_hbonds_refolded": 2.0, "plip_saltbridge_refolded": 2.0,
    "delta_sasa_refolded": 2.0,
}

EXIT_BETTER, EXIT_NO_EVIDENCE, EXIT_WORSE, EXIT_INSUFFICIENT = 0, 1, 2, 3


# --------------------------------------------------------------------------
# Ground truth: leaderboard download (to disk) and parsing (from disk)
# --------------------------------------------------------------------------

def parse_epochs(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _download(epoch: int, cache: Path, refresh: bool) -> Path | None:
    """Save one epoch's leaderboard JSON to disk; return the path or None.

    A 404 (epoch not scored yet, or no nanobody round) is recorded as a
    `.missing` marker only for epochs comfortably in the past, so a pending
    epoch is retried on the next run.
    """
    path = cache / f"{epoch}.json"
    if path.exists() and not refresh:
        return path
    if (cache / f"{epoch}.missing").exists() and not refresh:
        return None
    url = API.format(epoch=epoch)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nova-validate-scoring/1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
            tmp = path.with_suffix(".json.part")
            tmp.write_bytes(body)
            tmp.replace(path)
            return path
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            time.sleep(0.5 * 2 ** attempt)
        except Exception:
            time.sleep(0.5 * 2 ** attempt)
    print(f"  WARNING: could not download epoch {epoch}", file=sys.stderr)
    return None


def fetch_epochs(epochs: list[int], cache: Path, refresh: bool = False) -> dict[int, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda e: _download(e, cache, refresh), epochs))
    found = {e: p for e, p in zip(epochs, paths) if p is not None}
    if found:
        newest = max(found)
        for e, p in zip(epochs, paths):
            if p is None and e < newest - 3:
                (cache / f"{e}.missing").touch()
    return found


def latest_epochs(n: int, cache: Path, refresh: bool, start: int | None = None) -> list[int]:
    """Find the newest `n` published epochs by probing downward from `start`."""
    if start is None:
        # Probe upward from the newest cached epoch (or a recent anchor).
        cached = sorted(int(p.stem) for p in cache.glob("[0-9]*.json")) if cache.exists() else []
        e = cached[-1] if cached else 25600
        misses = 0
        while misses < 6:
            # Always re-check epochs past the newest cached one.
            if _download(e + 1, cache, refresh=True) is None:
                misses += 1
            else:
                misses = 0
            e += 1
        start = max(int(p.stem) for p in cache.glob("[0-9]*.json"))
    found: list[int] = []
    e, misses = start, 0
    while len(found) < n and misses < 30:
        if _download(e, cache, refresh) is not None:
            found.append(e)
            misses = 0
        else:
            misses += 1
        e -= 1
    return sorted(found)


def load_field(paths: dict[int, Path], target: str = TARGET) -> pd.DataFrame:
    """Parse saved leaderboard files into one row per published nanobody."""
    rows = []
    for epoch, path in sorted(paths.items()):
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            print(f"  WARNING: epoch {epoch} file is not JSON: {path}", file=sys.stderr)
            continue
        if not payload.get("success", True):
            continue
        for sub in payload.get("data") or []:
            for nb in sub.get("nanobodies") or []:
                if nb.get("protein_name") != target:
                    continue
                seq = str(nb.get("sequence") or "").strip().upper()
                fs = nb.get("final_score", sub.get("final_score"))
                if not seq or fs is None:
                    continue
                row = {"epoch": epoch, "uid": sub.get("uid"), "coldkey": sub.get("coldkey"),
                       "sequence": seq, "final_score": float(fs)}
                for m in METRICS:
                    v = nb.get(m)
                    row[m] = float(v) if v is not None else np.nan
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[np.isfinite(df.final_score) & (df.final_score > -900)].reset_index(drop=True)
    # within-epoch percentile of the published score, 0 = best
    df["published_pct"] = df.groupby("epoch").final_score.rank(pct=True, method="average")
    return df


# --------------------------------------------------------------------------
# Built-in formulas (all lower = better, evaluated per epoch field)
# --------------------------------------------------------------------------

def _dense(series: pd.Series, direction: str) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(method="dense", ascending=(direction == "min"))


def f_sum10(field: pd.DataFrame) -> np.ndarray:
    total = pd.Series(0.0, index=field.index)
    for m, d in METRICS.items():
        total += _dense(field[m], d)
    return total.to_numpy()


def f_category3(field: pd.DataFrame) -> np.ndarray:
    total = pd.Series(0.0, index=field.index)
    for names in CATEGORIES.values():
        cat = sum(_dense(field[m], METRICS[m]) for m in names)
        total += cat.rank(method="dense", ascending=True)
    return total.to_numpy()


def f_algo2(field: pd.DataFrame) -> np.ndarray:
    keep = field["liability_num_violations"] <= 50
    out = pd.Series(np.nan, index=field.index)
    work = field[keep]
    cols = {m: work[m].rank(ascending=(METRICS[m] == "min"), method="average") / w
            for m, w in ALGO2_WEIGHTS.items()}
    out[keep] = pd.DataFrame(cols).max(axis=1)
    return out.to_numpy()


FIELD_BUILTINS = {"sum10": f_sum10, "category3": f_category3, "algo2": f_algo2}
ROW_BUILTINS = {"published": "final_score", "published_pct": "published_pct"}


@dataclass
class ScoreSource:
    spec: str
    higher_is_better: bool = False

    @property
    def label(self) -> str:
        return self.spec + ("@higher" if self.higher_is_better else "")


def parse_source(spec: str) -> ScoreSource:
    higher = spec.endswith("@higher")
    return ScoreSource(spec[: -len("@higher")] if higher else spec, higher)


def apply_source(src: ScoreSource, lb: pd.DataFrame) -> pd.Series:
    """Score every row of the leaderboard frame; return lower-is-better values."""
    kind, _, rest = src.spec.partition(":")
    if kind == "builtin":
        if rest in ROW_BUILTINS:
            vals = lb[ROW_BUILTINS[rest]].astype(float)
        elif rest in FIELD_BUILTINS:
            vals = _per_epoch(lb, FIELD_BUILTINS[rest])
        else:
            raise SystemExit(f"unknown builtin {rest!r}; choose from "
                             f"{sorted(FIELD_BUILTINS) + sorted(ROW_BUILTINS)}")
    elif kind == "csv":
        path, _, col = rest.rpartition(":") if rest.count(":") >= 1 else (rest, "", "")
        if not path:
            path, col = rest, ""
        col = col or "score"
        table = pd.read_csv(path)
        if "sequence" not in table.columns or col not in table.columns:
            raise SystemExit(f"{path}: needs columns 'sequence' and {col!r}; has {list(table.columns)}")
        table["sequence"] = table["sequence"].astype(str).str.strip().str.upper()
        per_seq = table.groupby("sequence")[col].mean()
        vals = lb["sequence"].map(per_seq).astype(float)
    elif kind == "py":
        mod_name, _, fn_name = rest.partition(":")
        sys.path.insert(0, str(REPO))
        fn = getattr(importlib.import_module(mod_name), fn_name)
        vals = _per_epoch(lb, fn)
    else:
        raise SystemExit(f"bad score source {src.spec!r} (builtin:/csv:/py:)")
    vals = pd.Series(np.asarray(vals, dtype=float), index=lb.index)
    return -vals if src.higher_is_better else vals


def _per_epoch(lb: pd.DataFrame, fn) -> pd.Series:
    out = pd.Series(np.nan, index=lb.index)
    for _, grp in lb.groupby("epoch"):
        out.loc[grp.index] = np.asarray(fn(grp.copy()), dtype=float)
    return out


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------

def neighbor_outcomes(lb: pd.DataFrame, candidates: pd.DataFrame, max_edits: int,
                      future_only: bool) -> pd.Series:
    """Mean within-epoch percentile of each candidate's published relatives.

    Reference = every published sequence in the window, first appearance only.
    Relatives = same length, Hamming distance 1..max_edits (and, with
    future_only, published in a strictly later epoch than the candidate).
    """
    ref = lb.sort_values("epoch").drop_duplicates("sequence", keep="first")
    out = pd.Series(np.nan, index=candidates.index)
    for length, ref_l in ref.groupby(ref.sequence.str.len()):
        cand_l = candidates[candidates.sequence.str.len() == length]
        if cand_l.empty:
            continue
        R = np.frombuffer("".join(ref_l.sequence).encode(), dtype=np.uint8).reshape(-1, length)
        r_ep = ref_l.epoch.to_numpy()
        r_pct = ref_l.published_pct.to_numpy()
        Q = np.frombuffer("".join(cand_l.sequence).encode(), dtype=np.uint8).reshape(-1, length)
        q_ep = cand_l.epoch.to_numpy()
        for i in range(0, len(Q), 256):
            D = (Q[i:i + 256, None, :] != R[None, :, :]).sum(-1)
            mask = (D >= 1) & (D <= max_edits)
            if future_only:
                mask &= r_ep[None, :] > q_ep[i:i + 256, None]
            cnt = mask.sum(1)
            mean = np.where(cnt > 0, (mask * r_pct[None, :]).sum(1) / np.maximum(cnt, 1), np.nan)
            out.loc[cand_l.index[i:i + 256]] = mean
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(x).rank().to_numpy()
    y = pd.Series(y).rank().to_numpy()
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def within_epoch_spearman(df: pd.DataFrame, col: str) -> float:
    """n-weighted mean of per-epoch Spearman (epochs with >= 4 candidates)."""
    num = den = 0.0
    for _, g in df.groupby("epoch"):
        if len(g) < 4:
            continue
        r = spearman(g[col].to_numpy(), g.outcome.to_numpy())
        if np.isfinite(r):
            num += r * len(g)
            den += len(g)
    return num / den if den else float("nan")


@dataclass
class SliceResult:
    name: str
    n: int
    n_epochs: int
    rho_inc: float
    rho_pro: float
    diff: float
    ci_inc: tuple[float, float]
    ci_pro: tuple[float, float]
    ci_diff: tuple[float, float]
    within_inc: float = float("nan")
    within_pro: float = float("nan")
    verdict: str = ""
    notes: list[str] = dc_field(default_factory=list)


def bootstrap(df: pd.DataFrame, unit: str | None, B: int, seed: int):
    rng = np.random.default_rng(seed)
    x1, x2, y = df.incumbent.to_numpy(), df.proposed.to_numpy(), df.outcome.to_numpy()
    if unit:
        groups = [np.flatnonzero(df[unit].to_numpy() == u) for u in df[unit].unique()]
    else:
        groups = [np.array([i]) for i in range(len(df))]
    r1, r2 = [], []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        r1.append(spearman(x1[idx], y[idx]))
        r2.append(spearman(x2[idx], y[idx]))
    r1, r2 = np.array(r1), np.array(r2)
    d = r2 - r1
    ok = np.isfinite(d)
    q = lambda a: tuple(float(v) for v in np.nanpercentile(a, [2.5, 97.5]))
    return q(r1[np.isfinite(r1)]), q(r2[np.isfinite(r2)]), q(d[ok])


def evaluate_slice(name: str, df: pd.DataFrame, mode: str, B: int, seed: int,
                   min_n: int) -> SliceResult:
    df = df.dropna(subset=["incumbent", "proposed", "outcome"])
    n, n_ep = len(df), df.epoch.nunique()
    if n < min_n:
        return SliceResult(name, n, n_ep, *(float("nan"),) * 3, *((float("nan"),) * 2,) * 3,
                           verdict="INSUFFICIENT", notes=[f"N={n} < --min-n {min_n}"])
    unit = "epoch" if (mode == "own" and n_ep >= 8) else None
    ci_i, ci_p, ci_d = bootstrap(df, unit, B, seed)
    r_i = spearman(df.incumbent.to_numpy(), df.outcome.to_numpy())
    r_p = spearman(df.proposed.to_numpy(), df.outcome.to_numpy())
    res = SliceResult(name, n, n_ep, r_i, r_p, r_p - r_i, ci_i, ci_p, ci_d)
    if mode == "own":
        res.within_inc = within_epoch_spearman(df, "incumbent")
        res.within_pro = within_epoch_spearman(df, "proposed")
        if unit is None:
            res.notes.append(f"only {n_ep} epochs: bootstrap resamples candidates, "
                             "which understates uncertainty (candidates share fields)")
    else:
        res.notes.append("relatives are shared between candidates, so the "
                         "candidate bootstrap is optimistic: treat a CI that barely "
                         "excludes zero as NOT significant")
    if max(abs(r_i), abs(r_p)) < 0.2:
        res.notes.append("both formulas explain little of the outcome here (|rho| < 0.2): "
                         "neither is a strong predictor, so small differences are fragile")
    if ci_d[0] > 0:
        res.verdict = "BETTER"
    elif ci_d[1] < 0:
        res.verdict = "WORSE"
    else:
        res.verdict = "NO_DIFFERENCE"
    return res


def overall_verdict(results: list[SliceResult]) -> tuple[int, str, list[str]]:
    reasons: list[str] = []
    primary = results[0]
    if primary.verdict == "INSUFFICIENT":
        return EXIT_INSUFFICIENT, "INSUFFICIENT DATA", [f"primary slice '{primary.name}': {primary.notes[0]}"]
    worse = [r for r in results if r.verdict == "WORSE"]
    if worse:
        reasons += [f"slice '{r.name}': proposed rho {r.rho_pro:.3f} vs incumbent "
                    f"{r.rho_inc:.3f}, diff 95% CI ({r.ci_diff[0]:+.3f}, {r.ci_diff[1]:+.3f}) "
                    "entirely below zero" for r in worse]
        return EXIT_WORSE, "PROPOSED IS WORSE", reasons
    valid = [r for r in results if r.verdict != "INSUFFICIENT"]
    signs = {np.sign(r.diff) for r in valid if np.isfinite(r.diff) and r.diff != 0}
    if primary.verdict == "BETTER" and len(signs) <= 1:
        reasons.append(f"primary slice diff 95% CI ({primary.ci_diff[0]:+.3f}, "
                       f"{primary.ci_diff[1]:+.3f}) entirely above zero; same sign in every slice")
        return EXIT_BETTER, "PROPOSED IS BETTER", reasons
    if primary.verdict == "BETTER":
        reasons.append("primary slice favours the proposal but the sign FLIPS in other slices")
    else:
        reasons.append(f"primary slice diff 95% CI ({primary.ci_diff[0]:+.3f}, "
                       f"{primary.ci_diff[1]:+.3f}) contains zero")
    if len(signs) > 1:
        reasons.append("the point-estimate sign flips between slices -> the difference is noise")
    return EXIT_NO_EVIDENCE, "NO EVIDENCE OF IMPROVEMENT", reasons


MEANING = {
    EXIT_BETTER: ("The proposal ranks candidates measurably better than the incumbent on "
                  "real validator outcomes. This is the ONLY verdict that supports deploying "
                  "it. Still deploy dev/one-box first and watch the next epochs."),
    EXIT_NO_EVIDENCE: ("Do NOT deploy on this basis. 'No significant difference' means the data "
                       "cannot tell the two apart -- it is NOT evidence the proposal is safe, "
                       "equivalent, or better. Keep the incumbent."),
    EXIT_WORSE: ("Do NOT deploy. The proposal ranks candidates measurably WORSE than the "
                 "incumbent on real validator outcomes. If it is already live, revert it."),
    EXIT_INSUFFICIENT: ("Do NOT deploy. There is not enough real outcome data to judge. Widen "
                        "the epoch window or wait for more epochs; do not substitute intuition."),
}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if np.all(np.isfinite(ci)) else "[   n/a   ]"


def report(args, results: list[SliceResult], code: int, title: str, reasons: list[str],
           window: list[int]) -> str:
    bar = "=" * 78
    lines = [bar, "SCORING-CHANGE VALIDATION  (ground truth = published validator outcomes)", bar,
             f"incumbent : {args.incumbent}",
             f"proposed  : {args.proposed}",
             f"outcome   : {args.outcome}"
             + (f" (mean within-epoch percentile of published relatives; radii "
                f"{args.max_edits} edits; relatives={args.relatives})"
                if args.outcome == "neighbors" else " (own published final_score)"),
             f"epochs    : {window[0]}-{window[-1]} ({len(window)} published)"
             + (f", coldkey {args.coldkey[:8]}..." if args.outcome == "own" and args.coldkey != "all" else ""),
             f"statistic : Spearman rho vs outcome (lower score & lower outcome = better, so "
             "positive rho = good); paired bootstrap B=" + str(args.bootstrap), ""]
    hdr = f"{'slice':<28}{'N':>6}{'ep':>5}  {'rho incumbent':>22}  {'rho proposed':>22}  {'proposed-incumbent':>26}  verdict"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in results:
        lines.append(
            f"{r.name:<28}{r.n:>6}{r.n_epochs:>5}  "
            f"{r.rho_inc:>6.3f} {fmt_ci(r.ci_inc):>15}  {r.rho_pro:>6.3f} {fmt_ci(r.ci_pro):>15}  "
            f"{r.diff:>+7.3f} {fmt_ci(r.ci_diff):>18}  {r.verdict}")
    if args.outcome == "own":
        lines.append("")
        lines.append("within-epoch (stratified) Spearman, n-weighted over epochs with >= 4 candidates:")
        for r in results:
            lines.append(f"  {r.name:<20} incumbent {r.within_inc:6.3f}   proposed {r.within_pro:6.3f}")
    notes = sorted({n for r in results for n in r.notes})
    if notes:
        lines.append("")
        lines += [f"note: {n}" for n in notes]
    lines += ["", bar, f"VERDICT: {title}   (exit code {code})", bar]
    lines += [f"  - {r}" for r in reasons]
    lines += ["", "WHAT THIS MEANS: " + MEANING[code], bar]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Known-answer cases
# --------------------------------------------------------------------------

CASES = {
    "formula-20261009": dict(
        outcome="own", last=10, epochs=None, end_epoch=25613, coldkey=OUR_COLDKEY,
        incumbent="builtin:sum10", proposed="builtin:category3", slices=[],
        expect_exit=EXIT_WORSE, expect_rho=((0.999, 1.0), (0.905, 0.930)), expect_n=126,
        expect="incumbent rho = 1.000 (sum-of-ten IS the validator formula); proposed "
               "rho ~0.92; verdict PROPOSED IS WORSE (exit 2)",
    ),
    "parent20x-20261009": dict(
        outcome="neighbors", max_edits="1,2,3", relatives="both", epochs="25000-25615",
        incumbent="builtin:published",
        proposed=f"csv:{FIXTURES / 'hf20x_algo2_20261009.csv.gz'}:hf20x",
        expect_exit=EXIT_NO_EVIDENCE, expect_rho=None, expect_n=None,
        slices=["score100plus=incumbent>=100",
                "band100to140=incumbent>=100 and incumbent<=140"],
        expect="every rho between about -0.02 and 0.15; every diff CI contains zero; "
               "the sign of proposed-incumbent flips between slices (e.g. d<=2/all/all "
               "0.040 vs 0.069, d<=3/all/band100to140 0.147 vs 0.065); verdict NO "
               "EVIDENCE OF IMPROVEMENT (exit 1)",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=sorted(CASES), help="run a known-answer validation case")
    p.add_argument("--incumbent", help="score source currently live (see SCORE SOURCES)")
    p.add_argument("--proposed", help="score source you want to deploy")
    p.add_argument("--outcome", choices=["own", "neighbors"], default="own")
    p.add_argument("--epochs", help="explicit window, e.g. 25600-25616 or 25601,25605")
    p.add_argument("--last", type=int, default=10, help="newest N published epochs (default 10)")
    p.add_argument("--end-epoch", type=int, help="with --last: count back from this epoch")
    p.add_argument("--coldkey", default=OUR_COLDKEY,
                   help="own mode: whose submissions are candidates ('all' = whole field)")
    p.add_argument("--max-edits", default="2",
                   help="neighbors mode: Hamming radius, or a list '1,2,3' to backtest each "
                        "radius as its own slice (the first is the primary slice)")
    p.add_argument("--relatives", choices=["all", "future", "both"], default="all",
                   help="neighbors mode: all published relatives, only those published "
                        "AFTER the candidate (no look-ahead), or both as separate slices")
    p.add_argument("--future-only", dest="relatives", action="store_const", const="future",
                   help="shorthand for --relatives future")
    p.add_argument("--slice", action="append", default=[], metavar="NAME=QUERY",
                   help="extra slice as a pandas query over columns incumbent, proposed, "
                        "final_score, published_pct, epoch (repeatable). The first, "
                        "unfiltered 'all' slice is the primary one.")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--refresh", action="store_true", help="re-download cached epochs")
    p.add_argument("--json-out", type=Path, help="also write the result as JSON")
    p.add_argument("--dump-candidates", type=Path, help="write the scored candidate table (CSV)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expect = None
    if args.case:
        case = dict(CASES[args.case])
        expect = case.pop("expect")
        checks = {k: case.pop(k) for k in ("expect_exit", "expect_rho", "expect_n")}
        for k, v in case.items():
            if k == "slices":
                args.slice = list(v) + args.slice
            else:
                setattr(args, k, v)
    if not args.incumbent or not args.proposed:
        print("need --incumbent and --proposed (or --case)", file=sys.stderr)
        return EXIT_INSUFFICIENT

    # 1. ground truth to disk, then parse from disk
    if args.epochs:
        window = parse_epochs(args.epochs)
        paths = fetch_epochs(window, args.cache, args.refresh)
    else:
        args.cache.mkdir(parents=True, exist_ok=True)
        window = latest_epochs(args.last, args.cache, args.refresh, args.end_epoch)
        paths = fetch_epochs(window, args.cache, False)
    lb = load_field(paths)
    if lb.empty:
        print("no leaderboard data in window", file=sys.stderr)
        return EXIT_INSUFFICIENT
    window = sorted(lb.epoch.unique().tolist())
    print(f"leaderboard: {len(lb)} published nanobodies over {len(window)} epochs "
          f"({window[0]}-{window[-1]}), cached in {args.cache}", file=sys.stderr)

    # 2. score every published row with both formulas (over the full field)
    inc_src, pro_src = parse_source(args.incumbent), parse_source(args.proposed)
    lb["incumbent"] = apply_source(inc_src, lb)
    lb["proposed"] = apply_source(pro_src, lb)

    # 3. candidates + outcome(s); 4. slices (the first slice is the primary one)
    user_slices = []
    for spec in args.slice:
        name, sep, query = spec.partition("=")
        if not sep or not name or not query:
            raise SystemExit(f"--slice must be NAME=QUERY (NAME without '='): {spec!r}")
        user_slices.append((name, query))
    slices: list[tuple[str, pd.DataFrame]] = []
    if args.outcome == "own":
        cand = lb if args.coldkey == "all" else lb[lb.coldkey == args.coldkey]
        cand = cand.copy()
        cand["outcome"] = cand.final_score
        slices.append(("all", cand))
        slices += [(name, cand.query(q)) for name, q in user_slices]
    else:
        base = (lb.sort_values("epoch").drop_duplicates("sequence", keep="first")
                .dropna(subset=["incumbent", "proposed"]).copy())
        radii = [int(x) for x in str(args.max_edits).split(",") if x.strip()]
        rels = {"all": ["all"], "future": ["future"], "both": ["all", "future"]}[args.relatives]
        frames = []
        for radius in radii:
            for rel in rels:
                cand = base.copy()
                cand["outcome"] = neighbor_outcomes(lb, cand, radius, rel == "future")
                cand["radius"], cand["relatives"] = radius, rel
                frames.append(cand)
                tag = f"d<={radius}/{rel}"
                slices.append((f"{tag}/all", cand))
                slices += [(f"{tag}/{name}", cand.query(q)) for name, q in user_slices]
        cand = pd.concat(frames, ignore_index=True)
    results = [evaluate_slice(n, d, args.outcome, args.bootstrap, args.seed, args.min_n)
               for n, d in slices]
    code, title, reasons = overall_verdict(results)

    text = report(args, results, code, title, reasons, window)
    if expect:
        problems = []
        if code != checks["expect_exit"]:
            problems.append(f"exit {code} != expected {checks['expect_exit']}")
        if checks["expect_n"] is not None and results[0].n != checks["expect_n"]:
            problems.append(f"N {results[0].n} != expected {checks['expect_n']}")
        if checks["expect_rho"] is not None:
            for label, val, (lo, hi) in zip(("incumbent", "proposed"),
                                            (results[0].rho_inc, results[0].rho_pro),
                                            checks["expect_rho"]):
                if not lo <= val <= hi:
                    problems.append(f"{label} rho {val:.3f} outside [{lo}, {hi}]")
        text += (f"\nKNOWN-ANSWER CASE '{args.case}': expected -> {expect}\n"
                 + ("SELF-CHECK PASSED: the tool reproduces the known result.\n" if not problems
                    else "SELF-CHECK FAILED: " + "; ".join(problems)
                    + "\n  -> the tool or the data source changed; do not trust its verdicts "
                      "until this is understood.\n"))
    print(text)

    if args.dump_candidates:
        cand.to_csv(args.dump_candidates, index=False)
    if args.json_out:
        args.json_out.write_text(json.dumps({
            "incumbent": args.incumbent, "proposed": args.proposed, "outcome": args.outcome,
            "epochs": [window[0], window[-1], len(window)], "exit_code": code,
            "verdict": title, "deploy_supported": code == EXIT_BETTER, "reasons": reasons,
            "meaning": MEANING[code],
            "slices": [r.__dict__ for r in results],
        }, indent=2, default=float))
    return code


if __name__ == "__main__":
    sys.exit(main())
