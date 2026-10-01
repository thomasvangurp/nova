"""Assemble (parent, child, rank_sum) dataset from partial children scoring.

Reads local synced box results + prepare_children.json + archive.csv.
Produces a DataFrame with the columns train_parent_relative_ranker.feature_matrix
consumes, plus parent_rank_sum / child_rank_sum / target_improvement.
Read-only: writes a single gzipped CSV into --output-dir.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/Users/tvgdeena/nova-nanobody")
from nanobody_opt.simulate_round import METRICS, rank_sum_in_field  # noqa: E402

AMINO = "ACDEFGHIKLMNPQRSTVWY"
HOT = {0: "QR", 2: "TH", 113: "HF", 115: "AS", 120: "TS", 121: "ST", 123: "TP"}
METRIC_ORDER = list(METRICS.keys())


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def load_archive(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates("sequence", keep="last").reset_index(drop=True)
    for m in METRIC_ORDER:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    reference = df.tail(min(300, len(df))).reset_index(drop=True)
    return df, reference


def load_elite(path: Path) -> tuple[dict[int, set[str]], list[str]]:
    prep = json.loads(path.read_text())
    elite = [str(s).strip().upper() for s in prep["elite_parents"]]
    by_len: dict[int, set[str]] = {}
    for s in elite:
        by_len.setdefault(len(s), set()).add(s)
    return by_len, elite


def derive_parent(child: str, by_len: dict[int, set[str]], elite_rank: dict[str, int]) -> tuple[str | None, int | None, str | None]:
    """Best elite parent at Hamming distance 1; return (parent, pos, orig)."""
    L = len(child)
    pool = by_len.get(L, set())
    if not pool:
        return None, None, None
    cands = []
    for i, ch in enumerate(child):
        for aa in AMINO:
            if aa == ch:
                continue
            cand = child[:i] + aa + child[i + 1:]
            if cand in pool:
                cands.append((i, aa, cand))
    if not cands:
        return None, None, None
    if len(cands) == 1:
        pos, orig, parent = cands[0]
        return parent, pos, orig
    # prefer HOT-consistent
    hot = [(pos, orig, parent) for pos, orig, parent in cands
           if pos in HOT and child[pos] in HOT[pos]]
    if len(hot) == 1:
        pos, orig, parent = hot[0]
        return parent, pos, orig
    if hot:
        cands = hot
    # deterministic: earliest (best-score) elite parent
    cands.sort(key=lambda t: elite_rank[t[2]])
    pos, orig, parent = cands[0]
    return parent, pos, orig


def valid_pass_metrics(entry: dict) -> list[dict]:
    out = []
    for k, v in entry.items():
        if not (k.startswith("p") and k[1:].isdigit()):
            continue
        if not isinstance(v, dict) or not set(METRIC_ORDER).issubset(v):
            continue
        try:
            row = {m: float(v[m]) for m in METRIC_ORDER}
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(x) for x in row.values()):
            out.append(row)
    return out


def load_child_results(globs: list[str]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    seen_paths: list[str] = []
    for g in globs:
        for p in sorted(glob.glob(g)):
            seen_paths.append(p)
            d = json.loads(Path(p).read_text())
            entries = d.values() if isinstance(d, dict) else d
            for e in entries:
                seq = str(e["sequence"]).strip().upper()
                h = sha(seq)
                cur = merged.setdefault(h, {"sequence": seq})
                for k, v in e.items():
                    if k.startswith("p") and k not in cur:
                        cur[k] = v
    return merged, seen_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=Path("/Users/tvgdeena/nova-nanobody/scoring_backfill/archive.csv"))
    ap.add_argument("--prepare", type=Path, default=Path("/Users/tvgdeena/nova-nanobody/scoring_backfill/prepare_children.json"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--modal-length", type=int, default=127)
    args = ap.parse_args()

    archive, reference = load_archive(args.archive)
    by_len, elite = load_elite(args.prepare)
    elite_rank = {s: i for i, s in enumerate(elite)}

    result_globs = [
        str(Path("/Users/tvgdeena/nova-nanobody/scoring_backfill/children_merged_results.json")),
        str(Path("/Users/tvgdeena/nova-nanobody/scoring_backfill/results_children9/b*.json")),
        str(Path("/Users/tvgdeena/nova-nanobody/scoring_backfill/children_results9/b*.json")),
        str(Path("/Users/tvgdeena/nova-nanobody/scoring_backfill/children_results10/b*.json")),
    ]
    child_entries, seen_paths = load_child_results(result_globs)

    parent_metrics_by_seq = {r.sequence: r for _, r in archive.iterrows()}
    ref_field = reference[METRIC_ORDER].copy()

    records = []
    parent_confidence = Counter()
    for h, e in child_entries.items():
        child = e["sequence"]
        if len(child) != args.modal_length:
            continue
        passes = valid_pass_metrics(e)
        if not passes:
            continue
        parent, pos, orig = derive_parent(child, by_len, elite_rank)
        if parent is None:
            parent_confidence["no_parent"] += 1
            continue
        n_cands_proxy = "unique"  # filled below via separate pass is not tracked here
        parent_confidence["resolved"] += 1
        prow = parent_metrics_by_seq.get(parent)
        if prow is None:
            parent_confidence["parent_missing_archive"] += 1
            continue
        try:
            parent_metrics = {m: float(prow[m]) for m in METRIC_ORDER}
        except (TypeError, ValueError):
            parent_confidence["parent_bad_metrics"] += 1
            continue
        if not all(math.isfinite(x) for x in parent_metrics.values()):
            parent_confidence["parent_nonfinite"] += 1
            continue
        child_metrics = {m: float(np.median([p[m] for p in passes])) for m in METRIC_ORDER}
        parent_rank_sum = float(rank_sum_in_field(parent_metrics, ref_field))
        child_rank_sum = float(rank_sum_in_field(child_metrics, ref_field))
        record = {
            "sequence": child,
            "parent_sequence": parent,
            "edit_position": pos,
            "edit_from": orig,
            "edit_to": child[pos] if pos is not None else "",
            "epoch": int(e.get("epoch") or 0),
            "observation_count": len(passes),
            "proposal_type": "exhaustive_single_substitution",
            "edit_distance": 1.0,
            "mutations": f"{orig}{pos+1}{child[pos]}" if pos is not None else "",
            "proposal_prior_score": 0.0,
            "parent_selection_score": elite_rank.get(parent, -1),
            "screen_order": 0,
            "parent_rank_sum": parent_rank_sum,
            "child_rank_sum": child_rank_sum,
            "target_improvement": parent_rank_sum - child_rank_sum,
        }
        record.update({f"parent_{m}": parent_metrics[m] for m in METRIC_ORDER})
        record.update({f"child_{m}": child_metrics[m] for m in METRIC_ORDER})
        records.append(record)

    frame = pd.DataFrame(records)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "children_parent_exact_dataset.csv.gz", index=False)
    meta = {
        "source_paths": seen_paths,
        "archive_rows": len(archive),
        "reference_rows": len(reference),
        "elite_parents": len(elite),
        "child_result_entries": len(child_entries),
        "modal_length": args.modal_length,
        "records": len(frame),
        "observation_count_distribution": dict(sorted(Counter(frame.observation_count).items())),
        "unique_children": int(frame.sequence.nunique()),
        "unique_parents": int(frame.parent_sequence.nunique()),
        "parent_confidence": dict(parent_confidence),
        "target_improvement_stats": {
            "mean": float(frame.target_improvement.mean()),
            "median": float(frame.target_improvement.median()),
            "std": float(frame.target_improvement.std()),
            "min": float(frame.target_improvement.min()),
            "max": float(frame.target_improvement.max()),
        },
    }
    (out / "assembly_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps({
        "records": len(frame),
        "unique_children": int(frame.sequence.nunique()),
        "unique_parents": int(frame.parent_sequence.nunique()),
        "obs_dist": dict(sorted(Counter(frame.observation_count).items())),
        "parent_confidence": dict(parent_confidence),
        "target_improvement_mean": float(frame.target_improvement.mean()),
        "target_improvement_median": float(frame.target_improvement.median()),
        "children_per_parent_median": float(frame.groupby('parent_sequence').size().median()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
