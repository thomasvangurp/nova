#!/usr/bin/env python3
"""Multi-pass score EVERY rival's submissions and aggregate by coldkey.

The dashboard's "multi-pass vs rival" panel wants, per epoch, one entry per
rival with the fleet's mean rank-sum, its standard deviation, the fleet size
(number of sequences submitted), and the per-sequence rank-sums (distribution).

The leaderboard API exposes ``coldkey`` per submission (the stable identity --
the hotkey rotates each epoch). We group P20809 nanobodies by coldkey, run the
fast scorer N times over every unique sequence, compute the validator's dense
rank-sum vs the archive field each pass, and reduce to mean/stdev/fleet_size.

Runs on the box (scorer + archive + config live there). One scoring run covers
all rivals, not one per rival.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Corrected validator final_score (3-category-rank sum) lives in the package;
# this script runs from the box's repo root, so make the package importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt.simulate_round import validator_final_scores  # noqa: E402

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"
OUR_COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"

# Validator's Algorithm-2 (rank_binders): minmax over weighted metric ranks.
PROTEIN_WEIGHTS = {
    "design_iiptm": 1.0, "design_ptm": 2.0, "min_design_to_target_pae": 1.0,
    "plip_hbonds_refolded": 2.0, "plip_saltbridge_refolded": 2.0,
    "delta_sasa_refolded": 2.0,
}
METRIC_DEFS = {
    "design_iiptm": "max", "design_ptm": "max", "min_design_to_target_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max",
}
# All ten validator metrics. The corrected 3-category final_score needs
# interaction_pae / design_to_target_iptm / liability_score that the old
# minmax proxy never ranked. The fast scorer and the historical archive may
# both omit interaction_pae, so score_pass falls back to the minmax proxy when
# the full set is not present.
METRICS_NEEDED = sorted({
    "design_iiptm", "design_ptm", "design_to_target_iptm",
    "min_design_to_target_pae", "interaction_pae",
    "plip_hbonds_refolded", "plip_saltbridge_refolded",
    "delta_sasa_refolded", "liability_score", "liability_num_violations",
})


def fetch_leaderboard(epoch: int) -> list[tuple[str, str]]:
    """Return (coldkey, sequence) for every P20809 nanobody, deduped.

    A not-yet-scored (in-progress) epoch returns HTTP 404 from the API; treat
    that as "no data" rather than crashing so the quiet-moment loop can skip
    back to the most recent completed epoch.
    """
    req = urllib.request.Request(
        f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
        headers={"User-Agent": "nova-multipass-all-rivals/1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except Exception:
        return []
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for submission in payload.get("data") or []:
        coldkey = str(submission.get("coldkey") or "").strip()
        if not coldkey:
            continue
        for nb in submission.get("nanobodies") or []:
            if nb.get("protein_name") != TARGET:
                continue
            seq = str(nb.get("sequence") or "").strip().upper()
            if not seq or seq in seen:
                continue
            seen.add(seq)
            pairs.append((coldkey, seq))
    return pairs


def algo2_score(field: pd.DataFrame) -> pd.Series:
    """Validator Algorithm-2 minmax over weighted metric ranks (lower = better).

    Mirrors utils/minmax_weighted_rank.rank_binders: drop sequences with >50
    liability violations, then score = max over the 6 weighted metrics of
    (average rank / weight).
    """
    work = field[field["liability_num_violations"] <= 50].copy()
    rank_cols = {}
    for metric, weight in PROTEIN_WEIGHTS.items():
        if metric not in work.columns:
            continue
        ascending = METRIC_DEFS[metric] == "min"
        rank_cols[metric] = work[metric].rank(
            ascending=ascending, method="average"
        ) / weight
    return pd.DataFrame(rank_cols, index=work.index).max(axis=1)


def score_pass(sequences: list[str], run: int, *, archive: pd.DataFrame,
               config: Path, scorer: Path, out_root: Path
               ) -> tuple[dict[str, float], dict[str, dict], dict[str, dict[str, float]]]:
    """One fast-scorer pass -> (rank_sum, cif_ref, raw_metrics) per sequence."""
    inp = out_root / f"input_{run}.json"
    inp.write_text(json.dumps({"sequences": [{"sequence": s} for s in sequences]}))
    out_dir = out_root / f"score{run}"
    os.environ.setdefault(
        "PYTHONPATH", "/root/bgen_shim:/root/external_tools/boltzgen/src"
    )
    # The dashboard box is a 2x4090; the old 8x master ran this with all 8.
    # MULTIPASS_GPUS lets the host override without editing the script.
    gpus = os.environ.get("MULTIPASS_GPUS", "0,1")
    subprocess.run(
        [sys.executable, "-m", "nanobody_opt.score_boltzgen_waves",
         "--sequences", str(inp), "--config", str(config),
         "--target", TARGET, "--output-dir", str(out_dir),
         "--gpus", gpus, "--wave-size", str(len(sequences)),
         "--predict-batch-size", "2", "--limit", str(len(sequences)),
         "--scorer", str(scorer), "--capture-structures"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    payload = json.loads((out_dir / "exact_all.json").read_text())
    wave = payload["per_nanobody_components"]["0"]
    rows = {}
    for seq, comp in wave.items():
        m = comp.get(TARGET, {})
        rows[seq] = {k: float(v) for k, v in m.items()
                     if k in METRICS_NEEDED and isinstance(v, (int, float))}
    rival = pd.DataFrame(rows).T  # index = sequence
    corrected = (set(METRICS_NEEDED) <= set(archive.columns)
                 and set(METRICS_NEEDED) <= set(rival.columns))
    use_metrics = METRICS_NEEDED if corrected else sorted(
        set(PROTEIN_WEIGHTS) | {"liability_num_violations"}
    )
    archive_s = archive.set_index("sequence")[use_metrics]
    field = pd.concat([archive_s, rival])  # index = sequence
    # A rival sequence that is also in the archive appears twice; the rankers
    # then return a multi-row Series for it and float(scores[seq]) blows up.
    # Keep the rival's (last) row so the fresh multipass score wins the overlap.
    field = field[~field.index.duplicated(keep="last")]
    if corrected:
        scores = pd.Series(validator_final_scores(field), index=field.index)
    else:
        scores = algo2_score(field)  # index = sequence (liability-filtered)
    ranks = {seq: float(scores[seq]) for seq in rival.index if seq in scores.index}
    # Preserve the captured .cif reference per sequence (--capture-structures),
    # so it can be persisted to the mutation DB's observation_artifacts.
    structures: dict[str, dict] = {}
    for seq, by_target in (payload.get("structures") or {}).items():
        rec = by_target.get(TARGET, {})
        if rec:
            structures[seq] = {
                "path": rec.get("path"),
                "writer_path": rec.get("writer_path"),
                "sample_selection": rec.get("sample_selection"),
            }
    return ranks, structures, rows


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    epoch = int(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    archive_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
        f"/root/live/epoch{epoch}/producer/archive.csv")
    config = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(
        f"/root/live/epoch{epoch}/producer/config.yaml")
    # A stray epoch (25579) landed archive.csv/config.yaml at the epoch root
    # instead of producer/. Fall back so one odd layout doesn't block multipass.
    _alt_archive = archive_path.parent.parent / archive_path.name
    if not archive_path.exists() and _alt_archive.exists():
        archive_path = _alt_archive
    _alt_config = config.parent.parent / config.name
    if not config.exists() and _alt_config.exists():
        config = _alt_config
    scorer = Path(sys.argv[5]) if len(sys.argv) > 5 else Path(
        "/root/nanobody_opt/score_boltzgen_fast_adapter.py")
    out_root = Path(sys.argv[6]) if len(sys.argv) > 6 else Path(
        f"/root/multipass_all_rivals/{epoch}")
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = fetch_leaderboard(epoch)
    by_coldkey: dict[str, list[str]] = defaultdict(list)
    for coldkey, seq in pairs:
        by_coldkey[coldkey].append(seq)
    if not by_coldkey:
        print(f"NO_RIVALS epoch={epoch}", flush=True)
        return 1

    sequences = sorted({seq for _, seq in pairs})
    archive = pd.read_csv(archive_path).drop_duplicates("sequence", keep="last")
    archive["sequence"] = archive["sequence"].astype(str).str.upper()
    print(f"epoch={epoch} coldkeys={len(by_coldkey)} seqs={len(sequences)} "
          f"passes={n}", flush=True)

    per_pass: list[dict[str, float]] = []
    per_pass_structures: list[dict[str, dict]] = []
    per_pass_metrics: list[dict[str, dict[str, float]]] = []
    for run in range(1, n + 1):
        ranks, structs, metrics = score_pass(sequences, run, archive=archive,
                                             config=config, scorer=scorer,
                                             out_root=out_root)
        per_pass.append(ranks)
        per_pass_structures.append(structs)
        per_pass_metrics.append(metrics)
        print(f"  pass {run}/{n} done", flush=True)

    # A sequence can be rejected by the scorer in some passes (liability/fold
    # failure), so only aggregate over sequences present in EVERY pass.
    common = sorted(set.intersection(*(set(p) for p in per_pass)))
    seq_means = {s: float(np.mean([p[s] for p in per_pass])) for s in common}

    # Persist the multiscore into the mutation DB: one row per pass per
    # sequence, so _live_top50_from_db's COUNT(*)/AVG(score) sees public
    # sequences with a stable multi-pass rank-sum (high-fidelity parents).
    # The captured .cif reference rides along via the `structure` field, which
    # update_observation_store persists to observation_artifacts.
    try:
        from nanobody_opt.mutation_network import update_observation_store
        observations = []
        for run, (ranks, structs, metrics) in enumerate(
            zip(per_pass, per_pass_structures, per_pass_metrics), start=1
        ):
            for seq, score in ranks.items():
                observations.append({
                    "observation_id": f"public_multipass:epoch{epoch}:{seq}:pass{run}",
                    "sequence": seq,
                    "target": TARGET,
                    "epoch": epoch,
                    "source": "public_multipass_rivals",
                    "score_kind": "local_validator_replay_rank_sum",
                    "score": float(score),
                    "observation_count": 1,
                    "metrics": metrics.get(seq),
                    "structure": structs.get(seq),
                })
        written = update_observation_store(
            Path("/root/artifacts/nanobody_mutations.sqlite"), observations)
        print(f"DB_WRITE observations={written} sequences={len(seq_means)}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"DB_WRITE_FAILED {type(exc).__name__}: {exc}", flush=True)

    # Preserve the sequence -> mean rank-sum mapping (do NOT drop it) in the
    # artifact alongside the per-coldkey rollup.
    result: dict[str, dict] = {
        "ours": None,
        "rivals": {},
        "sequences": {s: float(seq_means[s]) for s in common},
    }
    for coldkey, seqs in sorted(by_coldkey.items(), key=lambda kv: -len(kv[1])):
        scored = [seq_means[s] for s in seqs if s in seq_means]
        entry = {
            "mean": float(np.mean(scored)) if scored else None,
            "stdev": float(np.std(scored)) if len(scored) > 1 else None,
            "fleet_size": len(seqs),
            "scores": [float(v) for v in sorted(scored)],
        }
        if coldkey == OUR_COLDKEY:
            result["ours"] = entry
        else:
            result["rivals"][coldkey] = entry

    out_json = out_root / f"multipass_{epoch}.json"
    out_json.write_text(json.dumps({str(epoch): result}, indent=2) + "\n")
    print(f"WROTE {out_json}", flush=True)
    print(f"  ours: {result['ours']}", flush=True)
    for ck, e in result["rivals"].items():
        stdev = e.get("stdev")
        stdev_s = f"{stdev:.1f}" if stdev is not None else "n/a"
        print(f"  {ck[:12]}...: mean={e['mean']:.1f} stdev={stdev_s} "
              f"fleet={e['fleet_size']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
