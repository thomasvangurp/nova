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

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

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
# The validator's actual score is the SUM of the ten metric dense ranks
# (rho=1.000 vs the published final_score), NOT the Algorithm-2 minmax above
# (~0.56). These ten metrics and directions drive sum_of_ten().
SUM_OF_TEN_DIRS = {
    "design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min", "interaction_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max", "liability_score": "min",
    "liability_num_violations": "min",
}
METRICS_NEEDED = sorted(SUM_OF_TEN_DIRS)

# Label for the per-pass rows this script writes (ARCHITECTURE_REDESIGN P2/P4).
# Before 2026-10-09 they went out as ``local_validator_replay_rank_sum`` -- the
# producer's ~200-scale label -- on a ~2e4-1.8e5 scale with empty metrics, so
# every generic replay reader pooled two incompatible scales and a formula
# change could only be repaired by re-folding on a GPU. Each row now carries:
#   score        sum-of-10 dense rank-sum vs the field recorded in provenance
#   metrics_json the raw per-pass BoltzGen metrics (formula-independent), so
#                any later formula/field can be recomputed with no GPU
#   provenance   formula, field rows + sha256, pass index/count, writer, cohort
PASS_SCORE_KIND = "multipass_sum10_pass"
WRITER = "multipass_all_rivals"
# The formula behind PASS_SCORE_KIND. It is registered in
# nanobody_opt/observation_labels.py, and main() proves that pass_formula()
# still computes it (golden field) before any GPU work. To change the formula,
# add a NEW label to the registry and change PASS_SCORE_KIND and
# PASS_FORMULA_ID together. Never re-use a label: 2026-10-09 algo2 ->
# category3 -> sum10 all landed under one label (docs/OBSERVATION_LABELS.md).
PASS_FORMULA_ID = "sum10"
# What the dashboard JSON's per-epoch "method" says this run measured.
MEASUREMENT = "local_fast_scorer_multipass_mean"
#: provenance.cohort of the bounded HF top-up rows (= hf_topup.TOPUP_COHORT).
TOPUP_COHORT = "hf_top200"


def fetch_our_intended(epoch: int) -> list[tuple[str, str]]:
    """Return (OUR_COLDKEY, sequence) for every sequence we intended to submit.

    ``fetch_leaderboard`` only sees what the validator actually published --
    an epoch where the validator rejected most of our fleet (e.g. as a
    duplicate resubmission) leaves "ours" built from whichever few survived,
    even though we generated and committed a full fleet. A local file dropped
    by the Pi right after each epoch's archive step
    (``/root/our_intended_sequences/{epoch}.json``, one ``{"uid", "sequence"}``
    row per committed submission) carries the complete intent regardless of
    what the validator kept, so merge it in. Missing file -> no supplement,
    not an error, since most boxes never receive this drop.
    """
    path = Path(f"/root/our_intended_sequences/{epoch}.json")
    if not path.is_file():
        return []
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for row in rows if isinstance(rows, list) else []:
        seq = str((row or {}).get("sequence") or "").strip().upper()
        if not seq or seq in seen:
            continue
        seen.add(seq)
        pairs.append((OUR_COLDKEY, seq))
    return pairs


def fetch_leaderboard(epoch: int) -> list[tuple[str, str]] | None:
    """Return (coldkey, sequence) for every P20809 nanobody, deduped.

    Returns ``None`` when the leaderboard is NOT AVAILABLE -- HTTP 404 for a
    not-yet-published epoch (the validator's publish lag is not bound to the
    72-minute epoch cadence; it has ranged from under an hour to several
    hours), any other HTTP/network error, or a payload without a ``data``
    list. ``None`` means "retry later", which is different from ``[]`` (the
    leaderboard was published but holds no P20809 nanobodies). Callers must
    not collapse the two: see ``main``.
    """
    req = urllib.request.Request(
        f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
        headers={"User-Agent": "nova-multipass-all-rivals/1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except Exception:
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for submission in data:
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


def sum_of_ten(field: pd.DataFrame) -> pd.Series:
    """Validator's actual score: sum of the ten metric dense ranks (lower=better).

    Empirically rho=1.000 against the published final_score over our own
    submissions, versus ~0.56 for the Algorithm-2 minmax above.
    """
    total = pd.Series(0.0, index=field.index)
    for metric, direction in SUM_OF_TEN_DIRS.items():
        if metric not in field.columns:
            continue
        total += field[metric].rank(method="dense", ascending=(direction == "min"))
    return total


def pass_formula(field: pd.DataFrame) -> pd.Series:
    """The formula every stored pass is ranked with (``PASS_FORMULA_ID``)."""
    return sum_of_ten(field)


def assert_pass_label() -> None:
    """Refuse to run if PASS_SCORE_KIND/PASS_FORMULA_ID/pass_formula disagree
    with the label registry (raises observation_labels.LabelViolation)."""
    here = Path(__file__).resolve().parent
    for root in (here, here.parent):  # box: /root; repo: tools/..
        if (root / "nanobody_opt").is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from nanobody_opt import observation_labels
    observation_labels.assert_formula(PASS_SCORE_KIND, PASS_FORMULA_ID, pass_formula)


def topup_settings() -> dict:
    """HF top-up knobs (ARCHITECTURE_REDESIGN R1), from the environment.

    MULTIPASS_TOPUP=0 turns it off; MULTIPASS_TOPUP_DEPTH (passes per HF
    candidate, default 10) and MULTIPASS_TOPUP_BUDGET (sequence-passes per
    run, default 500) bound it.
    """
    from nanobody_opt import hf_topup
    return {
        "enabled": os.environ.get("MULTIPASS_TOPUP", "1") != "0",
        "depth": int(os.environ.get("MULTIPASS_TOPUP_DEPTH", hf_topup.DEFAULT_DEPTH)),
        "budget": int(os.environ.get("MULTIPASS_TOPUP_BUDGET",
                                     hf_topup.DEFAULT_PASS_BUDGET)),
    }


def plan_hf_topup(archive: pd.DataFrame, batch: set[str], settings: dict
                  ) -> tuple[dict[str, int], dict[str, int]]:
    """``({sequence: passes_to_score}, {sequence: stored_passes})`` for this run.

    Candidates: the HF archive's rank_binders top-200 that are eligible
    parents (>= MIN_PARENT_DISTANCE from the validator's archive top-50, the
    dedup reference) and not in this epoch's leaderboard ``batch``. Each is
    topped up only by the passes it is missing. Never raises: the top-up is
    additive, so a failure here leaves the leaderboard run untouched.
    """
    if not settings.get("enabled"):
        return {}, {}
    try:
        from nanobody_opt import db, hf_topup
        top = hf_topup.hf_archive_top(archive, hf_topup.HF_TOP_K)
        candidates = hf_topup.eligible_parents(
            top, top[:hf_topup.VALIDATOR_TOP_N], hf_topup.MIN_PARENT_DISTANCE)
        with db.connect() as conn:
            existing = hf_topup.stored_pass_counts(conn, candidates)
        plan = hf_topup.plan_topup(candidates, existing, exclude=batch,
                                   depth=settings["depth"], budget=settings["budget"])
        done = sum(1 for s in candidates if existing.get(s, 0) >= settings["depth"])
        print(f"TOPUP_PLAN candidates={len(candidates)} at_depth={done} "
              f"this_run={len(plan)} sequence_passes={sum(plan.values())} "
              f"depth={settings['depth']} budget={settings['budget']}", flush=True)
        return plan, existing
    except BaseException as exc:  # noqa: BLE001 - incl. SystemExit from the loader
        print(f"TOPUP_PLAN_FAILED {type(exc).__name__}: {exc}", flush=True)
        try:
            from nanobody_opt.failure_log import record_failure
            record_failure(WRITER, "TOPUP_PLAN_FAILED", exc)
        except Exception:  # noqa: BLE001
            pass
        return {}, {}


def prepare_topup(archive: pd.DataFrame, batch: set[str]
                  ) -> tuple[dict, dict[str, int], dict[str, int], list[list[str]]]:
    """(settings, plan, stored pass counts, per-pass batches); never raises."""
    try:
        settings = topup_settings()
        from nanobody_opt import hf_topup
    except BaseException as exc:  # noqa: BLE001 - stale box tree without hf_topup
        print(f"TOPUP_PLAN_FAILED {type(exc).__name__}: {exc}", flush=True)
        return {"enabled": False}, {}, {}, []
    plan, existing = plan_hf_topup(archive, batch, settings)
    return settings, plan, existing, hf_topup.pass_batches(plan)


def raw_metrics_from_exact_all(payload: dict) -> dict[str, dict[str, float]]:
    """``{sequence: {metric: value}}`` for every finite numeric metric in one
    scorer pass's ``exact_all.json`` (wave 0, TARGET). Formula-independent."""
    wave = (payload.get("per_nanobody_components") or {}).get("0") or {}
    out: dict[str, dict[str, float]] = {}
    for seq, comp in wave.items():
        m = (comp or {}).get(TARGET, {}) or {}
        values = {k: float(v) for k, v in m.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)
                  and np.isfinite(float(v))}
        if values:
            out[str(seq).strip().upper()] = values
    return out


def field_fingerprint(archive_path: Path, archive: pd.DataFrame) -> dict:
    """What a stored rank-sum was ranked against, so it can be re-derived."""
    digest = hashlib.sha256()
    with open(archive_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"field": "hf_archive_full", "field_path": str(archive_path),
            "field_rows": int(len(archive)), "field_sha256": digest.hexdigest()}


#: How many of the most recently COMPLETED epochs (never the one being
#: scored) to pool into one real field. User, 2026-10-11: the static
#: hf_archive_full field (60k+ rows spanning the project's whole history)
#: correlates with the real validator's us-vs-rival outcome at only
#: rho=0.29/63% concordance (both us and the rival clear nearly all of a
#: field dominated by years of now-irrelevant early-generation sequences, so
#: real quality differences get diluted by density that neither of us is
#: actually competing against). A field pooled from the real last 10 epochs'
#: actual submissions (every coldkey, same raw metrics) measured rho=0.975 /
#: 89% concordance on 81 epochs of real leaderboard history -- field choice,
#: not local-scorer noise, explained nearly all of the gap.
RECENT_FIELD_EPOCHS = 10


def _recent_validator_field(epoch: int) -> tuple[pd.DataFrame, list[int]]:
    """Real field pooled from the last RECENT_FIELD_EPOCHS completed epochs.

    Reads nanobody_opt.mutation_network's already-synced cache
    (observations.source='public_validator_api', written continuously by
    seed_mutation_database.py from the same public leaderboard API this
    script already calls elsewhere) -- never a live HTTP call in this
    script's own hot path. NOVA_LOCAL_SQLITE unset here (this runs on the
    dashboard box, not master) routes to the central MySQL replica
    automatically; the path argument is only used when that env var is set.
    """
    from nanobody_opt.rank_exact import fetch_recent_fields_from_db, pool_recent_fields

    fields = fetch_recent_fields_from_db(
        Path("/root/artifacts/nanobody_mutations.sqlite"),
        target=TARGET, before_epoch=epoch, count=RECENT_FIELD_EPOCHS,
    )
    if not fields:
        return pd.DataFrame(columns=METRICS_NEEDED), []
    epochs = [e for e, _ in fields]
    pooled = pool_recent_fields(fields)
    return pooled, epochs


def recent_field_fingerprint(recent_field: pd.DataFrame, epochs: list[int]) -> dict:
    """What a stored rank-sum was ranked against, so it can be re-derived."""
    return {"field": "recent_validator_epochs", "field_epochs": epochs,
            "field_rows": int(len(recent_field))}


def pass_observations(*, epoch: int, run: int, passes: int,
                      ranks: dict[str, float], raw: dict[str, dict[str, float]],
                      structs: dict[str, dict], cohort_of: dict[str, str],
                      field: dict, extra_provenance: dict | None = None,
                      ) -> list[dict]:
    """One DB observation per scored sequence for one pass (see PASS_SCORE_KIND)."""
    rows = []
    for seq, score in ranks.items():
        row = {
            "observation_id": f"public_multipass:epoch{epoch}:{seq}:pass{run}",
            "sequence": seq,
            "target": TARGET,
            "epoch": epoch,
            # `source` stays the writer's population tag for continuity; the
            # cohort (ours vs rival) is explicit in provenance, because this
            # source has always included our own intended sequences.
            "source": "public_multipass_rivals",
            "score_kind": PASS_SCORE_KIND,
            "score": float(score),
            "observation_count": 1,
            "metrics": raw.get(seq) or {},
            "structure": structs.get(seq),
            # Every key below lands in provenance_json.
            "writer": WRITER,
            "formula": PASS_FORMULA_ID,
            "cohort": cohort_of.get(seq, "rival"),
            "pass_index": run,
            "passes": passes,
            **field,
            **(extra_provenance or {}),
        }
        rows.append(row)
    return rows


def score_pass(sequences: list[str], run: int, *, recent_field: pd.DataFrame,
               config: Path, scorer: Path, out_root: Path
               ) -> tuple[dict[str, float], dict[str, dict], dict[str, dict[str, float]]]:
    """One fast-scorer pass -> ({sequence: rank_sum}, {sequence: cif_ref},
    {sequence: raw metrics})."""
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
    return rank_pass_payload(payload, recent_field)


def rank_pass_payload(payload: dict, recent_field: pd.DataFrame
                      ) -> tuple[dict[str, float], dict[str, dict], dict[str, dict[str, float]]]:
    """Rank one pass's ``exact_all.json`` against the pooled recent-epoch
    field (sum-of-10). See RECENT_FIELD_EPOCHS / _recent_validator_field."""
    wave = payload["per_nanobody_components"]["0"]
    rows = {}
    for seq, comp in wave.items():
        m = comp.get(TARGET, {})
        rows[seq] = {k: float(v) for k, v in m.items()
                     if k in METRICS_NEEDED and isinstance(v, (int, float))}
    raw = raw_metrics_from_exact_all(payload)
    from nanobody_opt.rank_exact import rank_metrics_against_field
    ranks = rank_metrics_against_field(rows, recent_field)
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
    return ranks, structures, raw


#: Sources trusted for reusing OUR OWN stored observations in place of a
#: fresh GPU pass -- the producer's own exact-scoring pipeline, same set
#: build_top50_index.REPLAY_SCALE_SOURCES trusts (ARCHITECTURE_REDESIGN new
#: finding 9). Never rival data: a rival's sequence has no local observations
#: here by construction, so this set only ever matches our own sequences.
OUR_RAW_SOURCES = (
    "live_exact_screen", "live_exact_repeat", "local_exact_raw",
    "children_8x", "children_20x", "children_40x", "backfill_8x",
    "exhaustive_8x",
)


def local_passes_for_ours(
    sequences: list[str], n: int, recent_field: pd.DataFrame
) -> tuple[list[tuple[dict[str, float], dict[str, dict], dict[str, dict[str, float]]]],
           set[str]]:
    """Build up to ``n`` synthetic scorer passes for OUR OWN sequences from
    already-stored raw metrics, instead of spending fresh GPU time re-scoring
    something we already measured during the race itself.

    Every sequence here is ours (the caller only passes OUR_COLDKEY's
    sequences) -- a rival's sequence is never in our local observation store,
    so this can never accidentally reuse a rival's old score as if it were
    fresh. Returns (passes, short_sequences): ``passes`` has the SAME shape
    ``score_pass`` returns (one (ranks, structs, raw) tuple per pass, fed
    through the identical rank_pass_payload/pass_formula so the result is
    byte-for-byte what a real pass would have produced for the same raw
    metrics), and ``short_sequences`` are the ones with fewer than ``n``
    stored observations -- those get a real GPU pass instead, so a
    thin-evidence sequence is never silently degraded.
    """
    from nanobody_opt import db

    if not sequences:
        return [], set()
    conn = db.connect()
    placeholders = ",".join(["%s"] * len(sequences))
    source_placeholders = ",".join(["%s"] * len(OUR_RAW_SOURCES))
    rows = conn.execute(
        f"""
        SELECT n.sequence, o.metrics_json, o.observation_id
        FROM observations o JOIN nodes n ON n.id = o.node_id
        WHERE n.target = %s AND n.sequence IN ({placeholders})
          AND o.source IN ({source_placeholders})
          AND o.metrics_json IS NOT NULL AND o.metrics_json != ''
        ORDER BY o.observation_id
        """,
        (TARGET, *sequences, *OUR_RAW_SOURCES),
    ).fetchall()
    by_seq: dict[str, list[dict[str, float]]] = defaultdict(list)
    for seq, metrics_json, _obs_id in rows:
        try:
            m = json.loads(metrics_json)
        except (TypeError, ValueError):
            continue
        if all(k in m and isinstance(m[k], (int, float)) for k in METRICS_NEEDED):
            by_seq[seq].append({k: float(m[k]) for k in METRICS_NEEDED})

    short_sequences = {seq for seq in sequences if len(by_seq.get(seq, [])) < n}
    reusable = [seq for seq in sequences if seq not in short_sequences]
    if not reusable:
        return [], short_sequences

    passes: list[tuple[dict, dict, dict]] = []
    for run in range(n):
        wave = {
            seq: {TARGET: by_seq[seq][run]} for seq in reusable
        }
        payload = {"per_nanobody_components": {"0": wave}, "structures": {}}
        ranks, structs, raw = rank_pass_payload(payload, recent_field)
        passes.append((ranks, structs, raw))
    return passes, short_sequences


def _write_ours_only_early(
    epoch: int, our_pairs: list[tuple[str, str]], archive_path: Path, n: int,
    out_root: Path,
) -> None:
    """Write whatever of OUR OWN distribution is already known, straight from
    stored observations, with no GPU time and no dependency on the validator's
    leaderboard publish lag -- "we can backfill rivals when they arrive"
    (user, 2026-10-10). Used both as the pre-leaderboard fallback (``pairs``
    from ``fetch_our_intended``) and the early in-run write once rivals are
    known too (see the call in ``main``, which reuses this for the exact same
    reason: there is nothing to wait for on our own side).

    Only sequences with >= n stored observations are included -- anything
    thinner is simply left out of this early write (not GPU-scored here; the
    full run picks it up once rivals are available, or it accumulates enough
    observations from the race's own activity). Never overwrites an existing
    file that already has real rival data.
    """
    sequences = sorted({seq for _, seq in our_pairs})
    if not sequences:
        return
    recent_field, recent_epochs = _recent_validator_field(epoch)
    field = recent_field_fingerprint(recent_field, recent_epochs)
    local_passes, short_sequences = local_passes_for_ours(sequences, n, recent_field)
    reused_sequences = set(sequences) - short_sequences
    if not reused_sequences:
        print(f"  ours-only early write: 0 of {len(sequences)} of our own "
              "sequence(s) have enough stored observations yet", flush=True)
        return
    seq_means = {
        s: float(np.mean([p[0][s] for p in local_passes])) for s in reused_sequences
    }
    out_json = out_root / f"multipass_{epoch}.json"
    existing: dict = {}
    if out_json.is_file():
        try:
            existing = json.loads(out_json.read_text()).get(str(epoch), {})
        except (OSError, json.JSONDecodeError):
            existing = {}
    if existing.get("rivals"):
        return
    scores = sorted(seq_means.values())
    result = {
        "ours": {
            "mean": float(np.mean(scores)),
            "stdev": float(np.std(scores)) if len(scores) > 1 else None,
            "fleet_size": len(sequences),
            "scores": [float(v) for v in scores],
        },
        "rivals": {},
        "sequences": {s: v for s, v in seq_means.items()},
        "method": {
            "score_kind": PASS_SCORE_KIND, "formula": PASS_FORMULA_ID,
            "measurement": MEASUREMENT, "passes": n,
            "field": field.get("field"), "field_rows": field.get("field_rows"),
            "field_sha256": field.get("field_sha256"), "writer": WRITER,
        },
        "preliminary": True,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({str(epoch): result}, indent=2) + "\n")
    print(f"WROTE_EARLY {out_json} ours={len(reused_sequences)} (rivals "
          f"pending, {len(short_sequences)} of our own sequences too thin "
          "to reuse)", flush=True)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    epoch = int(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    # Before any GPU time: the label we write must still mean this formula.
    assert_pass_label()
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
    # Gate the FULL (ours+rivals) result on the public leaderboard: writing
    # ours-only output used to make the caller's skip-if-exists cache never
    # retry the epoch once rivals were actually published (25616-25618 were
    # stuck with 0 rivals). That cache now keys off `rivals` being non-empty
    # (score_latest_rival_epoch.sh's complete()), so an ours-only write below
    # is safe: it no longer causes that regression, only removes the wait.
    if pairs is None or not {ck for ck, _ in pairs if ck != OUR_COLDKEY}:
        reason = "not published yet or API error" if pairs is None else \
            "published but zero rival coldkeys (suspicious)"
        print(f"LEADERBOARD_UNAVAILABLE epoch={epoch} ({reason}); trying "
              "our-own-intent fallback so 'ours' isn't blocked on the "
              "validator's publish lag", flush=True)
        intended = fetch_our_intended(epoch)
        if intended:
            archive_path_fallback = archive_path if archive_path.exists() else None
            if archive_path_fallback is not None:
                _write_ours_only_early(
                    epoch, intended, archive_path_fallback, n, out_root)
            else:
                print(f"ARCHIVE_UNAVAILABLE epoch={epoch} {archive_path}; "
                      "cannot score the our-own-intent fallback either",
                      flush=True)
        else:
            print(f"NO_INTENDED_SEQUENCES epoch={epoch}; nothing to score "
                  "yet, retry later", flush=True)
        return 3
    leaderboard_sequences = {seq for _, seq in pairs}
    supplement = [
        (coldkey, seq) for coldkey, seq in fetch_our_intended(epoch)
        if seq not in leaderboard_sequences
    ]
    if supplement:
        print(
            f"epoch={epoch} our-intended supplement: {len(supplement)} "
            "sequence(s) not on the public leaderboard (e.g. rejected as a "
            "duplicate resubmission), scored anyway", flush=True,
        )
        pairs = pairs + supplement
    by_coldkey: dict[str, list[str]] = defaultdict(list)
    for coldkey, seq in pairs:
        by_coldkey[coldkey].append(seq)
    if not by_coldkey:
        print(f"NO_RIVALS epoch={epoch}", flush=True)
        return 1

    sequences = sorted({seq for _, seq in pairs})
    # Still loaded for hf_topup's unrelated use below (identifying HF-archive
    # parent candidates) -- the RANKING field is now the recent-epochs pool,
    # not this static archive. See _recent_validator_field.
    archive = pd.read_csv(archive_path).drop_duplicates("sequence", keep="last")
    archive["sequence"] = archive["sequence"].astype(str).str.upper()
    recent_field, recent_epochs = _recent_validator_field(epoch)
    print(f"epoch={epoch} coldkeys={len(by_coldkey)} seqs={len(sequences)} "
          f"passes={n} recent_field_epochs={recent_epochs} "
          f"recent_field_rows={len(recent_field)}", flush=True)

    field = recent_field_fingerprint(recent_field, recent_epochs)
    cohort_of = {seq: ("ours" if coldkey == OUR_COLDKEY else "rival")
                 for coldkey, seq in pairs}

    # Bounded HF top-up (R1): HF-archive parent candidates ride along in the
    # same scorer passes, each only for the passes it is still missing. They
    # are kept out of the leaderboard aggregation and the dashboard numbers.
    topup_cfg, topup_plan, topup_existing, topup_batches = prepare_topup(
        archive, set(sequences))
    for seq in topup_plan:
        cohort_of[seq] = TOPUP_COHORT
    leaderboard_set = set(sequences)

    # Never spend GPU time re-scoring our own sequences: the race already
    # measured them, often dozens to hundreds of times (ARCHITECTURE_REDESIGN
    # new finding 9). Reuse those stored raw metrics for any of our sequences
    # with at least `n` of them; anything thinner still gets a real pass so
    # it is never degraded, just not optimized.
    our_sequences = sorted(seq for seq, c in cohort_of.items() if c == "ours")
    local_passes, short_sequences = (
        local_passes_for_ours(our_sequences, n, recent_field) if our_sequences else ([], set())
    )
    reused_sequences = set(our_sequences) - short_sequences
    if reused_sequences:
        print(f"  reused {len(reused_sequences)} of our own sequence(s) from "
              f"stored observations, no GPU time spent on them", flush=True)
    gpu_sequences = [s for s in sequences if s not in reused_sequences]

    # Write OUR OWN distribution immediately, before spending any time on the
    # (much slower) real rival GPU passes: the race already measured these
    # sequences, so there is nothing to wait for. "rivals" stays {} here --
    # score_latest_rival_epoch.sh's complete() check (needs >=1 rival) and the
    # dashboard sync (nanobody_dash's build.py) both already treat an
    # ours-only file as legitimate, just not yet complete, so this rides the
    # existing 5/10-minute refresh+sync cadence instead of waiting ~20-30
    # minutes for the rival passes too.
    if reused_sequences:
        _write_ours_only_early(
            epoch, [(OUR_COLDKEY, s) for s in our_sequences], archive_path, n,
            out_root)

    per_pass: list[dict[str, float]] = []
    per_pass_structures: list[dict[str, dict]] = []
    per_pass_raw: list[dict[str, dict[str, float]]] = []
    topup_rows: list[tuple[int, dict, dict, dict]] = []  # (run, ranks, structs, raw)
    rounds = max(n, len(topup_batches))
    for run in range(1, rounds + 1):
        extra = topup_batches[run - 1] if run <= len(topup_batches) else []
        batch = (gpu_sequences if run <= n else []) + extra
        ranks, structs, raw = score_pass(batch, run, recent_field=recent_field, config=config,
                                         scorer=scorer, out_root=out_root)
        if run <= n and local_passes:
            local_ranks, local_structs, local_raw = local_passes[run - 1]
            ranks = {**ranks, **local_ranks}
            structs = {**structs, **local_structs}
            raw = {**raw, **local_raw}
        if run <= n:
            per_pass.append({s: v for s, v in ranks.items() if s in leaderboard_set})
            per_pass_structures.append(
                {s: v for s, v in structs.items() if s in leaderboard_set})
            per_pass_raw.append({s: v for s, v in raw.items() if s in leaderboard_set})
        if extra:
            keep = set(extra)
            topup_rows.append((
                run, {s: v for s, v in ranks.items() if s in keep},
                {s: v for s, v in structs.items() if s in keep},
                {s: v for s, v in raw.items() if s in keep}))
        print(f"  pass {run}/{rounds} done (topup={len(extra)})", flush=True)

    # A sequence can be rejected by the scorer in some passes (liability/fold
    # failure), so only aggregate over sequences present in EVERY pass.
    common = sorted(set.intersection(*(set(p) for p in per_pass)))
    seq_means = {s: float(np.mean([p[s] for p in per_pass])) for s in common}

    # Persist every pass into the mutation DB, one row per pass per sequence,
    # WITH its raw metrics and the field it was ranked against (see
    # PASS_SCORE_KIND). The captured .cif reference rides along via the
    # `structure` field, which update_observation_store persists to
    # observation_artifacts.
    #
    # A failed write no longer exits 0: the JSON below is still written (the
    # dashboard panel needs it, and the raw passes stay on disk for
    # tools/backfill_multipass_metrics.py), but the run returns
    # EXIT_DB_WRITE_FAILED and records the failure for the liveness checker.
    db_failed = False
    try:
        from nanobody_opt.mutation_network import update_observation_store
        observations = []
        for run, (ranks, structs, raw) in enumerate(
            zip(per_pass, per_pass_structures, per_pass_raw), start=1
        ):
            observations.extend(pass_observations(
                epoch=epoch, run=run, passes=n, ranks=ranks, raw=raw,
                structs=structs, cohort_of=cohort_of, field=field))
        written = update_observation_store(
            Path("/root/artifacts/nanobody_mutations.sqlite"), observations)
        print(f"DB_WRITE observations={written} sequences={len(seq_means)} "
              f"score_kind={PASS_SCORE_KIND}", flush=True)
    except Exception as exc:  # noqa: BLE001
        db_failed = True
        print(f"DB_WRITE_FAILED {type(exc).__name__}: {exc}", flush=True)
        try:
            from nanobody_opt.failure_log import record_failure
            record_failure(WRITER, "DB_WRITE_FAILED", exc, epoch=epoch, passes=n)
        except Exception:  # noqa: BLE001 - stale box tree without the helper
            import traceback
            traceback.print_exc()

    # The HF top-up passes: same label, source, writer, formula and field as
    # the leaderboard rows (one scale), cohort=hf_top200. Same observation id
    # scheme, so backfill_multipass_metrics re-imports of this epoch's score
    # dirs are no-ops on them. Written separately: a failure here is logged and
    # self-heals (the passes are still missing, so the next run re-plans them);
    # it does not fail the leaderboard run.
    topup_written = 0
    if topup_rows:
        try:
            from nanobody_opt.mutation_network import update_observation_store
            observations = []
            for run, ranks, structs, raw in topup_rows:
                for seq in ranks:
                    observations.extend(pass_observations(
                        epoch=epoch, run=run, passes=topup_plan[seq],
                        ranks={seq: ranks[seq]}, raw=raw, structs=structs,
                        cohort_of=cohort_of, field=field,
                        extra_provenance={
                            "topup": True,
                            "topup_target_depth": topup_cfg.get("depth"),
                            "stored_passes_before": int(topup_existing.get(seq, 0)),
                        }))
            topup_written = update_observation_store(
                Path("/root/artifacts/nanobody_mutations.sqlite"), observations)
            print(f"TOPUP_DB_WRITE observations={topup_written} "
                  f"sequences={len({o['sequence'] for o in observations})} "
                  f"score_kind={PASS_SCORE_KIND}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"TOPUP_DB_WRITE_FAILED {type(exc).__name__}: {exc}", flush=True)
            try:
                from nanobody_opt.failure_log import record_failure
                record_failure(WRITER, "TOPUP_DB_WRITE_FAILED", exc, epoch=epoch)
            except Exception:  # noqa: BLE001
                pass

    # Preserve the sequence -> mean rank-sum mapping (do NOT drop it) in the
    # artifact alongside the per-coldkey rollup.
    result: dict[str, dict] = {
        "ours": None,
        "rivals": {},
        "sequences": {s: float(seq_means[s]) for s in common},
        # How these numbers were made, so the dashboard never joins two
        # measurement regimes into one line (docs/OBSERVATION_LABELS.md).
        "method": {
            "score_kind": PASS_SCORE_KIND,
            "formula": PASS_FORMULA_ID,
            "measurement": MEASUREMENT,
            "passes": n,
            "field": field.get("field"),
            "field_rows": field.get("field_rows"),
            "field_sha256": field.get("field_sha256"),
            "writer": WRITER,
        },
        # The bounded HF top-up that rode along (R1). Counts only: its
        # sequences are parent candidates, not part of this epoch's field.
        "topup": {
            "cohort": TOPUP_COHORT,
            "enabled": bool(topup_cfg.get("enabled")),
            "depth": topup_cfg.get("depth"),
            "sequences": len(topup_plan),
            "sequence_passes": int(sum(topup_plan.values())),
            "rows_written": int(topup_written or 0),
        },
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
    if db_failed:
        print(f"EXIT {EXIT_DB_WRITE_FAILED}: multipass JSON written but the DB "
              "write failed (see DB_WRITE_FAILED above)", flush=True)
        return EXIT_DB_WRITE_FAILED
    return 0


#: JSON written, DB rows not: the caller must surface this, not cache it as OK.
EXIT_DB_WRITE_FAILED = 4


if __name__ == "__main__":
    raise SystemExit(main())
