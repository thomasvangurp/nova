"""Wire scored children candidates into the per-uid fleet submission path.

This is an ADDITIVE lane alongside ``tools/publish_current_nanobody.py``, not a
replacement for it.  The producer/publisher keeps running unchanged; this module
reads the children score bank out of the mutation database and writes the same
``nanobody-fleet`` artifact shape the submit host already consumes, so the
existing commit mechanism (``deploy/submit_host/pi_submit.py`` ->
``commit_submission.py`` / ``pi_submit.commit_batch``) does the rest unchanged.

Data source (production)::

    /root/artifacts/nanobody_mutations.sqlite
      observations  WHERE source='children_8x'
                     AND score_kind='local_validator_replay_rank_sum'
                     AND observation_count>=8
      nodes         (sequence)

One observation row per (source, epoch, sequence) carries the ten BoltzGen metrics
in ``metrics_json`` (including ``interaction_pae`` -- the fast replay emits all ten),
and a stored ``score`` that sums those ten metric dense ranks.  The top-N children
are re-ranked by the corrected validator ``final_score`` (rank each of confidence /
physical_interaction / developability within the field and sum the three ranks), via
``simulate_round.validator_final_scores``.

Selection -> uid binding is deterministic and reuses the fleet module's own
primitives: child rank ``i`` (0-indexed, ascending mean rank sum) binds to
``crew[i]`` where ``crew`` is sorted by uid ascending -- exactly what
``fleet.discover_fleet`` returns and ``fleet.assign_sequences`` consumes.  Any
child whose sequence is already in the published-sequence ledger is skipped
before ranking, so a previously published child never reappears.

CDR3 novelty.  Before binding, an optional CDR3-novelty filter (default ON)
drops any child whose CDR3 sits at or above ``fleet.CDR3_DUPLICATE_SIMILARITY``
(0.85) identity to the CDR3 of a top-50 historical sequence -- the validator's
own clonotype duplicate rule (``utils/nanobodies.is_duplicate`` tier 2.1,
mirrored by ``fleet.same_clonotype``).  CDR3s are extracted the way
``index_top_sequences`` does: ``metanano.utils.cdr_utils.extract_cdrs``
(abnumber), falling back to igblast via
``metanano.utils.igblast_nativeness.features_to_cdrs``.  A child whose CDR3
cannot be extracted is KEPT (an unproven duplicate must not cost a submission),
and the filter fails open to no-op when no archive / CDR3 index is supplied.

Model selection.  By default children are ranked by mean DB replay rank sum.
Passing ``--rank-model <model.joblib>`` (with ``--children-dataset <csv>``)
switches to the trained parent-relative ranker: each candidate's
``predicted_child_score = parent_rank_sum - predicted_improvement``, computed
through ``train_parent_relative_ranker.feature_matrix`` /
``prediction_distribution``.  A child that already carries a full 8x
``local_validator_replay_rank_sum`` score in the DB uses that measured mean
(ground truth) instead of the prediction; the model covers only children
lacking an 8x score.  The CDR3-novelty filter still runs after selection.

No wallet, no hotkey, no chain write.  This module reads a local chain-state
file and the SQLite bank read-only, then either prints what WOULD be submitted
per uid (default, dry-run) or writes the fleet artifact to ``--output`` for the
submit host to pull.  Writing the artifact is a plain file write, not a
commitment: the Pi's existing ``nova-submit-loop`` / ``pi_submit.py`` performs
the signing and extrinsics.

Invocation (dry-run -- prints per-uid, writes nothing)::

    python -m nanobody_opt.children_submit \
        --database /root/artifacts/nanobody_mutations.sqlite \
        --chain-state /root/artifacts/chain_state.json \
        --coldkey <our-coldkey-ss58> \
        --ledger /root/artifacts/nanobody_published_sequences.txt

Invocation (produce the handoff file for the submit host)::

    python -m nanobody_opt.children_submit \
        --database /root/artifacts/nanobody_mutations.sqlite \
        --chain-state /root/artifacts/chain_state.json \
        --coldkey <our-coldkey-ss58> \
        --ledger /root/artifacts/nanobody_published_sequences.txt \
        --output /root/artifacts/nanobody_children_result.json

The submit host then reads it exactly as it reads the producer's artifact:

    python deploy/submit_host/pi_submit.py \
        --nanobody /path/to/nanobody_children_result.json \
        --epoch <epoch> --reaction <1..5> --crew <crew.json> ...

or, on the Pi's pull path, pull it over in place of the producer's
``nanobody_result.json``.  ``--top-n`` defaults to the number of uids the
coldkey currently holds; passing it explicitly clamps the pool to that many
children.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt import db, fleet  # noqa: E402

TARGET = "P20809"
DEFAULT_SOURCE = "children_8x"
DEFAULT_SCORE_KIND = "local_validator_replay_rank_sum"
# Fetch this many extra children past --top-n so the corrected-score re-ranking
# (which uses the exact replay's interaction_pae) has a pool to reorder from.
# Kept small: top_children issues one per-child metrics query, so the margin
# multiplies that latency; 2x is enough because the corrected and stored scores
# are rank-sum cousins, not orthogonal.
CORRECTED_FETCH_MARGIN = 2
DEFAULT_OUTPUT = "/root/artifacts/nanobody_children_result.json"
DEFAULT_RESERVE_BLOCKS = 55
MINIMUM_OBSERVATION_COUNT = 8
SUPPORTED_CHAIN_STATE_SCHEMAS = frozenset({1, 2})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return payload


def read_chain_state(path: Path, max_age: float) -> dict[str, Any]:
    """Read the miner's exported chain view, refusing a stale or torn one."""
    payload = _read_json(path)
    try:
        schema_version = int(payload["schema_version"])
        block = int(payload["block"])
        epoch_length = int(payload["epoch_length"])
        epoch = int(payload["epoch"])
        age = time.time() - float(payload["updated_at_unix"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"{path} is malformed: {error}") from error
    if schema_version not in SUPPORTED_CHAIN_STATE_SCHEMAS:
        raise SystemExit(f"unsupported chain-state schema: {schema_version}")
    if epoch_length < 1 or epoch != block // epoch_length:
        raise SystemExit(f"{path} epoch is internally inconsistent")
    if age < -5 or age > max_age:
        raise SystemExit(f"{path} is stale by {age:.1f}s")
    return {**payload, "block": block, "epoch_length": epoch_length, "epoch": epoch}


def discover_crew(chain: dict[str, Any], coldkey: str, netuid: int) -> list[fleet.FleetMember]:
    """Every uid the chain attributes to ``coldkey`` this epoch, lowest first."""
    address = fleet.resolve_coldkey(coldkey)
    return fleet.discover_fleet(chain.get("neurons"), address, netuid=netuid)


def _open_readonly(database: Path) -> db.Connection:
    try:
        return db.connect()
    except Exception as error:
        raise SystemExit(f"cannot open the mutation database: {error}") from error


#: Number of most-recent ledger entries treated as the validator's live top-50
#: whole-sequence dedup set.  The archive used to build the CDR3 index lags the
#: chain by ~4 epochs (it ends at epoch25452 while the chain is at 25456), so our
#: 25453 winners -- now in the top-50 -- are NOT in it and their single-edit
#: siblings pass the CDR3 gate.  The ledger is append-ordered (record_published
#: opens "a"), so its tail is exactly our recent submissions.  ~13/epoch over
#: five epochs is 65; this drops children of the ~13 already-won parents and lets
#: the selector fall through to the next tier of parents, whose children are >=7
#: edits (0.945 identity) from the winners and therefore not duplicates.
RECENT_PUBLISHED_N = 65


def _published_array(published: Sequence[str], length: int) -> np.ndarray:
    """Recent published sequences of `length`, as a uint8 matrix."""
    if not published:
        return np.zeros((0, length), dtype=np.uint8)
    rows = [[ord(c) for c in s] for s in published if len(s) == length]
    return np.array(rows, dtype=np.uint8) if rows else np.zeros((0, length), dtype=np.uint8)


def _near_duplicate(sequence: str, published: np.ndarray, *, limit: float = 0.95) -> bool:
    """True when `sequence` is >=limit identical to a recent published sequence.

    Mirrors utils/nanobodies.is_duplicate tier 1 (whole-sequence identity >= 0.95),
    which the validator enforces against the top-50 historical -- a set that now
    contains our own prior winners.  A child one edit from a parent we already
    won with is therefore a certain drop next epoch, even though it is not an
    EXACT ledger match.  Hamming <= (1-limit)*len catches it.  `published` is the
    RECENT ledger tail (RECENT_PUBLISHED_N), not the full history: filtering
    against every sequence we ever published over-blocks, because the archive
    (and hence the top-50) only lags by a few epochs, not our whole history.
    """
    if published.size == 0:
        return False
    seq = np.frombuffer(sequence.encode(), dtype=np.uint8)
    if seq.shape[0] != published.shape[1]:
        return False
    max_mismatch = int((1.0 - limit) * seq.shape[0])
    return bool((published != seq).sum(axis=1).min() <= max_mismatch)


def top_children(
    database: Path,
    *,
    source: str,
    score_kind: str,
    top_n: int,
    exclude: set[str],
    published: np.ndarray,
) -> list[dict[str, Any]]:
    """Top-N children by mean replay rank sum (lower is better), deduped.

    Only nodes carrying at least one observation with ``observation_count`` at
    ``MINIMUM_OBSERVATION_COUNT`` are eligible.  Returns one entry per node with
    its sequence, mean/median/p90 score, summed observation count and the
    mean-of-passes metric aggregate from a representative observation.
    """
    connection = _open_readonly(database)
    try:
        aggregate = connection.execute(
            """
            SELECT n.id, n.sequence,
                   AVG(o.score),
                   MIN(o.epoch),
                   COUNT(*),
                   SUM(o.observation_count)
            FROM observations o
            JOIN nodes n ON n.id = o.node_id
            WHERE o.source = %s
              AND o.score_kind = %s
            GROUP BY n.id
            HAVING SUM(o.observation_count) >= %s
            ORDER BY AVG(o.score) ASC, n.sequence ASC
            """,
            (source, score_kind, MINIMUM_OBSERVATION_COUNT),
        ).fetchall()
        chosen: list[dict[str, Any]] = []
        for node_id, sequence, mean_score, first_epoch, rows, observation_count in aggregate:
            sequence = str(sequence).strip().upper()
            if not sequence or sequence in exclude:
                continue
            if _near_duplicate(sequence, published):
                continue
            representative = connection.execute(
                """
                SELECT metrics_json, provenance_json, score
                FROM observations
                WHERE node_id = %s AND source = %s AND score_kind = %s
                ORDER BY observation_id ASC
                LIMIT 1
                """,
                (node_id, source, score_kind),
            ).fetchone()
            metrics: dict[str, Any] = {}
            median_score: float | None = None
            p90_score: float | None = None
            if representative is not None:
                metrics_raw, provenance_raw, score = representative
                try:
                    parsed = json.loads(metrics_raw or "{}")
                    if isinstance(parsed, dict):
                        metrics = parsed
                except json.JSONDecodeError:
                    metrics = {}
                try:
                    provenance = json.loads(provenance_raw or "{}")
                except json.JSONDecodeError:
                    provenance = {}
                if isinstance(provenance, Mapping):
                    try:
                        median_score = float(provenance["score_median"])
                    except (KeyError, TypeError, ValueError):
                        median_score = None
                    try:
                        p90_score = float(provenance["score_p90"])
                    except (KeyError, TypeError, ValueError):
                        p90_score = None
            chosen.append({
                "sequence": sequence,
                "mean_score": float(mean_score),
                "median_score": median_score if median_score is not None else float(mean_score),
                "p90_score": p90_score if p90_score is not None else float(mean_score),
                "observation_count": int(observation_count or 0),
                "first_epoch": int(first_epoch),
                "metrics": metrics,
            })
            if len(chosen) >= top_n:
                break
    finally:
        connection.close()
    return chosen


def _corrected_score_map(database: Path, source: str, score_kind: str) -> dict[str, float]:
    """Per-sequence corrected validator ``final_score`` from the selection bank.

    The stored replay score sums all ten metric dense ranks; the validator actually
    ranks each of the three metric categories within the field and sums those three
    ranks (see ``simulate_round.validator_final_scores``).  Recompute it over the
    same ``source``/``score_kind`` the selection uses, from ``metrics_json``.  The
    fast replay carries ``interaction_pae`` and the other nine metrics, so this does
    not need the exact scorer.  Returns ``{}`` when no rows exist, so callers fall
    back to the stored replay score.
    """
    from nanobody_opt.simulate_round import METRICS, validator_final_scores

    connection = _open_readonly(database)
    try:
        rows = connection.execute(
            """
            SELECT n.sequence, o.metrics_json, o.epoch
            FROM observations o JOIN nodes n ON n.id = o.node_id
            WHERE n.target = %s
              AND o.source = %s
              AND o.score_kind = %s
              AND o.metrics_json IS NOT NULL
            """,
            (TARGET, source, score_kind),
        ).fetchall()
    finally:
        connection.close()

    by_epoch: dict[int, list[tuple[str, dict[str, float]]]] = {}
    for sequence, raw, epoch in rows:
        try:
            metrics = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(metrics, dict) or any(metrics.get(m) is None for m in METRICS):
            continue
        by_epoch.setdefault(int(epoch or 0), []).append(
            (str(sequence).strip().upper(), {m: float(metrics[m]) for m in METRICS})
        )
    if not by_epoch:
        return {}

    import pandas as pd

    merged: dict[str, list[float]] = {}
    for group in by_epoch.values():
        frame = pd.DataFrame([m for _, m in group])
        scores = validator_final_scores(frame)
        for (seq, _m), score in zip(group, scores):
            if not np.isnan(score):
                merged.setdefault(seq, []).append(float(score))
    return {seq: float(np.mean(v)) for seq, v in merged.items()}


def reorder_children_by_corrected_score(
    children: list[dict[str, Any]], corrected: dict[str, float]
) -> list[dict[str, Any]]:
    """Overwrite each child's score with the corrected value where available, re-sort.

    Only children present in ``corrected`` are re-scored; the rest keep their stored
    replay score, so a partial exact-coverage DB degrades rather than errors.
    """
    if not corrected:
        return children
    for child in children:
        value = corrected.get(child["sequence"])
        if value is not None:
            child["mean_score"] = value
            child["median_score"] = value
            child["p90_score"] = value
    return sorted(children, key=lambda child: (child["mean_score"], child["sequence"]))


def _extract_cdrs_like_validator(sequence: str) -> dict[str, str] | None:
    """CDR1/2/3 for one sequence, the way ``index_top_sequences`` extracts them.

    abnumber first (``metanano.utils.cdr_utils.extract_cdrs``), then the
    igblast fallback through ``metanano.utils.igblast_nativeness.features_to_cdrs``
    -- the same order and the same call shapes as utils/nanobodies.py lines
    166-183.  Both imports live in the metanano environment; when that
    environment is not importable this returns None and the caller fails open.
    """
    try:
        from metanano.utils.cdr_utils import extract_cdrs
    except Exception:
        return None
    cdrs = extract_cdrs(sequence)
    if cdrs:
        return {key: str(cdrs.get(key) or "") for key in ("cdr1", "cdr2", "cdr3")}
    try:
        from metanano.utils.igblast_nativeness import features_to_cdrs
        from utils.nanobodies import compute_igblast_nativeness

        handle = tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False)
        handle.write(f">seq\n{sequence}\n")
        handle.close()
        try:
            features = compute_igblast_nativeness(handle.name)
            if features:
                cdrs = features_to_cdrs(features[0]["features"])
                if cdrs:
                    return {key: str(cdrs.get(key) or "") for key in ("cdr1", "cdr2", "cdr3")}
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
    except Exception:
        return None
    return None


def top50_historical_cdr3s(archive: Path, n: int) -> list[str]:
    """CDR3s of the top-n historical sequences, as the validator indexes them.

    Top-n comes from the same Algorithm-2 worst-weighted-rank the diversity gate
    uses (``diversity.minmax_weighted_rank``, the nanobody_opt mirror of
    ``utils/minmax_weighted_rank.rank_binders``), then each sequence's CDR3 is
    extracted the validator's way.  Empty list when the archive or the metanano
    stack is unavailable, so the caller fails open.
    """
    try:
        import pandas as pd

        from nanobody_opt import diversity
    except Exception:
        return []
    try:
        frame = pd.read_csv(archive).drop_duplicates("sequence", keep="first")
    except (OSError, KeyError, ValueError):
        return []
    top = diversity.minmax_weighted_rank(frame, k=n)
    cdr3s: list[str] = []
    for sequence in top.sequence.astype(str).str.strip().str.upper():
        cdrs = _extract_cdrs_like_validator(sequence)
        if cdrs and cdrs.get("cdr3"):
            cdr3s.append(cdrs["cdr3"].strip().upper())
    return cdr3s


def _load_cdr_map(path: Path | None) -> dict[str, dict[str, str]]:
    """Load a ``{"cdrs": {sequence: {cdr1,cdr2,cdr3}}}`` annotation file."""
    if path is None:
        return {}
    payload = _read_json(path)
    raw = payload.get("cdrs") if isinstance(payload, dict) else payload
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for sequence, regions in raw.items():
        if not isinstance(regions, Mapping):
            continue
        out[str(sequence).strip().upper()] = {
            key: str(regions.get(key) or "") for key in ("cdr1", "cdr2", "cdr3")
        }
    return out


def _load_cdr3_index(path: Path) -> list[str]:
    """Load a precomputed top-50 CDR3 set (``{"cdr3s": [...]}`` or a CDR map)."""
    payload = _read_json(path)
    raw = payload.get("cdr3s") if isinstance(payload, dict) else payload
    if isinstance(raw, list):
        return [str(item).strip().upper() for item in raw if str(item).strip()]
    cdr3s = []
    if isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, Mapping):
                cdr3 = str(value.get("cdr3") or "").strip().upper()
                if cdr3:
                    cdr3s.append(cdr3)
            elif str(value).strip():
                cdr3s.append(str(value).strip().upper())
    return cdr3s


def _load_top50_sequences(path: Path) -> list[str]:
    """Load the top-50 whole sequences from a ``build_top50_index.py`` JSON.

    The validator's ``is_duplicate`` tier 1 is whole-sequence identity >= 0.95
    against these exact sequences, so the whole-sequence gate needs the
    sequences themselves, not just their CDR3s.  Returns [] when the file has no
    ``sequences`` field (e.g. the older CDR3-only index), so the caller falls
    back to the recent-ledger tail.
    """
    payload = _read_json(path)
    raw = payload.get("sequences") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    return [str(item).strip().upper() for item in raw if str(item).strip()]


def cdr3_novelty_filter(
    children: Sequence[dict[str, Any]],
    top50_cdr3s: Sequence[str],
    *,
    child_cdrs: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Drop children whose CDR3 the validator would call a top-50 duplicate.

    The duplicate line is ``fleet.CDR3_DUPLICATE_SIMILARITY`` (0.85) positional
    identity over the CDR3, computed by ``fleet._region_similarity`` -- the same
    arithmetic ``fleet.same_clonotype`` and ``is_duplicate`` tier 2.1 use.
    Returns ``(kept, dropped, unannotated)``; a child whose CDR3 cannot be
    extracted is kept, because an unproven duplicate must not cost a slot.
    """
    targets = [cdr3 for cdr3 in top50_cdr3s if cdr3]
    if not targets:
        return list(children), [], 0
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    unannotated = 0
    known = dict(child_cdrs or {})
    for child in children:
        cdrs = known.get(child["sequence"])
        if cdrs is None:
            cdrs = _extract_cdrs_like_validator(child["sequence"])
        cdr3 = (cdrs.get("cdr3") or "").strip().upper() if cdrs else ""
        if not cdr3:
            unannotated += 1
            kept.append(child)
            continue
        if any(
            fleet._region_similarity(cdr3, other) >= fleet.CDR3_DUPLICATE_SIMILARITY
            for other in targets
        ):
            dropped.append(child)
        else:
            kept.append(child)
    return kept, dropped, unannotated


def load_rank_model(path: Path) -> Any:
    """Load the persisted parent-relative ranker (joblib)."""
    if not path.is_file():
        raise SystemExit(f"rank model is missing: {path}")
    import joblib

    return joblib.load(path)


def model_candidate_frame(path: Path, length: int) -> Any:
    """Read the (parent, child) candidate frame ``feature_matrix`` consumes.

    Expected columns are the ``assemble_children_dataset.py`` output:
    sequence, parent_sequence, edit_distance, proposal_type,
    proposal_prior_score, parent_selection_score, observation_count,
    parent_rank_sum, and parent_<metric> for every metric in
    ``simulate_round.METRICS``. Rows whose sequence or parent is not ``length``
    residues are dropped, as the training driver does.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    frame["sequence"] = frame.sequence.astype(str).str.strip().str.upper()
    frame["parent_sequence"] = frame.parent_sequence.astype(str).str.strip().str.upper()
    frame = frame[
        (frame.sequence.str.len() == length)
        & (frame.parent_sequence.str.len() == length)
    ].reset_index(drop=True)
    return frame


def predict_children(model: Any, frame: Any, length: int) -> Any:
    """predicted_child_score = parent_rank_sum - predicted_improvement.

    Reuses ``train_parent_relative_ranker.feature_matrix`` /
    ``prediction_distribution`` (the per-tree mean), so the feature encoding and
    the prediction are the exact ones the ranker was trained with.
    """
    from nanobody_opt import train_parent_relative_ranker as tpr

    mean, _deviation, _probability = tpr.prediction_distribution(
        model, tpr.feature_matrix(frame, length=length)
    )
    output = frame.copy()
    output["predicted_improvement"] = mean
    output["predicted_child_score"] = output["parent_rank_sum"] - mean
    return output


def measured_means(
    database: Path, *, source: str, score_kind: str
) -> dict[str, dict[str, Any]]:
    """Children with a full 8x DB score -> measured mean (ground truth)."""
    connection = _open_readonly(database)
    try:
        rows = connection.execute(
            """
            SELECT n.sequence, AVG(o.score), SUM(o.observation_count)
            FROM observations o JOIN nodes n ON n.id = o.node_id
            WHERE o.source = %s AND o.score_kind = %s
            GROUP BY n.id
            HAVING SUM(o.observation_count) >= %s
            """,
            (source, score_kind, MINIMUM_OBSERVATION_COUNT),
        ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for sequence, mean, observation_count in rows:
            sequence = str(sequence).strip().upper()
            metrics: dict[str, Any] = {}
            representative = connection.execute(
                """
                SELECT o.metrics_json
                FROM observations o JOIN nodes n ON n.id = o.node_id
                WHERE n.sequence = %s AND o.source = %s AND o.score_kind = %s
                ORDER BY o.observation_id ASC LIMIT 1
                """,
                (sequence, source, score_kind),
            ).fetchone()
            if representative is not None:
                try:
                    parsed = json.loads(representative[0] or "{}")
                    if isinstance(parsed, dict):
                        metrics = parsed
                except json.JSONDecodeError:
                    metrics = {}
            out[sequence] = {
                "mean": float(mean),
                "observation_count": int(observation_count or 0),
                "metrics": metrics,
            }
        return out
    finally:
        connection.close()


def rank_with_model(
    predicted: Any,
    measured: Mapping[str, Mapping[str, Any]],
    *,
    exclude: set[str],
    top_n: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Merge measured ground truth over model predictions, best-first.

    Precedence: a child with a full 8x DB score uses its measured mean; every
    other child uses ``predicted_child_score``. Returns (children, by_model,
    by_measured) where the counts are over the SELECTED top-n, ascending (lower
    is better).
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in predicted.itertuples(index=False):
        sequence = str(record.sequence).strip().upper()
        if not sequence or sequence in exclude:
            continue
        seen.add(sequence)
        if sequence in measured:
            entry = measured[sequence]
            effective = float(entry["mean"])
            observation_count = int(entry["observation_count"])
            metrics = dict(entry["metrics"])
            source = "measured"
        else:
            effective = float(record.predicted_child_score)
            observation_count = int(getattr(record, "observation_count", 0) or 0)
            metrics = {}
            source = "model"
        rows.append({
            "sequence": sequence,
            "mean_score": effective,
            "median_score": effective,
            "p90_score": effective,
            "observation_count": observation_count,
            "metrics": metrics,
            "score_source": source,
        })
    # A measured 8x child absent from the dataset is still a candidate: the DB
    # ground truth is its score, and it needs no feature row.
    for sequence, entry in measured.items():
        if sequence in exclude or sequence in seen:
            continue
        rows.append({
            "sequence": sequence,
            "mean_score": float(entry["mean"]),
            "median_score": float(entry["mean"]),
            "p90_score": float(entry["mean"]),
            "observation_count": int(entry["observation_count"]),
            "metrics": dict(entry["metrics"]),
            "score_source": "measured",
        })
    rows.sort(key=lambda child: (child["mean_score"], child["sequence"]))
    selected = rows[:top_n]
    by_model = sum(1 for child in selected if child["score_source"] == "model")
    by_measured = sum(1 for child in selected if child["score_source"] == "measured")
    return selected, by_model, by_measured


def _checkpoint_for(child: dict[str, Any], *, target: str, epoch: int) -> dict[str, Any]:
    """A minimal per-sequence checkpoint shaped like publish_checkpoint output.

    The fleet artifact wraps each entry's ``checkpoint`` verbatim, so the
    submit host's own validation sees the same fields the producer's does.  The
    children lane has no Layer-2 / novelty / pose evidence of its own, so those
    sections are absent rather than invented.
    """
    metrics = {str(key): float(value) for key, value in child["metrics"].items()
               if _finite(value)}
    return {
        "publication_stage": "children",
        "target": target,
        "epoch": epoch,
        "sequences": [child["sequence"]],
        "validation": {
            "boltzgen": {
                "exact": True,
                "metrics": metrics,
                "rank_method": "dense",
                "metric_aggregation": "arithmetic_mean",
            },
        },
        "exact": {
            "scorer": "nova-boltzgen-validator-wrapper",
            "observation_count": child["observation_count"],
            "metrics": metrics,
        },
        "selection": {
            "method": "children-8x-mean-replay-rank-sum",
            "score": child["mean_score"],
            "score_median": child["median_score"],
            "score_p90": child["p90_score"],
            "win_probability": None,
            "score_semantics": (
                "mean validator-replay p90 rank sum across 8 passes; lower is "
                "better"
            ),
        },
    }


def _finite(value: Any) -> bool:
    try:
        return float(value) == float(value)
    except (TypeError, ValueError):
        return False


def build_artifact(
    *,
    target: str,
    epoch: int,
    epoch_length: int,
    block: int,
    crew: Sequence[fleet.FleetMember],
    children: Sequence[dict[str, Any]],
    reserve_blocks: int,
) -> tuple[dict[str, Any], list[fleet.Assignment]]:
    """Bind child rank i to crew[i] and build the nanobody-fleet artifact."""
    if not children:
        raise SystemExit("no eligible children remain after deduplication")
    pool = [child["sequence"] for child in children]
    bound = min(len(crew), len(pool))
    try:
        blocks = fleet.commitment_blocks(
            epoch=epoch,
            epoch_length=epoch_length,
            reserve_blocks=reserve_blocks,
            count=bound,
            block=block if block >= 0 else None,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    assignments = fleet.assign_sequences(crew, pool, blocks)
    checkpoints = {
        child["sequence"]: _checkpoint_for(child, target=target, epoch=epoch)
        for child in children[: len(assignments)]
    }
    artifact = fleet.fleet_artifact(
        target=target,
        epoch=epoch,
        stage="children",
        assignments=assignments,
        checkpoints=checkpoints,
        unassigned=fleet.uncovered(crew, assignments),
        sequence_pool=pool,
    )
    return artifact, assignments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("/root/artifacts/nanobody_mutations.sqlite"),
        help="mutation database holding the children score bank (read-only)",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--score-kind", default=DEFAULT_SCORE_KIND)
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="number of children to select; defaults to the number of uids the "
             "coldkey holds",
    )
    parser.add_argument(
        "--chain-state",
        type=Path,
        default=Path("/root/artifacts/chain_state.json"),
    )
    parser.add_argument("--maximum-chain-state-age", type=float, default=90)
    parser.add_argument(
        "--coldkey",
        help="our coldkey ss58; the fleet is discovered from the chain by this "
             "address. Defaults to the wallet coldkeypub if omitted.",
    )
    parser.add_argument("--wallet-name", default="mining_ops")
    parser.add_argument(
        "--wallet-path",
        type=Path,
        default=Path.home() / ".bittensor" / "wallets",
    )
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("/root/artifacts/nanobody_published_sequences.txt"),
        help="published-sequence tracker; any child already here is skipped",
    )
    parser.add_argument(
        "--cdr-novelty-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop children whose CDR3 is a validator duplicate of a top-50 "
             "historical CDR3 (default ON); --no-cdr-novelty-filter disables",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="historical archive CSV for the target; when set, the top-50 "
             "historical CDR3 set is computed from it (Algorithm-2 rank + "
             "abnumber/igblast CDR extraction)",
    )
    parser.add_argument(
        "--cdr3-index",
        type=Path,
        default=None,
        help="precomputed top-50 CDR3 JSON ({\"cdr3s\": [...]} or a CDR map); "
             "skips archive ranking and CDR extraction at submit time",
    )
    parser.add_argument(
        "--children-cdrs",
        type=Path,
        default=None,
        help="precomputed child CDR JSON ({\"cdrs\": {seq: {cdr1,cdr2,cdr3}}}); "
             "skips per-child abnumber/igblast extraction",
    )
    parser.add_argument("--top50-count", type=int, default=50)
    parser.add_argument(
        "--rank-model",
        type=Path,
        default=None,
        help="persisted parent-relative ranker (joblib); when set, model-based "
             "selection replaces the mean-score ranking. Requires "
             "--children-dataset. Default OFF (current behavior).",
    )
    parser.add_argument(
        "--no-rank-model",
        action="store_true",
        help="explicitly disable model-based selection even if --rank-model is set",
    )
    parser.add_argument(
        "--children-dataset",
        type=Path,
        default=None,
        help="(parent, child) feature CSV for model mode (assemble_children_dataset.py "
             "output shape); required when --rank-model is set",
    )
    parser.add_argument("--model-length", type=int, default=127)
    parser.add_argument(
        "--reserve-blocks",
        type=int,
        default=DEFAULT_RESERVE_BLOCKS,
        help="end-of-epoch blocks reserved for commitments",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the fleet artifact here; omit to dry-run (print only)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="assert the chain-state nanobody_target equals this",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    chain = read_chain_state(args.chain_state, args.maximum_chain_state_age)
    epoch = int(chain["epoch"])
    epoch_length = int(chain["epoch_length"])
    block = int(chain["block"])
    target = str(chain.get("nanobody_target") or TARGET).strip()
    if args.target is not None and target != args.target:
        raise SystemExit(
            f"chain-state target is {target!r}, not {args.target!r}"
        )
    coldkey = fleet.resolve_coldkey(
        args.coldkey,
        wallet_root=args.wallet_path,
        wallet_name=args.wallet_name,
    )
    crew = discover_crew(chain, coldkey, args.netuid)
    print(
        f"children lane: epoch={epoch} block={block} target={target} "
        f"coldkey={coldkey} fleet={len(crew)} uid(s)",
        flush=True,
    )
    top_n = args.top_n if args.top_n is not None else len(crew)
    if top_n < 1:
        raise SystemExit("--top-n must be positive")
    ledger = fleet.read_ledger(args.ledger)
    exclude = set(ledger)
    # Whole-sequence dedup runs against the validator's live top-50, reconstructed
    # as the FRESH API index (build_top50_index.py) UNION our recent ledger tail.
    # The API index is fresh but may lag one epoch; the recent tail covers that.
    whole_seq_dedup = list(ledger[-RECENT_PUBLISHED_N:])
    if args.cdr3_index is not None and Path(args.cdr3_index).is_file():
        whole_seq_dedup.extend(_load_top50_sequences(args.cdr3_index))
    published = _published_array(whole_seq_dedup, args.model_length)
    model_mode = args.rank_model is not None and not args.no_rank_model
    if model_mode:
        if args.children_dataset is None:
            raise SystemExit(
                "--rank-model requires --children-dataset (the (parent, child) "
                "feature CSV from assemble_children_dataset.py)"
            )
        model = load_rank_model(args.rank_model)
        estimators = getattr(model, "estimators_", None)
        print(
            f"RANK_MODEL_LOADED path={args.rank_model} type={type(model).__name__} "
            f"estimators={len(estimators) if estimators is not None else '?'}",
            flush=True,
        )
        frame = model_candidate_frame(args.children_dataset, args.model_length)
        predicted = predict_children(model, frame, args.model_length)
        measured = measured_means(
            args.database, source=args.source, score_kind=args.score_kind
        )
        children, by_model, by_measured = rank_with_model(
            predicted, measured, exclude=exclude, top_n=top_n
        )
        print(
            f"children lane: model mode: pool={len(predicted)} "
            f"measured_8x={len(measured)} selected_by_model={by_model} "
            f"selected_by_score={by_measured}; "
            f"ledger holds {len(exclude)} published sequence(s)",
            flush=True,
        )
    else:
        print(
            f"children lane: selecting top-{top_n} by mean {args.score_kind} "
            f"(source={args.source}); ledger holds {len(exclude)} published sequence(s)",
            flush=True,
        )
        corrected = _corrected_score_map(args.database, args.source, args.score_kind)
        children = top_children(
            args.database,
            source=args.source,
            score_kind=args.score_kind,
            top_n=top_n * CORRECTED_FETCH_MARGIN,
            exclude=exclude,
            published=published,
        )
        children = reorder_children_by_corrected_score(children, corrected)
        children = children[:top_n]
    if not children:
        raise SystemExit(
            f"no eligible {args.source} children in {args.database} after "
            "deduplication; nothing to submit"
        )
    print(
        f"children lane: {len(children)} distinct child(ren) after dedup",
        flush=True,
    )
    if args.cdr_novelty_filter:
        top50_cdr3s: list[str] = []
        if args.cdr3_index is not None:
            top50_cdr3s = _load_cdr3_index(args.cdr3_index)
        elif args.archive is not None:
            top50_cdr3s = top50_historical_cdr3s(args.archive, args.top50_count)
        if top50_cdr3s:
            child_cdrs = (
                _load_cdr_map(args.children_cdrs)
                if args.children_cdrs is not None
                else None
            )
            children, dropped, unannotated = cdr3_novelty_filter(
                children, top50_cdr3s, child_cdrs=child_cdrs
            )
            print(
                f"children lane: CDR3-novelty filter: top50_cdr3s={len(top50_cdr3s)} "
                f"dropped={len(dropped)} unannotated={unannotated} kept={len(children)}",
                flush=True,
            )
            for child in dropped:
                print(
                    f"CHILD_CDR3_DROPPED sequence={child['sequence']} "
                    f"mean_score={child['mean_score']:.2f}",
                    flush=True,
                )
            if not children:
                raise SystemExit(
                    "CDR3-novelty filter dropped every child; nothing to submit"
                )
        else:
            print(
                "children lane: CDR3-novelty filter enabled but no --archive or "
                "--cdr3-index supplied; skipping (fail-open)",
                flush=True,
            )
    else:
        print("children lane: CDR3-novelty filter disabled", flush=True)
    artifact, assignments = build_artifact(
        target=target,
        epoch=epoch,
        epoch_length=epoch_length,
        block=block,
        crew=crew,
        children=children,
        reserve_blocks=args.reserve_blocks,
    )
    for index, assignment in enumerate(assignments):
        child = children[index]
        print(
            f"CHILD_SUBMIT uid={assignment.uid} hotkey={assignment.hotkey} "
            f"block={assignment.commit_block} "
            f"mean_score={child['mean_score']:.2f} "
            f"obs={child['observation_count']} "
            f"sequence={assignment.sequence}",
            flush=True,
        )
    for member in artifact["unassigned_uids"]:
        print(
            f"CHILD_SUBMIT_UNCOVERED uid={member['uid']} hotkey={member['hotkey']} "
            "no distinct child available",
            flush=True,
        )
    if args.output is None:
        print(
            f"children lane: dry-run; no artifact written. "
            f"Pass --output to write the fleet artifact for the submit host.",
            flush=True,
        )
        return 0
    fleet.write_atomic_json(args.output, artifact)
    print(
        f"children lane: wrote {args.output} ({len(assignments)} submission(s), "
        f"pool {len(artifact['sequence_pool'])})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
