"""Apply the trained parent-relative ranker to candidate single-edits.

Loads ``model.joblib`` (ExtraTrees, produced by ``train_parent_relative_ranker``)
and predicts ``target_improvement = parent_rank_sum - child_rank_sum`` for each
parent -> child single-edit, so the emitted candidate list is front-loaded with
the edits most likely to beat their parent.

The model is **parent-relative**: its 0.81 Spearman on the chronologically
sealed holdout is on the improvement *within* a parent (which residue edit is
best for a given parent), not on absolute child quality. So use it to order
residues for each parent -- parent selection remains a separate, upstream step.

This module is deliberately self-contained: it carries verbatim copies of
``feature_matrix`` / ``prediction_distribution`` / ``_region_summary`` /
``AA_PROPERTIES`` / ``PROPOSAL_TYPES`` from ``train_parent_relative_ranker`` so
it does NOT import that script's heavy module-level dependencies (``rank_exact``
-> ``mutation_network`` -> ``pose_stability``). If the trainer's feature
encoding changes, this file must be updated to match.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nanobody_opt.regions import find_regions
from nanobody_opt.simulate_round import METRICS, rank_sum_in_field

# --- verbatim from train_parent_relative_ranker.py (keep in sync) -----------

# Inlined (not imported from backtest_mutation_ranker, which pulls rank_exact ->
# mutation_network -> pose_stability at module level and would defeat the point
# of keeping this inference path lightweight).
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}

PROPOSAL_TYPES = (
    "exhaustive_single_substitution",
    "local_substitution",
    "whole_cdr_motif",
)

AA_PROPERTIES = {
    # Kyte-Doolittle hydropathy, approximate charge, volume, polarity flag.
    "A": (1.8, 0.0, 88.6, 0.0), "C": (2.5, 0.0, 108.5, 0.0),
    "D": (-3.5, -1.0, 111.1, 1.0), "E": (-3.5, -1.0, 138.4, 1.0),
    "F": (2.8, 0.0, 189.9, 0.0), "G": (-0.4, 0.0, 60.1, 0.0),
    "H": (-3.2, 0.1, 153.2, 1.0), "I": (4.5, 0.0, 166.7, 0.0),
    "K": (-3.9, 1.0, 168.6, 1.0), "L": (3.8, 0.0, 166.7, 0.0),
    "M": (1.9, 0.0, 162.9, 0.0), "N": (-3.5, 0.0, 114.1, 1.0),
    "P": (-1.6, 0.0, 112.7, 0.0), "Q": (-3.5, 0.0, 143.8, 1.0),
    "R": (-4.5, 1.0, 173.4, 1.0), "S": (-0.8, 0.0, 89.0, 1.0),
    "T": (-0.7, 0.0, 116.1, 1.0), "V": (4.2, 0.0, 140.0, 0.0),
    "W": (-0.9, 0.0, 227.8, 0.0), "Y": (-1.3, 0.0, 193.6, 1.0),
}


def _region_summary(parent: str, sequence: str) -> list[float]:
    try:
        regions = find_regions(parent)
        region_sets = [
            set(range(*regions.cdr1)),
            set(range(*regions.cdr2)),
            set(range(*regions.cdr3)),
        ]
    except ValueError:
        region_sets = [set(), set(), set()]
    changed = {
        position
        for position, (left, right) in enumerate(zip(parent, sequence, strict=True))
        if left != right
    }
    terminal = set(range(min(10, len(parent)))) | set(
        range(max(len(parent) - 17, 0), len(parent))
    )
    cdr_counts = [len(changed & positions) for positions in region_sets]
    framework = len(changed - set().union(*region_sets))
    return [*map(float, cdr_counts), float(framework), float(len(changed & terminal))]


def feature_matrix(frame: pd.DataFrame, *, length: int) -> np.ndarray:
    """Encode parent context, exact mutation identity, route, and parent metrics."""
    property_count = len(next(iter(AA_PROPERTIES.values())))
    width = (
        2 * length * len(AMINO_ACIDS)
        + length
        + length * property_count
        + len(METRICS)
        + len(PROPOSAL_TYPES)
        + 9
    )
    matrix = np.zeros((len(frame), width), dtype=np.float32)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        sequence = str(row.sequence)
        parent = str(row.parent_sequence)
        if len(sequence) != length or len(parent) != length:
            raise ValueError("all v2 sequences must have the modal length")
        offset = 0
        for position, residue in enumerate(sequence):
            matrix[row_index, offset + position * len(AMINO_ACIDS) + AA_INDEX[residue]] = 1
        offset += length * len(AMINO_ACIDS)
        for position, residue in enumerate(parent):
            matrix[row_index, offset + position * len(AMINO_ACIDS) + AA_INDEX[residue]] = 1
        offset += length * len(AMINO_ACIDS)
        for position, (before, after) in enumerate(zip(parent, sequence, strict=True)):
            if before != after:
                matrix[row_index, offset + position] = 1
        offset += length
        for position, (before, after) in enumerate(zip(parent, sequence, strict=True)):
            before_properties = AA_PROPERTIES[before]
            after_properties = AA_PROPERTIES[after]
            for property_index, (left, right) in enumerate(
                zip(before_properties, after_properties, strict=True)
            ):
                matrix[
                    row_index, offset + position * property_count + property_index
                ] = right - left
        offset += length * property_count
        for metric_index, name in enumerate(METRICS):
            matrix[row_index, offset + metric_index] = float(
                getattr(row, f"parent_{name}")
            )
        offset += len(METRICS)
        proposal_type = str(row.proposal_type)
        if proposal_type in PROPOSAL_TYPES:
            matrix[row_index, offset + PROPOSAL_TYPES.index(proposal_type)] = 1
        offset += len(PROPOSAL_TYPES)
        summary = [
            float(row.edit_distance),
            float(row.proposal_prior_score),
            float(row.parent_selection_score),
            float(row.observation_count),
            *_region_summary(parent, sequence),
        ]
        matrix[row_index, offset : offset + len(summary)] = summary
    return matrix


def prediction_distribution(model: Any, x: np.ndarray) -> tuple[np.ndarray, ...]:
    tree_predictions = np.vstack([tree.predict(x) for tree in model.estimators_])
    mean = tree_predictions.mean(axis=0)
    standard_deviation = tree_predictions.std(axis=0)
    probability_improves = (tree_predictions > 0).mean(axis=0)
    return mean, standard_deviation, probability_improves

# ---------------------------------------------------------------------------


def load_model(path: Path | str) -> Any:
    import joblib

    return joblib.load(str(path))


def rank_single_edits(
    model: Any,
    archive: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    length: int,
) -> list[dict[str, Any]]:
    """Re-order single-edit candidates by predicted improvement (best first).

    ``candidates`` is a list of ``{"sequence", "parent", ...}`` dicts (the
    competitor_lane output shape). ``archive`` is the epoch archive with a
    ``sequence`` column and every ``METRICS`` column. Candidates whose parent is
    absent from the archive (or the wrong length) are dropped -- the model
    cannot encode them -- but every dropped candidate is still emitted, appended
    after the ranked block so no candidate is silently lost.
    """
    if not candidates:
        return candidates

    archive = archive.copy()
    archive["sequence"] = archive["sequence"].astype(str).str.strip().str.upper()
    archive = archive.drop_duplicates("sequence", keep="last")
    archive = archive[archive["sequence"].str.len() == length]
    if archive.empty:
        return candidates

    parent_by_seq = archive.set_index("sequence")
    reference = archive.tail(min(300, len(archive))).reset_index(drop=True)

    # parent-goodness prior == the trainer's fallback for parent_selection_score.
    parent_selection_score: dict[str, float] = {}
    def _parent_prior(parent: str, parent_row: pd.Series) -> float:
        if parent in parent_selection_score:
            return parent_selection_score[parent]
        try:
            metrics = {name: float(parent_row[name]) for name in METRICS}
            value = float(rank_sum_in_field(metrics, reference))
        except (KeyError, TypeError, ValueError):
            value = 0.0
        parent_selection_score[parent] = value
        return value

    encoded_rows: list[dict[str, Any]] = []
    encoded_index: list[int] = []
    for index, cand in enumerate(candidates):
        parent = str(cand.get("parent") or "").strip().upper()
        sequence = str(cand.get("sequence") or "").strip().upper()
        if len(parent) != length or len(sequence) != length:
            continue
        if parent not in parent_by_seq.index:
            continue
        parent_row = parent_by_seq.loc[parent]
        if isinstance(parent_row, pd.DataFrame):
            parent_row = parent_row.iloc[-1]
        row: dict[str, Any] = {
            "sequence": sequence,
            "parent_sequence": parent,
            "proposal_type": "exhaustive_single_substitution",
            "edit_distance": 1,
            "proposal_prior_score": 0.0,
            "parent_selection_score": _parent_prior(parent, parent_row),
            "observation_count": 0,
        }
        for name in METRICS:
            try:
                row[f"parent_{name}"] = float(parent_row[name])
            except (KeyError, TypeError, ValueError):
                row[f"parent_{name}"] = 0.0
        encoded_rows.append(row)
        encoded_index.append(index)

    if not encoded_rows:
        return candidates

    frame = pd.DataFrame(encoded_rows)
    mean, _deviation, probability = prediction_distribution(
        model, feature_matrix(frame, length=length)
    )
    # The model is parent-relative (improvement = parent_rank_sum -
    # child_rank_sum), so rank WITHIN each parent, not globally. Emit in
    # round-robin order: every parent's best edit before any parent's
    # second-best, so no single parent crowds out the others.
    parent_order: list[str] = []
    parent_positions: dict[str, list[int]] = {}
    for i, row in enumerate(encoded_rows):
        parent = str(row["parent_sequence"])
        if parent not in parent_positions:
            parent_order.append(parent)
            parent_positions[parent] = []
        parent_positions[parent].append(i)

    per_parent_sorted: dict[str, list[int]] = {}
    max_len = 0
    for parent in parent_order:
        positions = parent_positions[parent]
        per_parent_sorted[parent] = sorted(
            positions,
            key=lambda i: (probability[i], mean[i]),
            reverse=True,
        )
        max_len = max(max_len, len(positions))

    ranked_index: list[int] = []
    for k in range(max_len):
        for parent in parent_order:
            if k < len(per_parent_sorted[parent]):
                ranked_index.append(encoded_index[per_parent_sorted[parent][k]])

    dropped = [i for i in range(len(candidates)) if i not in set(ranked_index)]
    return [candidates[i] for i in ranked_index] + [candidates[i] for i in dropped]
