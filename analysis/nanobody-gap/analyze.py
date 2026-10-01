#!/usr/bin/env python3
"""Compare our submitted VHHs with the leading miners and the proposal graph.

This is deliberately an offline, read-only analysis.  Miner attribution comes
from the public leaderboard database and is joined to graph nodes by the exact
``(epoch, sequence)`` pair: mutation-graph nodes are sequence identities and do
not themselves carry a submitter identity.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanobody_dash import DEFAULT_COLDKEY
from nanobody_opt.regions import find_regions


TARGET = "P20809"
LOCAL_SOURCES = ("live_exact_screen", "live_exact_repeat", "local_exact_raw")
METRIC_DIRECTIONS = {
    "design_iiptm": "max",
    "design_ptm": "max",
    "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min",
    "interaction_pae": "min",
    "plip_hbonds_refolded": "max",
    "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max",
    "liability_score": "min",
    "liability_num_violations": "min",
}


def rows(connection: sqlite3.Connection, query: str, parameters: tuple = ()) -> list[dict]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(query, parameters)]


def miner_stats(entries: list[dict]) -> list[dict]:
    by_owner: dict[str, dict[str, Any]] = {}
    for entry in entries:
        owner = entry["coldkey"]
        stat = by_owner.setdefault(
            owner,
            {"entries": 0, "wins": 0, "top10": 0, "epochs": collections.Counter()},
        )
        stat["entries"] += 1
        stat["wins"] += entry["rank"] == 1
        stat["top10"] += entry["rank"] <= 10
        stat["epochs"][entry["epoch"]] += 1
    result = []
    for owner, stat in by_owner.items():
        sizes = list(stat["epochs"].values())
        result.append(
            {
                "coldkey": owner,
                "label": "US" if owner == DEFAULT_COLDKEY else owner[:8],
                "ours": owner == DEFAULT_COLDKEY,
                "entries": stat["entries"],
                "wins": stat["wins"],
                "top10": stat["top10"],
                "epochs": len(stat["epochs"]),
                "median_fleet": statistics.median(sizes),
                "win_per_entry": stat["wins"] / stat["entries"],
                "top10_per_entry": stat["top10"] / stat["entries"],
            }
        )
    return sorted(result, key=lambda item: (-item["win_per_entry"], -item["entries"]))


def write_csv(path: Path, records: Iterable[dict]) -> None:
    records = list(records)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def empirical_residue_features(
    entries: list[dict], cohort_labels: dict[str, set[str]], consensus: str
) -> list[dict]:
    """Residue prevalence gaps, preserving every miner as a separate cohort."""
    sequence_length = len(consensus)
    output = []
    ours = [entry for entry in entries if entry["coldkey"] == DEFAULT_COLDKEY]
    for label, owners in cohort_labels.items():
        cohort = [entry for entry in entries if entry["coldkey"] in owners]
        if not cohort:
            continue
        for position in range(sequence_length):
            residues = sorted({entry["sequence"][position] for entry in cohort + ours})
            for residue in residues:
                top_count = sum(entry["sequence"][position] == residue for entry in cohort)
                our_count = sum(entry["sequence"][position] == residue for entry in ours)
                top_rate = top_count / len(cohort)
                our_rate = our_count / len(ours)
                if max(top_rate, our_rate) < 0.01:
                    continue
                cohort_carriers = [
                    entry for entry in cohort if entry["sequence"][position] == residue
                ]
                our_carriers = [
                    entry for entry in ours if entry["sequence"][position] == residue
                ]
                output.append(
                    {
                        "cohort": label,
                        "position_zero": position,
                        "position_one": position + 1,
                        "consensus": consensus[position],
                        "residue": residue,
                        "cohort_count": top_count,
                        "cohort_rate": top_rate,
                        "our_count": our_count,
                        "our_rate": our_rate,
                        "rate_gap": top_rate - our_rate,
                        "cohort_top10_rate": (
                            sum(item["rank"] <= 10 for item in cohort_carriers)
                            / len(cohort_carriers)
                            if cohort_carriers
                            else None
                        ),
                        "our_top10_rate": (
                            sum(item["rank"] <= 10 for item in our_carriers)
                            / len(our_carriers)
                            if our_carriers
                            else None
                        ),
                    }
                )
    return sorted(output, key=lambda item: (-abs(item["rate_gap"]), item["cohort"]))


def haplotype_features(
    entries: list[dict], selected: list[dict], cohort_labels: dict[str, set[str]]
) -> list[dict]:
    """Joint prevalence of the strongest residue features (real haplotypes)."""
    # One residue per position, because overlapping alternatives make the joint
    # pattern opaque.  Keep only features enriched in the leading miners.
    features = []
    seen_positions = set()
    for item in selected:
        position = item["position_zero"]
        if item["rate_gap"] <= 0 or position in seen_positions:
            continue
        features.append((position, item["residue"]))
        seen_positions.add(position)
        if len(features) >= 12:
            break
    ours = [entry for entry in entries if entry["coldkey"] == DEFAULT_COLDKEY]
    output = []
    for label, owners in cohort_labels.items():
        cohort = [entry for entry in entries if entry["coldkey"] in owners]
        for left_index, left in enumerate(features):
            for right in features[left_index + 1 :]:
                def carries(entry: dict) -> bool:
                    return (
                        entry["sequence"][left[0]] == left[1]
                        and entry["sequence"][right[0]] == right[1]
                    )

                top_carriers = [entry for entry in cohort if carries(entry)]
                our_carriers = [entry for entry in ours if carries(entry)]
                top_rate = len(top_carriers) / len(cohort) if cohort else 0.0
                our_rate = len(our_carriers) / len(ours) if ours else 0.0
                output.append(
                    {
                        "cohort": label,
                        "left_position_zero": left[0],
                        "left_residue": left[1],
                        "right_position_zero": right[0],
                        "right_residue": right[1],
                        "cohort_count": len(top_carriers),
                        "cohort_rate": top_rate,
                        "our_count": len(our_carriers),
                        "our_rate": our_rate,
                        "rate_gap": top_rate - our_rate,
                        "cohort_top10_rate": (
                            sum(item["rank"] <= 10 for item in top_carriers)
                            / len(top_carriers)
                            if top_carriers
                            else None
                        ),
                        "our_top10_rate": (
                            sum(item["rank"] <= 10 for item in our_carriers)
                            / len(our_carriers)
                            if our_carriers
                            else None
                        ),
                    }
                )
    return sorted(output, key=lambda item: -item["rate_gap"])


def region_features(
    entries: list[dict], cohort_labels: dict[str, set[str]], consensus: str
) -> list[dict]:
    """Framework/CDR edit burden and literature-motivated CDR3 composition.

    All public sequences in this competition snapshot have the same length, so
    fixed consensus-derived spans avoid turning a mutated anchor into an
    apparent insertion/deletion.  These are competition coordinates rather
    than formal IMGT numbers.
    """
    regions = find_regions(consensus)
    spans = {
        "cdr1": set(range(*regions.cdr1)),
        "cdr2": set(range(*regions.cdr2)),
        "cdr3": set(range(*regions.cdr3)),
    }
    spans["framework"] = set(range(len(consensus))) - set().union(*spans.values())
    output = []
    groups = {"US": {DEFAULT_COLDKEY}, **cohort_labels}
    for label, owners in groups.items():
        cohort = [entry for entry in entries if entry["coldkey"] in owners]
        if not cohort:
            continue
        per_sequence = []
        for entry in cohort:
            sequence = entry["sequence"]
            record = {
                f"{name}_edits": sum(sequence[index] != consensus[index] for index in indices)
                for name, indices in spans.items()
            }
            cdr3 = sequence[regions.cdr3[0] : regions.cdr3[1]]
            record.update(
                {
                    "cdr3_ygsrvi_fraction": sum(residue in "YGSRVI" for residue in cdr3)
                    / len(cdr3),
                    "cdr3_rs_fraction": sum(residue in "RS" for residue in cdr3) / len(cdr3),
                    "cdr3_cm_fraction": sum(residue in "CM" for residue in cdr3) / len(cdr3),
                }
            )
            per_sequence.append(record)
        output.append(
            {
                "cohort": label,
                "entries": len(cohort),
                "cdr1_start_zero": regions.cdr1[0],
                "cdr1_end_exclusive": regions.cdr1[1],
                "cdr2_start_zero": regions.cdr2[0],
                "cdr2_end_exclusive": regions.cdr2[1],
                "cdr3_start_zero": regions.cdr3[0],
                "cdr3_end_exclusive": regions.cdr3[1],
                **{
                    f"median_{field}": statistics.median(item[field] for item in per_sequence)
                    for field in per_sequence[0]
                },
                **{
                    f"mean_{field}": statistics.mean(item[field] for item in per_sequence)
                    for field in per_sequence[0]
                },
            }
        )
    return output


def public_metric_rows(
    graph: sqlite3.Connection, entries: list[dict], cohort_for_owner: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    """Decompose the public rank-sum gap by the ten validator metrics."""
    public = rows(
        graph,
        """
        SELECT n.sequence, o.epoch, o.metrics_json
        FROM observations o JOIN nodes n ON n.id=o.node_id
        WHERE n.target=? AND o.source='public_validator_api'
        """,
        (TARGET,),
    )
    metrics_by_key = {
        (item["epoch"], item["sequence"]): json.loads(item["metrics_json"])
        for item in public
    }
    joined = []
    for entry in entries:
        metrics = metrics_by_key.get((entry["epoch"], entry["sequence"]))
        if not metrics:
            continue
        joined.append({**entry, "cohort": cohort_for_owner.get(entry["coldkey"], "other"), **metrics})

    # Recreate validator per-metric ranks inside each epoch.  Normalising by
    # field size makes epochs with 42 and 53 entries comparable.
    for epoch in sorted({item["epoch"] for item in joined}):
        epoch_rows = [item for item in joined if item["epoch"] == epoch]
        for metric, direction in METRIC_DIRECTIONS.items():
            ordered = sorted(
                epoch_rows,
                key=lambda item: float(item[metric]),
                reverse=(direction == "max"),
            )
            for rank, item in enumerate(ordered, 1):
                item[f"{metric}__rank_fraction"] = rank / len(ordered)

    output = []
    cohorts = sorted({item["cohort"] for item in joined if item["cohort"] != "other"})
    for cohort in cohorts:
        group = [item for item in joined if item["cohort"] == cohort]
        for metric, direction in METRIC_DIRECTIONS.items():
            values = [float(item[metric]) for item in group]
            ranks = [float(item[f"{metric}__rank_fraction"]) for item in group]
            top10_values = [
                float(item[metric]) for item in group if item["rank"] <= 10
            ]
            output.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "direction": direction,
                    "entries": len(values),
                    "raw_median": statistics.median(values),
                    "mean_rank_fraction": statistics.mean(ranks),
                    "top10_raw_median": (
                        statistics.median(top10_values) if top10_values else None
                    ),
                }
            )
    by_key = {(item["cohort"], item["metric"]): item for item in output}
    gaps = []
    for cohort in cohorts:
        if cohort == "US":
            continue
        for metric in METRIC_DIRECTIONS:
            top = by_key[(cohort, metric)]
            ours = by_key[("US", metric)]
            gaps.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "direction": top["direction"],
                    "cohort_mean_rank_fraction": top["mean_rank_fraction"],
                    "our_mean_rank_fraction": ours["mean_rank_fraction"],
                    # Positive means the competitor is better (lower rank fraction).
                    "competitor_advantage": (
                        ours["mean_rank_fraction"] - top["mean_rank_fraction"]
                    ),
                    "cohort_raw_median": top["raw_median"],
                    "our_raw_median": ours["raw_median"],
                }
            )
    return output, sorted(gaps, key=lambda item: -item["competitor_advantage"])


def graph_coverage(
    graph: sqlite3.Connection,
    entries: list[dict],
    cohort_for_owner: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    node_rows = rows(
        graph,
        "SELECT id, sequence FROM nodes WHERE target=? AND length=127",
        (TARGET,),
    )
    node_by_sequence = {item["sequence"]: item["id"] for item in node_rows}
    local_min_epoch = {
        int(node_id): int(epoch)
        for node_id, epoch in graph.execute(
            """
            SELECT node_id, MIN(epoch) FROM observations
            WHERE source='live_exact_screen' AND epoch IS NOT NULL
            GROUP BY node_id
            """
        )
    }
    lineage_min_epoch = {
        int(node_id): int(epoch)
        for node_id, epoch in graph.execute(
            """
            SELECT child_node_id, MIN(epoch) FROM lineage_edges
            WHERE source='live_exact_screen' AND epoch >= 0
            GROUP BY child_node_id
            """
        )
    }
    adjacency: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for source, target, distance in graph.execute(
        "SELECT source_node_id,target_node_id,distance FROM edges"
    ):
        adjacency[int(source)].append((int(target), int(distance)))
        adjacency[int(target)].append((int(source), int(distance)))

    winners = {
        entry["epoch"]: entry["sequence"] for entry in entries if entry["rank"] == 1
    }
    winner_epochs = sorted(winners)
    winner_vectors = {
        epoch: np.frombuffer(winners[epoch].encode("ascii"), dtype="S1")
        for epoch in winner_epochs
    }
    detail = []
    for entry in entries:
        node_id = node_by_sequence.get(entry["sequence"])
        exact_epoch = local_min_epoch.get(node_id) if node_id is not None else None
        lineage_epoch = lineage_min_epoch.get(node_id) if node_id is not None else None
        nearest = None
        if exact_epoch is not None and exact_epoch <= entry["epoch"]:
            nearest = 0
        elif node_id is not None:
            eligible = [
                distance
                for other, distance in adjacency.get(node_id, [])
                if local_min_epoch.get(other, 10**9) <= entry["epoch"]
            ]
            nearest = min(eligible) if eligible else None
        vector = np.frombuffer(entry["sequence"].encode("ascii"), dtype="S1")
        same_winner_distance = int(np.count_nonzero(vector != winner_vectors[entry["epoch"]]))
        previous = [epoch for epoch in winner_epochs if epoch < entry["epoch"]]
        previous_winner_distance = None
        if previous:
            previous_winner_distance = min(
                int(np.count_nonzero(vector != winner_vectors[epoch])) for epoch in previous
            )
        detail.append(
            {
                "epoch": entry["epoch"],
                "uid": entry["uid"],
                "coldkey_label": (
                    "US" if entry["coldkey"] == DEFAULT_COLDKEY else entry["coldkey"][:8]
                ),
                "cohort": cohort_for_owner.get(entry["coldkey"], "other"),
                "rank": entry["rank"],
                "score": entry["score"],
                "graph_node": node_id is not None,
                "exact_screen_before": exact_epoch is not None and exact_epoch <= entry["epoch"],
                "exact_screen_any": exact_epoch is not None,
                "lineage_before": lineage_epoch is not None and lineage_epoch <= entry["epoch"],
                "nearest_exact_graph_distance_before": nearest,
                "same_epoch_winner_distance": same_winner_distance,
                "previous_winner_distance": previous_winner_distance,
            }
        )

    summary = []
    labels = sorted({item["cohort"] for item in detail if item["cohort"] != "other"})
    for label in labels:
        group = [item for item in detail if item["cohort"] == label]
        distances = [
            item["nearest_exact_graph_distance_before"]
            for item in group
            if item["nearest_exact_graph_distance_before"] is not None
        ]
        previous = [item["previous_winner_distance"] for item in group if item["previous_winner_distance"] is not None]
        summary.append(
            {
                "cohort": label,
                "entries": len(group),
                "graph_node_rate": sum(item["graph_node"] for item in group) / len(group),
                "exact_screen_before_rate": sum(item["exact_screen_before"] for item in group) / len(group),
                "exact_screen_any_rate": sum(item["exact_screen_any"] for item in group) / len(group),
                "lineage_before_rate": sum(item["lineage_before"] for item in group) / len(group),
                "within_one_before_rate": sum(
                    item["nearest_exact_graph_distance_before"] is not None
                    and item["nearest_exact_graph_distance_before"] <= 1
                    for item in group
                ) / len(group),
                "within_two_before_rate": sum(
                    item["nearest_exact_graph_distance_before"] is not None
                    and item["nearest_exact_graph_distance_before"] <= 2
                    for item in group
                ) / len(group),
                "within_six_before_rate": sum(
                    item["nearest_exact_graph_distance_before"] is not None
                    and item["nearest_exact_graph_distance_before"] <= 6
                    for item in group
                ) / len(group),
                "resolved_nearest_median": statistics.median(distances) if distances else None,
                "same_winner_distance_median": statistics.median(
                    item["same_epoch_winner_distance"] for item in group
                ),
                "previous_winner_distance_median": statistics.median(previous) if previous else None,
            }
        )
    return detail, summary


def exact_top_entry_distances(
    graph: sqlite3.Connection,
    entries: list[dict],
    leading_owners: set[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Exact Hamming distance from top-miner top entries to our prior screens.

    The persisted graph is deliberately bounded to sixteen neighbours, so it is
    an efficient coverage diagnostic but cannot certify that an unresolved
    sequence is farther than six edits.  The highest-value entries are few
    enough to compare directly against every sequence we had exact-screened by
    that epoch.
    """
    local = rows(
        graph,
        """
        SELECT n.sequence, MIN(o.epoch) AS first_epoch
        FROM observations o JOIN nodes n ON n.id=o.node_id
        WHERE n.target=? AND n.length=127 AND o.source='live_exact_screen'
        GROUP BY n.id,n.sequence
        """,
        (TARGET,),
    )
    local_vectors = np.asarray(
        [[ord(residue) for residue in item["sequence"]] for item in local],
        dtype=np.uint8,
    )
    local_epochs = np.asarray([item["first_epoch"] for item in local], dtype=np.int32)
    local_sequences = [item["sequence"] for item in local]
    regions = find_regions(local_sequences[0])

    def region_for(position: int) -> str:
        for label, span in (
            ("cdr1", regions.cdr1),
            ("cdr2", regions.cdr2),
            ("cdr3", regions.cdr3),
        ):
            if span[0] <= position < span[1]:
                return label
        return "framework"
    selected = [
        entry for entry in entries
        if entry["coldkey"] in leading_owners and entry["rank"] <= 10
    ]
    detail = []
    for entry in selected:
        eligible_indices = np.flatnonzero(local_epochs <= entry["epoch"])
        eligible = local_vectors[eligible_indices]
        query = np.frombuffer(entry["sequence"].encode("ascii"), dtype=np.uint8)
        distances = np.count_nonzero(eligible != query, axis=1)
        distance = int(distances.min())
        nearest_indices = eligible_indices[np.flatnonzero(distances == distance)]
        # A nearest sequence first seen most recently is the most faithful view
        # of the search state immediately before submission.  Lexicographic
        # order only makes ties deterministic.
        nearest_index = max(
            (int(index) for index in nearest_indices),
            key=lambda index: (int(local_epochs[index]), local_sequences[index]),
        )
        nearest_sequence = local_sequences[nearest_index]
        mutations = [
            {
                "position_zero": position,
                "position_one": position + 1,
                "from": nearest_sequence[position],
                "to": entry["sequence"][position],
                "region": region_for(position),
            }
            for position in range(len(entry["sequence"]))
            if nearest_sequence[position] != entry["sequence"][position]
        ]
        detail.append(
            {
                "epoch": entry["epoch"],
                "miner": entry["coldkey"][:8],
                "uid": entry["uid"],
                "public_rank": entry["rank"],
                "public_score": entry["score"],
                "nearest_prior_exact_hamming": distance,
                "nearest_prior_exact_first_epoch": int(local_epochs[nearest_index]),
                "nearest_prior_exact_sequence": nearest_sequence,
                "missing_mutations": json.dumps(mutations, separators=(",", ":")),
            }
        )
    summary = []
    for label, subset in (
        ("top10", detail),
        ("wins", [item for item in detail if item["public_rank"] == 1]),
    ):
        distances = [item["nearest_prior_exact_hamming"] for item in subset]
        summary.append(
            {
                "subset": label,
                "entries": len(subset),
                "median": statistics.median(distances) if distances else None,
                "p90": float(np.quantile(distances, 0.9)) if distances else None,
                "within_zero_rate": sum(value == 0 for value in distances) / len(distances) if distances else None,
                "within_one_rate": sum(value <= 1 for value in distances) / len(distances) if distances else None,
                "within_two_rate": sum(value <= 2 for value in distances) / len(distances) if distances else None,
                "within_six_rate": sum(value <= 6 for value in distances) / len(distances) if distances else None,
            }
        )
    edit_summary = []
    for label, subset in (
        ("top10", detail),
        ("wins", [item for item in detail if item["public_rank"] == 1]),
    ):
        counts: collections.Counter[tuple[int, str, str, str]] = collections.Counter()
        for item in subset:
            for mutation in json.loads(item["missing_mutations"]):
                counts[
                    (
                        mutation["position_zero"],
                        mutation["from"],
                        mutation["to"],
                        mutation["region"],
                    )
                ] += 1
        for (position, source, target, region), count in counts.most_common():
            edit_summary.append(
                {
                    "subset": label,
                    "position_zero": position,
                    "position_one": position + 1,
                    "from": source,
                    "to": target,
                    "region": region,
                    "count": count,
                    "entry_rate": count / len(subset) if subset else None,
                }
            )
    return detail, summary, edit_summary


def replay_cohort(entries: list[dict], leading_owners: list[str], limit_epochs: int = 12) -> list[dict]:
    """Pre-register matched sequences for a future repeated exact replay.

    Selection uses the public result only to define the question explicitly:
    each recent epoch contributes the leading miners' best observed entry and
    our best observed entry.  Repeated exact scoring must report the selection
    bias (winner's curse); it cannot be called prospective validation.
    """
    epochs = sorted({entry["epoch"] for entry in entries}, reverse=True)
    result = []
    for epoch in epochs:
        ours = [
            entry for entry in entries
            if entry["epoch"] == epoch and entry["coldkey"] == DEFAULT_COLDKEY
        ]
        leaders = [
            entry for entry in entries
            if entry["epoch"] == epoch and entry["coldkey"] in set(leading_owners)
        ]
        if not ours or not leaders:
            continue
        for cohort, item in (
            ("US", min(ours, key=lambda row: row["rank"])),
            ("TOP_MINERS", min(leaders, key=lambda row: row["rank"])),
        ):
            result.append(
                {
                    "epoch": epoch,
                    "cohort": cohort,
                    "uid": item["uid"],
                    "public_rank": item["rank"],
                    "public_score": item["score"],
                    "sequence": item["sequence"],
                    "selection_rule": "best_public_entry_within_epoch_and_cohort",
                }
            )
        if len({item["epoch"] for item in result}) >= limit_epochs:
            break
    return sorted(result, key=lambda item: (item["epoch"], item["cohort"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", type=Path, default=Path("data/nanobody_dash.sqlite"))
    parser.add_argument("--graph", type=Path, default=Path("analysis/nanobody-gap/mutations.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("analysis/nanobody-gap/results"))
    args = parser.parse_args()

    leaderboard = sqlite3.connect(args.leaderboard)
    graph = sqlite3.connect(args.graph)
    entries = rows(leaderboard, "SELECT * FROM entries WHERE sequence IS NOT NULL ORDER BY epoch,rank")
    stats = miner_stats(entries)
    ours = next(item for item in stats if item["ours"])
    active = [item for item in stats if item["entries"] >= 300 and not item["ours"]]
    leading = [item for item in active if item["win_per_entry"] > ours["win_per_entry"]][:5]
    core = leading[:3]
    cohort_labels = {
        "TOP3": {item["coldkey"] for item in core},
        **{item["label"]: {item["coldkey"]} for item in core},
    }
    cohort_for_owner = {DEFAULT_COLDKEY: "US"}
    cohort_for_owner.update({item["coldkey"]: "TOP3" for item in core})
    individual_cohort_for_owner = {DEFAULT_COLDKEY: "US"}
    individual_cohort_for_owner.update(
        {item["coldkey"]: item["label"] for item in core}
    )

    sequences = [entry["sequence"] for entry in entries]
    consensus = "".join(
        collections.Counter(sequence[position] for sequence in sequences).most_common(1)[0][0]
        for position in range(len(sequences[0]))
    )
    features = empirical_residue_features(entries, cohort_labels, consensus)
    top3_features = [item for item in features if item["cohort"] == "TOP3"]
    haplotypes = haplotype_features(entries, top3_features, cohort_labels)
    regions = region_features(entries, cohort_labels, consensus)
    metric_rows, metric_gaps = public_metric_rows(graph, entries, cohort_for_owner)
    individual_metric_rows, individual_metric_gaps = public_metric_rows(
        graph, entries, individual_cohort_for_owner
    )
    metric_rows.extend(
        item for item in individual_metric_rows if item["cohort"] != "US"
    )
    metric_gaps.extend(individual_metric_gaps)
    coverage_detail, coverage_summary = graph_coverage(graph, entries, cohort_for_owner)
    exact_distance_detail, exact_distance_summary, missing_edit_summary = exact_top_entry_distances(
        graph, entries, {item["coldkey"] for item in core}
    )
    replay = replay_cohort(entries, [item["coldkey"] for item in core])

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "miner_stats.csv", stats)
    write_csv(args.output / "residue_signatures.csv", features)
    write_csv(args.output / "haplotype_signatures.csv", haplotypes)
    write_csv(args.output / "region_signatures.csv", regions)
    write_csv(args.output / "public_metric_summary.csv", metric_rows)
    write_csv(args.output / "public_metric_gaps.csv", metric_gaps)
    write_csv(args.output / "graph_coverage_detail.csv", coverage_detail)
    write_csv(args.output / "graph_coverage_summary.csv", coverage_summary)
    write_csv(args.output / "exact_top_entry_distances.csv", exact_distance_detail)
    write_csv(args.output / "nearest_missing_edits.csv", missing_edit_summary)
    (args.output / "replay_cohort.json").write_text(json.dumps(replay, indent=2) + "\n")
    report = {
        "epochs": [min(entry["epoch"] for entry in entries), max(entry["epoch"] for entry in entries)],
        "entries": len(entries),
        "consensus": consensus,
        "ours": ours,
        "leading_miners": leading,
        "core_top_miners": core,
        "graph_coverage": coverage_summary,
        "exact_top_entry_distance": exact_distance_summary,
        "top_residue_gaps": top3_features[:20],
        "top_haplotype_gaps": [item for item in haplotypes if item["cohort"] == "TOP3"][:20],
        "region_signatures": regions,
        "top_nearest_missing_edits": [
            item for item in missing_edit_summary if item["subset"] == "top10"
        ][:20],
        "top_metric_gaps": [item for item in metric_gaps if item["cohort"] == "TOP3"],
        "replay_pairs": len(replay) // 2,
        "attribution": "exact leaderboard (epoch,sequence) join; graph provenance has no coldkey/hotkey",
        "nearest_distance_caveat": "edges are a bounded 16-neighbour graph with maximum Hamming distance 6; unresolved means no stored eligible neighbour, not a proven distance >6",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "entries": len(entries),
        "top_miners": [item["label"] for item in core],
        "replay_pairs": len(replay) // 2,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
