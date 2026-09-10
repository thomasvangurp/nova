"""Scored vector index and mutation graph for historical VHH sequences."""
from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nanobody_opt.regions import find_regions


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
AA_PROPERTY = {
    **dict.fromkeys("AVILMFWY", "hydrophobic"),
    **dict.fromkeys("STNQ", "polar"),
    **dict.fromkeys("KRH", "positive"),
    **dict.fromkeys("DE", "negative"),
    **dict.fromkeys("CGP", "special"),
}


@dataclass(frozen=True)
class MutationNode:
    sequence: str
    score: float
    observations: int
    sources: tuple[str, ...]
    epochs: tuple[int, ...]


@dataclass(frozen=True)
class MutationNeighbour:
    sequence: str
    score: float
    distance: int
    adjusted_score: float
    mutations: tuple[str, ...]
    observations: int
    sources: tuple[str, ...]
    supported_effect_count: int = 0
    effect_uncertainty: float = 0.0


def mutations(left: str, right: str) -> tuple[str, ...]:
    if len(left) != len(right):
        raise ValueError("mutation endpoints have different lengths")
    return tuple(
        f"{before}{position + 1}{after}"
        for position, (before, after) in enumerate(
            zip(left, right, strict=True)
        )
        if before != after
    )


def mutation_contexts(
    left: str,
    right: str,
    *,
    contact_positions: Iterable[int] = (),
) -> list[dict[str, Any]]:
    """Describe substitutions with stable VHH-region and chemistry features."""
    if len(left) != len(right):
        raise ValueError("mutation endpoints have different lengths")
    contacts = set(contact_positions)
    try:
        regions = find_regions(left)
        region_by_position = {
            position: name
            for name, bounds in (
                ("cdr1", regions.cdr1),
                ("cdr2", regions.cdr2),
                ("cdr3", regions.cdr3),
            )
            for position in range(*bounds)
        }
    except ValueError:
        region_by_position = {}
    contexts = []
    for position, (before, after) in enumerate(zip(left, right, strict=True)):
        if before == after:
            continue
        terminal = (
            "n_terminal" if position < 10 else
            "c_terminal" if position >= len(left) - 17 else
            "internal"
        )
        contexts.append({
            "position": position + 1,
            "from_residue": before,
            "to_residue": after,
            "region": region_by_position.get(position, "framework"),
            "terminal_class": terminal,
            "contact_class": "contact" if position in contacts else "unknown",
            "property_change": (
                "conservative"
                if AA_PROPERTY[before] == AA_PROPERTY[after]
                else f"{AA_PROPERTY[before]}_to_{AA_PROPERTY[after]}"
            ),
        })
    return contexts


class MutationNetwork:
    """Compact position-vector store with virtual-node nearest neighbours."""

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        *,
        mutation_effects: Iterable[dict[str, Any]] = (),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            sequence = str(record.get("sequence", "")).strip().upper()
            try:
                score = float(record["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not sequence
                or not math.isfinite(score)
                or any(residue not in AA_INDEX for residue in sequence)
            ):
                continue
            grouped[sequence].append({**record, "score": score})
        if not grouped:
            raise ValueError("mutation network has no valid scored sequences")
        lengths = {len(sequence) for sequence in grouped}
        if len(lengths) != 1:
            raise ValueError("mutation network requires one sequence length")
        self.length = lengths.pop()
        self.nodes: list[MutationNode] = []
        for sequence, observations in grouped.items():
            public = [
                item for item in observations
                if item.get("score_kind") == "validator_final_rank_sum"
            ]
            scoring_observations = public or observations
            scores = sorted(float(item["score"]) for item in scoring_observations)
            sources = tuple(sorted({
                str(item.get("source", "unknown")) for item in observations
            }))
            epochs = tuple(sorted({
                int(item["epoch"])
                for item in observations
                if item.get("epoch") is not None
            }))
            # Preserve an elite observation without letting a single lucky
            # score fully dominate repeated evidence.
            robust_score = 0.5 * scores[0] + 0.5 * float(np.median(scores))
            self.nodes.append(MutationNode(
                sequence=sequence,
                score=robust_score,
                observations=sum(
                    int(item.get("observation_count", 1))
                    for item in scoring_observations
                ),
                sources=sources,
                epochs=epochs,
            ))
        self.nodes.sort(key=lambda node: (node.score, node.sequence))
        self._by_sequence = {node.sequence: node for node in self.nodes}
        self._contacts: dict[str, set[int]] = {}
        for sequence, observations in grouped.items():
            contacts = set()
            for item in observations:
                pose = item.get("pose_stability") or {}
                contacts.update(
                    int(position) - 1
                    for position in pose.get(
                        "consensus_nanobody_contact_positions", []
                    )
                    if 1 <= int(position) <= len(sequence)
                )
            self._contacts[sequence] = contacts
        self._vectors = np.asarray([
            [AA_INDEX[residue] for residue in node.sequence]
            for node in self.nodes
        ], dtype=np.uint8)
        self._effects: dict[tuple[int, str, str, str, str], dict[str, Any]] = {}
        for effect in mutation_effects:
            key = (
                int(effect["position"]),
                str(effect["from_residue"]),
                str(effect["to_residue"]),
                str(effect["region"]),
                str(effect["contact_class"]),
            )
            self._effects[key] = effect

    def _prospective_adjustment(
        self,
        source: str,
        candidate: str,
        *,
        distance_penalty: float,
    ) -> tuple[float, int, float]:
        adjustment = 0.0
        supported = 0
        uncertainty_squared = 0.0
        for context in mutation_contexts(
            source,
            candidate,
            contact_positions=self._contacts.get(source, set()),
        ):
            key = (
                int(context["position"]),
                str(context["from_residue"]),
                str(context["to_residue"]),
                str(context["region"]),
                str(context["contact_class"]),
            )
            effect = self._effects.get(key)
            if effect is None:
                adjustment += distance_penalty
                continue
            support = int(effect["support"])
            shrinkage = support / (support + 3.0)
            uncertainty = float(effect["uncertainty"])
            adjustment += (
                shrinkage * float(effect["median_score_delta"])
                + uncertainty / math.sqrt(max(support, 1))
            )
            supported += 1
            uncertainty_squared += uncertainty**2
        return adjustment, supported, math.sqrt(uncertainty_squared)

    def neighbours(
        self,
        sequence: str,
        *,
        limit: int = 16,
        maximum_distance: int | None = 6,
        distance_penalty: float = 5.0,
    ) -> list[MutationNeighbour]:
        sequence = sequence.strip().upper()
        if len(sequence) != self.length or any(
            residue not in AA_INDEX for residue in sequence
        ):
            raise ValueError("query is incompatible with mutation network")
        if limit < 1 or distance_penalty < 0:
            raise ValueError("invalid neighbour query configuration")
        vector = np.asarray([AA_INDEX[residue] for residue in sequence], dtype=np.uint8)
        distances = np.count_nonzero(self._vectors != vector, axis=1)
        baseline_order = sorted(
            range(len(self.nodes)),
            key=lambda index: (
                self.nodes[index].score + distance_penalty * int(distances[index]),
                int(distances[index]),
                self.nodes[index].sequence,
            ),
        )
        compatible = [
            index for index in baseline_order
            if maximum_distance is None
            or int(distances[index]) <= maximum_distance
        ]
        # Refine a generous vector shortlist with contextual directional
        # effects. This keeps virtual-node queries cheap even after thousands
        # of historical sequences have been ingested.
        refined = []
        for index in compatible[: max(256, limit * 16)]:
            distance = int(distances[index])
            node = self.nodes[index]
            adjustment, supported, uncertainty = self._prospective_adjustment(
                node.sequence,
                sequence,
                distance_penalty=distance_penalty,
            )
            refined.append((
                node.score + adjustment,
                distance,
                node.sequence,
                index,
                supported,
                uncertainty,
            ))
        refined.sort()
        found = []
        for adjusted_score, distance, _, index, supported, uncertainty in refined:
            node = self.nodes[index]
            found.append(MutationNeighbour(
                sequence=node.sequence,
                score=node.score,
                distance=distance,
                adjusted_score=adjusted_score,
                mutations=mutations(node.sequence, sequence),
                observations=node.observations,
                sources=node.sources,
                supported_effect_count=supported,
                effect_uncertainty=uncertainty,
            ))
            if len(found) >= limit:
                break
        return found

    def evidence(
        self,
        sequence: str,
        *,
        limit: int = 16,
        maximum_distance: int | None = 6,
        distance_penalty: float = 5.0,
        consensus_size: int = 5,
    ) -> dict[str, Any]:
        neighbours = self.neighbours(
            sequence,
            limit=limit,
            maximum_distance=maximum_distance,
            distance_penalty=distance_penalty,
        )
        if not neighbours:
            neighbours = self.neighbours(
                sequence,
                limit=limit,
                maximum_distance=None,
                distance_penalty=distance_penalty,
            )
        consensus = neighbours[: max(1, consensus_size)]
        return {
            "predecessor_min_distance": min(item.distance for item in neighbours),
            "predecessor_potential_score": neighbours[0].adjusted_score,
            "prospective_score": neighbours[0].adjusted_score,
            "prospective_effect_support": (
                neighbours[0].supported_effect_count
            ),
            "prospective_uncertainty": neighbours[0].effect_uncertainty,
            "predecessor_consensus_score": float(np.mean([
                item.adjusted_score for item in consensus
            ])),
            "predecessor_support_within_six": sum(
                item.distance <= 6 for item in neighbours
            ),
            "predecessor_neighbours": [asdict(item) for item in neighbours],
        }

    def graph(self, *, edge_distance: int = 6, edge_limit: int = 16) -> dict[str, Any]:
        edges = []
        seen: set[tuple[str, str]] = set()
        for node in self.nodes:
            for neighbour in self.neighbours(
                node.sequence,
                limit=edge_limit + 1,
                maximum_distance=edge_distance,
            ):
                if neighbour.distance == 0:
                    continue
                endpoints = tuple(sorted((node.sequence, neighbour.sequence)))
                if endpoints in seen:
                    continue
                seen.add(endpoints)
                left = self._by_sequence[endpoints[0]]
                right = self._by_sequence[endpoints[1]]
                edges.append({
                    "left_sequence": endpoints[0],
                    "right_sequence": endpoints[1],
                    "distance": neighbour.distance,
                    "mutations_left_to_right": mutations(endpoints[0], endpoints[1]),
                    "score_delta_right_minus_left": right.score - left.score,
                })
        return {
            "schema_version": 1,
            "sequence_length": self.length,
            "node_count": len(self.nodes),
            "edge_count": len(edges),
            "nodes": [asdict(node) for node in self.nodes],
            "edges": edges,
        }

    def save(self, path: Path, *, edge_distance: int = 6, edge_limit: int = 16) -> None:
        payload = self.graph(edge_distance=edge_distance, edge_limit=edge_limit)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY,
            target TEXT NOT NULL,
            sequence TEXT NOT NULL,
            length INTEGER NOT NULL,
            residue_vector BLOB NOT NULL,
            UNIQUE(target, sequence)
        );
        CREATE TABLE IF NOT EXISTS observations (
            observation_id TEXT PRIMARY KEY,
            node_id INTEGER NOT NULL REFERENCES nodes(id),
            epoch INTEGER,
            source TEXT NOT NULL,
            score_kind TEXT NOT NULL,
            score REAL NOT NULL,
            observation_count INTEGER NOT NULL DEFAULT 1,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            pose_json TEXT,
            provenance_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS observations_node_epoch
            ON observations(node_id, epoch);
        CREATE TABLE IF NOT EXISTS edges (
            source_node_id INTEGER NOT NULL REFERENCES nodes(id),
            target_node_id INTEGER NOT NULL REFERENCES nodes(id),
            distance INTEGER NOT NULL,
            mutations_json TEXT NOT NULL,
            context_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(source_node_id, target_node_id)
        );
        CREATE TABLE IF NOT EXISTS mutation_effects (
            target TEXT NOT NULL,
            position INTEGER NOT NULL,
            from_residue TEXT NOT NULL,
            to_residue TEXT NOT NULL,
            region TEXT NOT NULL,
            contact_class TEXT NOT NULL,
            support INTEGER NOT NULL,
            median_score_delta REAL NOT NULL,
            uncertainty REAL NOT NULL,
            graph_version TEXT NOT NULL,
            PRIMARY KEY(
                target, position, from_residue, to_residue,
                region, contact_class, graph_version
            )
        );
        CREATE TABLE IF NOT EXISTS proposal_scores (
            epoch INTEGER NOT NULL,
            node_id INTEGER NOT NULL REFERENCES nodes(id),
            graph_version TEXT NOT NULL,
            prospective_score REAL NOT NULL,
            prospective_rank INTEGER NOT NULL,
            features_json TEXT NOT NULL,
            PRIMARY KEY(epoch, node_id, graph_version)
        );
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', '1')"
    )
    return connection


def update_observation_store(
    path: Path, observations: Iterable[dict[str, Any]]
) -> int:
    """Transactionally upsert uniquely identified scored VHH observations."""
    written = 0
    with _connect(path) as connection:
        for observation in observations:
            observation_id = str(observation.get("observation_id", "")).strip()
            sequence = str(observation.get("sequence", "")).strip().upper()
            target = str(observation.get("target", "")).strip()
            try:
                score = float(observation["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not observation_id
                or not sequence
                or not target
                or not math.isfinite(score)
                or any(residue not in AA_INDEX for residue in sequence)
            ):
                continue
            vector = bytes(AA_INDEX[residue] for residue in sequence)
            connection.execute(
                """
                INSERT INTO nodes(target, sequence, length, residue_vector)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(target, sequence) DO UPDATE SET
                    length=excluded.length,
                    residue_vector=excluded.residue_vector
                """,
                (target, sequence, len(sequence), vector),
            )
            node_id = int(connection.execute(
                "SELECT id FROM nodes WHERE target=? AND sequence=?",
                (target, sequence),
            ).fetchone()[0])
            excluded = {
                "observation_id", "sequence", "target", "epoch", "source",
                "score_kind", "score", "observation_count", "metrics",
                "pose_stability",
            }
            provenance = {
                key: value for key, value in observation.items() if key not in excluded
            }
            connection.execute(
                """
                INSERT INTO observations(
                    observation_id, node_id, epoch, source, score_kind, score,
                    observation_count, metrics_json, pose_json, provenance_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_id) DO UPDATE SET
                    node_id=excluded.node_id,
                    epoch=excluded.epoch,
                    source=excluded.source,
                    score_kind=excluded.score_kind,
                    score=excluded.score,
                    observation_count=excluded.observation_count,
                    metrics_json=excluded.metrics_json,
                    pose_json=excluded.pose_json,
                    provenance_json=excluded.provenance_json
                """,
                (
                    observation_id,
                    node_id,
                    observation.get("epoch"),
                    str(observation.get("source", "unknown")),
                    str(observation.get("score_kind", "unknown")),
                    score,
                    int(observation.get("observation_count", 1)),
                    json.dumps(observation.get("metrics") or {}, sort_keys=True),
                    json.dumps(observation.get("pose_stability"), sort_keys=True)
                    if observation.get("pose_stability") is not None else None,
                    json.dumps(provenance, sort_keys=True),
                ),
            )
            written += 1
    return written


def load_observations(path: Path, *, target: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT n.sequence, o.score, o.epoch, o.source, o.score_kind,
                   o.observation_count, o.metrics_json, o.pose_json
            FROM observations o
            JOIN nodes n ON n.id=o.node_id
            WHERE n.target=?
            ORDER BY o.epoch, o.observation_id
            """,
            (target,),
        ).fetchall()
    return [
        {
            "sequence": row[0],
            "score": row[1],
            "epoch": row[2],
            "source": row[3],
            "score_kind": row[4],
            "observation_count": row[5],
            "metrics": json.loads(row[6]),
            "pose_stability": json.loads(row[7]) if row[7] else None,
        }
        for row in rows
    ]


def load_mutation_effects(path: Path, *, target: str) -> list[dict[str, Any]]:
    """Load the latest complete directional-effect snapshot for a target."""
    if not path.is_file():
        return []
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key=?",
            (f"graph_version:{target}",),
        ).fetchone()
        if row is None:
            return []
        graph_version = str(row[0])
        rows = connection.execute(
            """
            SELECT position, from_residue, to_residue, region, contact_class,
                   support, median_score_delta, uncertainty, graph_version
            FROM mutation_effects
            WHERE target=? AND graph_version=?
            """,
            (target, graph_version),
        ).fetchall()
    return [
        {
            "position": row[0],
            "from_residue": row[1],
            "to_residue": row[2],
            "region": row[3],
            "contact_class": row[4],
            "support": row[5],
            "median_score_delta": row[6],
            "uncertainty": row[7],
            "graph_version": row[8],
        }
        for row in rows
    ]


def _robust_node_scores(
    connection: sqlite3.Connection,
    *,
    target: str,
    sequence_length: int,
) -> tuple[list[int], list[str], np.ndarray, dict[str, float]]:
    rows = connection.execute(
        """
        SELECT n.id, n.sequence, n.residue_vector, o.score, o.score_kind,
               o.epoch, o.source, o.observation_count
        FROM nodes n
        JOIN observations o ON o.node_id=n.id
        WHERE n.target=? AND n.length=?
        ORDER BY n.id, o.observation_id
        """,
        (target, sequence_length),
    ).fetchall()
    records = [
        {
            "sequence": row[1],
            "score": row[3],
            "score_kind": row[4],
            "epoch": row[5],
            "source": row[6],
            "observation_count": row[7],
        }
        for row in rows
    ]
    if not records:
        raise ValueError(f"mutation database has no observations for {target}")
    network = MutationNetwork(records)
    score_by_sequence = {node.sequence: node.score for node in network.nodes}
    first_by_id: dict[int, tuple[str, bytes]] = {}
    for node_id, sequence, vector, *_ in rows:
        first_by_id.setdefault(int(node_id), (str(sequence), bytes(vector)))
    node_ids = sorted(first_by_id)
    sequences = [first_by_id[node_id][0] for node_id in node_ids]
    vectors = np.frombuffer(
        b"".join(first_by_id[node_id][1] for node_id in node_ids),
        dtype=np.uint8,
    ).reshape(len(node_ids), sequence_length)
    return node_ids, sequences, vectors, score_by_sequence


def rebuild_nearest_edges(
    path: Path,
    *,
    target: str,
    edge_limit: int = 16,
    maximum_distance: int = 6,
    workers: int = 16,
) -> dict[str, Any]:
    """Rebuild a bounded nearest-neighbour graph and empirical mutation effects."""
    if edge_limit < 1 or maximum_distance < 1 or workers < 1:
        raise ValueError("invalid nearest-edge rebuild configuration")
    with _connect(path) as connection:
        edge_rows: dict[tuple[int, int], tuple[int, int, int, str, str]] = {}
        scores: dict[str, float] = {}
        contact_rows = connection.execute(
            """
            SELECT n.id, o.pose_json FROM nodes n
            JOIN observations o ON o.node_id=n.id
            WHERE n.target=? AND o.pose_json IS NOT NULL
            """,
            (target,),
        ).fetchall()
        contacts_by_node_id: dict[int, set[int]] = defaultdict(set)
        for node_id, pose_json in contact_rows:
            pose = json.loads(pose_json)
            contacts_by_node_id[int(node_id)].update(
                int(position) - 1
                for position in pose.get(
                    "consensus_nanobody_contact_positions", []
                )
                if int(position) >= 1
            )
        lengths = connection.execute(
            """
            SELECT length, COUNT(*) FROM nodes
            WHERE target=? GROUP BY length ORDER BY length
            """,
            (target,),
        ).fetchall()
        if not lengths:
            raise ValueError(f"mutation database has no observations for {target}")
        node_count = 0
        for sequence_length, _ in lengths:
            node_ids, sequences, vectors, partition_scores = _robust_node_scores(
                connection,
                target=target,
                sequence_length=int(sequence_length),
            )
            node_count += len(node_ids)
            scores.update(partition_scores)
            neighbour_count = min(edge_limit + 1, len(node_ids))

            # Each worker handles independent query rows. NumPy releases the
            # GIL in the comparison/count, while the query-row design bounds
            # peak RAM even for a large historical bank.
            def nearest(index: int) -> list[tuple[int, int]]:
                distances = np.count_nonzero(vectors != vectors[index], axis=1)
                if neighbour_count == len(node_ids):
                    candidates = np.arange(len(node_ids))
                else:
                    candidates = np.argpartition(
                        distances, neighbour_count - 1
                    )[:neighbour_count]
                ordered = sorted(
                    (
                        (int(distances[other]), int(other))
                        for other in candidates
                        if other != index
                        and int(distances[other]) <= maximum_distance
                    ),
                    key=lambda item: (item[0], sequences[item[1]]),
                )
                return ordered[:edge_limit]

            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(nearest, range(len(node_ids)))
                for index, neighbours in enumerate(results):
                    for distance, other in neighbours:
                        left_index, right_index = sorted((index, other))
                        key = (node_ids[left_index], node_ids[right_index])
                        left = sequences[left_index]
                        right = sequences[right_index]
                        edge_rows[key] = (
                            key[0],
                            key[1],
                            distance,
                            json.dumps(mutations(left, right)),
                            json.dumps(
                                mutation_contexts(
                                    left,
                                    right,
                                    contact_positions=(
                                        contacts_by_node_id[key[0]]
                                        | contacts_by_node_id[key[1]]
                                    ),
                                ),
                                sort_keys=True,
                            ),
                        )

        connection.execute(
            """
            DELETE FROM edges
            WHERE source_node_id IN (SELECT id FROM nodes WHERE target=?)
            """,
            (target,),
        )
        connection.executemany(
            """
            INSERT INTO edges(
                source_node_id, target_node_id, distance,
                mutations_json, context_json
            ) VALUES(?, ?, ?, ?, ?)
            """,
            edge_rows.values(),
        )

        observation_rows = connection.execute(
            """
            SELECT n.id, o.epoch, o.score_kind, o.source, o.score
            FROM nodes n
            JOIN observations o ON o.node_id=n.id
            WHERE n.target=? AND o.epoch IS NOT NULL
            """,
            (target,),
        ).fetchall()
        matched_scores: dict[
            int, dict[tuple[int, str, str], list[float]]
        ] = defaultdict(lambda: defaultdict(list))
        for node_id, epoch, score_kind, source, score in observation_rows:
            matched_scores[int(node_id)][
                (int(epoch), str(score_kind), str(source))
            ].append(float(score))

        graph_material = "\n".join(
            f"{sequence}\t{scores[sequence]:.8f}"
            for sequence in sorted(scores)
        )
        import hashlib

        graph_version = hashlib.sha256(graph_material.encode()).hexdigest()[:16]
        effects: dict[tuple[int, str, str, str, str], list[float]] = defaultdict(list)
        for source_id, target_id, distance, _, context_json in edge_rows.values():
            if distance != 1:
                continue
            context = json.loads(context_json)[0]
            forward = (
                int(context["position"]),
                str(context["from_residue"]),
                str(context["to_residue"]),
                str(context["region"]),
                str(context["contact_class"]),
            )
            reverse = (forward[0], forward[2], forward[1], forward[3], forward[4])
            common = (
                matched_scores[source_id].keys()
                & matched_scores[target_id].keys()
            )
            for comparison in common:
                source_score = float(np.mean(
                    matched_scores[source_id][comparison]
                ))
                target_score = float(np.mean(
                    matched_scores[target_id][comparison]
                ))
                delta = target_score - source_score
                effects[forward].append(delta)
                effects[reverse].append(-delta)
        connection.execute(
            "DELETE FROM mutation_effects WHERE target=? AND graph_version=?",
            (target, graph_version),
        )
        effect_rows = []
        for key, deltas in effects.items():
            median = float(np.median(deltas))
            uncertainty = float(np.median(np.abs(np.asarray(deltas) - median)))
            effect_rows.append((
                target, *key, len(deltas), median, uncertainty, graph_version
            ))
        connection.executemany(
            """
            INSERT INTO mutation_effects(
                target, position, from_residue, to_residue, region,
                contact_class, support, median_score_delta, uncertainty,
                graph_version
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            effect_rows,
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
            (f"graph_version:{target}", graph_version),
        )
    return {
        "target": target,
        "graph_version": graph_version,
        "node_count": node_count,
        "edge_count": len(edge_rows),
        "mutation_effect_count": len(effect_rows),
    }


def ranking_observations(
    ranking: Path,
    *,
    epoch: int,
    target: str,
    source: str,
) -> list[dict[str, Any]]:
    """Convert a local exact ranking into provenance-preserving graph records."""
    payload = json.loads(ranking.read_text())
    observations = []
    for candidate in payload.get("candidates", []):
        sequence = str(candidate.get("sequence", "")).strip().upper()
        score = candidate.get("selection_score")
        if not sequence or score is None:
            continue
        observations.append({
            "observation_id": (
                f"{source}:epoch{epoch}:"
                f"{candidate.get('observation_count', 0)}:{sequence}"
            ),
            "sequence": sequence,
            "score": float(score),
            "score_kind": "local_validator_replay_rank_sum",
            "source": source,
            "epoch": epoch,
            "target": target,
            "observation_count": int(candidate.get("observation_count", 0)),
            "score_median": candidate.get("score_median"),
            "score_p90": candidate.get("score_p90"),
            "metrics": candidate.get("metrics") or candidate.get("metric_medians", {}),
            "pose_stability": candidate.get("pose_stability"),
        })
    return observations
