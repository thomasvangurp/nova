from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Iterable

from .types import Candidate, ScoredCandidate


SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    name TEXT PRIMARY KEY,
    reaction INTEGER NOT NULL,
    components TEXT NOT NULL,
    smiles TEXT NOT NULL,
    inchikey TEXT NOT NULL,
    heavy_atoms INTEGER NOT NULL,
    rotatable_bonds INTEGER NOT NULL,
    proxy_mean REAL NOT NULL,
    proxy_std REAL NOT NULL,
    affinity_probability_binary REAL,
    affinity_pred_value REAL,
    score REAL,
    scorer TEXT,
    elapsed_seconds REAL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS candidates_reaction_score ON candidates(reaction, score DESC);
CREATE UNIQUE INDEX IF NOT EXISTS candidates_inchikey ON candidates(inchikey);
"""


class CandidateArchive:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.executescript(SCHEMA)

    def add_candidates(self, candidates: Iterable[Candidate]) -> int:
        rows = [
            (
                item.name, item.reaction, json.dumps(item.components), item.smiles, item.inchikey,
                item.heavy_atoms, item.rotatable_bonds, item.proxy_mean, item.proxy_std, time.time(),
            )
            for item in candidates
        ]
        with sqlite3.connect(self.path) as connection:
            connection.executemany(
                """INSERT OR IGNORE INTO candidates
                   (name,reaction,components,smiles,inchikey,heavy_atoms,rotatable_bonds,proxy_mean,proxy_std,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def add_scores(self, scores: Iterable[ScoredCandidate]) -> int:
        rows = [
            (
                item.affinity_probability_binary, item.affinity_pred_value, item.score,
                item.scorer, item.elapsed_seconds, time.time(), item.candidate.name,
            )
            for item in scores
        ]
        with sqlite3.connect(self.path) as connection:
            connection.executemany(
                """UPDATE candidates SET affinity_probability_binary=?, affinity_pred_value=?, score=?,
                   scorer=?, elapsed_seconds=?, updated_at=? WHERE name=?""",
                rows,
            )
        return len(rows)

    def top(self, reaction: int, limit: int = 500, scored_only: bool = True) -> list[ScoredCandidate]:
        predicate = "AND score IS NOT NULL" if scored_only else ""
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                f"""SELECT name,smiles,reaction,components,inchikey,heavy_atoms,rotatable_bonds,
                    proxy_mean,proxy_std,affinity_probability_binary,affinity_pred_value,score,scorer,elapsed_seconds
                    FROM candidates WHERE reaction=? {predicate}
                    ORDER BY COALESCE(score, proxy_mean) DESC LIMIT ?""",
                (reaction, limit),
            ).fetchall()
        results = []
        for row in rows:
            candidate = Candidate(row[0], row[1], row[2], tuple(json.loads(row[3])), row[4], row[5], row[6], row[7], row[8])
            results.append(ScoredCandidate(candidate, row[9] or 0.0, row[10] or 0.0, row[11] if row[11] is not None else row[7], row[12] or "proxy", row[13] or 0.0))
        return results

    @staticmethod
    def atomic_write_result(path: str | Path, scores: Iterable[ScoredCandidate]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": time.time(), "molecules": [item.as_dict() for item in scores]}
        fd, temporary_name = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
