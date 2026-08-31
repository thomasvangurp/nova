from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .archive import CandidateArchive
from .selection import select_submission
from .types import Candidate, ScoredCandidate, parse_reaction_name


RELEASE_SCHEMA_VERSION = 1
VALIDATOR_EXACT_SCORER = (
    "boltz-2.2.1-r3-s100-d1-ar5-as100-ad3-skip0-skipconf0"
)


class ReleaseGateError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenBaseline:
    reaction: int
    epoch: int
    score: float
    uid: int | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_validator_exact_scorer(value: object) -> bool:
    return value == VALIDATOR_EXACT_SCORER


def scored_candidates_from_payload(payload: dict[str, Any]) -> list[ScoredCandidate]:
    records = payload.get("scores")
    if not isinstance(records, list) or not records:
        raise ReleaseGateError("exact score payload has no score records")
    scored: list[ScoredCandidate] = []
    for item in records:
        if not is_validator_exact_scorer(item.get("scorer")):
            raise ReleaseGateError(
                f"non-exact scorer for {item.get('name')}: {item.get('scorer')!r}"
            )
        name = str(item["name"])
        reaction = int(item["reaction"])
        components = tuple(int(value) for value in item["components"])
        name_reaction, name_components = parse_reaction_name(name)
        if (name_reaction, name_components) != (reaction, components):
            raise ReleaseGateError(f"candidate identity fields disagree: {name}")
        candidate = Candidate(
            name,
            str(item["smiles"]),
            reaction,
            components,
            str(item["inchikey"]),
            int(item["heavy_atoms"]),
            int(item["rotatable_bonds"]),
            float(item.get("proxy_mean", 0.0)),
            float(item.get("proxy_std", 0.0)),
        )
        values = (
            float(item["affinity_probability_binary"]),
            float(item["affinity_pred_value"]),
            float(item["score"]),
            float(item.get("elapsed_seconds", 0.0)),
        )
        if not all(math.isfinite(value) for value in values):
            raise ReleaseGateError(f"non-finite exact score record: {candidate.name}")
        scored.append(
            ScoredCandidate(
                candidate,
                values[0],
                values[1],
                values[2],
                VALIDATOR_EXACT_SCORER,
                values[3],
            )
        )
    return scored


def load_frozen_baseline(path: str | Path, reaction: int) -> FrozenBaseline:
    payload = json.loads(Path(path).read_text())
    if int(payload.get("schema_version", 0)) != RELEASE_SCHEMA_VERSION:
        raise ReleaseGateError("unsupported baseline schema")
    item = (payload.get("reactions") or {}).get(str(reaction))
    if not isinstance(item, dict):
        raise ReleaseGateError(f"no frozen baseline for reaction {reaction}")
    return FrozenBaseline(
        reaction=reaction,
        epoch=int(item["epoch"]),
        score=float(item["score"]),
        uid=int(item["uid"]) if item.get("uid") is not None else None,
    )


def release_manifest(
    selected: Iterable[ScoredCandidate],
    reaction: int,
    exact_scores_path: str | Path,
    exact_count: int,
    baseline: FrozenBaseline,
    reduced_scores_path: str | Path | None = None,
) -> dict[str, Any]:
    values = list(selected)
    selection_score = float(sum(item.score for item in values))
    gap = selection_score - baseline.score
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "qualified" if gap >= 0 else "rejected",
        "reaction": reaction,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scorer": VALIDATOR_EXACT_SCORER,
        "exact_count": exact_count,
        "selected_count": len(values),
        "selection_score": selection_score,
        "exact_scores_sha256": sha256_file(exact_scores_path),
        "reduced_scores_sha256": (
            sha256_file(reduced_scores_path) if reduced_scores_path is not None else None
        ),
        "baseline": {
            "epoch": baseline.epoch,
            "uid": baseline.uid,
            "score": baseline.score,
            "gap": gap,
        },
    }


def validate_release_payload(
    payload: dict[str, Any], reaction: int, count: int = 20
) -> None:
    release = payload.get("release")
    if not isinstance(release, dict):
        raise ReleaseGateError("submission checkpoint has no exact release manifest")
    if int(release.get("schema_version", 0)) != RELEASE_SCHEMA_VERSION:
        raise ReleaseGateError("unsupported release manifest schema")
    if release.get("status") != "qualified":
        raise ReleaseGateError(f"release is not qualified: {release.get('status')!r}")
    if int(release.get("reaction", -1)) != reaction:
        raise ReleaseGateError("release reaction does not match challenge")
    if not is_validator_exact_scorer(release.get("scorer")):
        raise ReleaseGateError("release was not produced by validator-matched Boltz")
    molecules = payload.get("molecules")
    if not isinstance(molecules, list) or len(molecules) < count:
        raise ReleaseGateError(f"release contains fewer than {count} molecules")
    selected = molecules[:count]
    if int(release.get("selected_count", -1)) != count:
        raise ReleaseGateError("release selected count does not match submission count")
    if any(not is_validator_exact_scorer(item.get("scorer")) for item in selected):
        raise ReleaseGateError("release contains proxy or reduced-score molecules")
    names = [str(item.get("name")) for item in selected]
    inchikeys = [str(item.get("inchikey")) for item in selected]
    if len(set(names)) != count or len(set(inchikeys)) != count:
        raise ReleaseGateError("release contains duplicate molecules")
    observed = float(sum(float(item["score"]) for item in selected))
    declared = float(release.get("selection_score", float("nan")))
    if not math.isclose(observed, declared, rel_tol=0.0, abs_tol=1e-12):
        raise ReleaseGateError("release selection score does not match its molecules")
    baseline = release.get("baseline") or {}
    if observed < float(baseline.get("score", float("inf"))):
        raise ReleaseGateError("release does not beat its frozen baseline")
    if int(release.get("exact_count", 0)) < count:
        raise ReleaseGateError("release exact shortlist is incomplete")


def promote_exact_checkpoint(
    exact_scores_path: str | Path,
    archive_path: str | Path,
    result_path: str | Path,
    baseline_path: str | Path,
    reaction: int,
    reduced_scores_path: str | Path | None = None,
) -> dict[str, Any]:
    exact_path = Path(exact_scores_path)
    payload = json.loads(exact_path.read_text())
    scored = scored_candidates_from_payload(payload)
    wrong = [item.candidate.name for item in scored if item.candidate.reaction != reaction]
    if wrong:
        raise ReleaseGateError(f"exact payload contains another reaction: {wrong[0]}")
    if len(scored) < 20:
        raise ReleaseGateError(f"need at least 20 exact scores, received {len(scored)}")
    archive = CandidateArchive(archive_path)
    archive.add_candidates(item.candidate for item in scored)
    archive.add_scores(scored)
    # Bind every selected row to this exact score artifact. Older archive rows
    # remain useful parents for later search rounds, but cannot silently enter a
    # release whose manifest hashes only the current score file.
    selected = select_submission(scored)
    baseline = load_frozen_baseline(baseline_path, reaction)
    manifest = release_manifest(
        selected,
        reaction,
        exact_path,
        len(scored),
        baseline,
        reduced_scores_path,
    )
    if manifest["status"] != "qualified":
        raise ReleaseGateError(
            "exact checkpoint misses frozen baseline by "
            f"{manifest['baseline']['gap']:.6f}"
        )
    CandidateArchive.atomic_write_result(result_path, selected, release=manifest)
    return manifest
