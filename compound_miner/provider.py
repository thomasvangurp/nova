from __future__ import annotations

import json
import os
from pathlib import Path

from .types import parse_reaction_name
from .release import validate_release_payload


def result_path(reaction: int) -> Path:
    template = os.environ.get("NOVA_CANDIDATE_RESULT", "artifacts/result_rxn{reaction}.json")
    path = Path(template.format(reaction=reaction)).expanduser()
    if not path.is_absolute():
        path = Path(__file__).parents[1] / path
    return path.resolve()


def load_submission(
    reaction: int,
    count: int = 20,
    target: str | None = None,
    epoch: int | None = None,
) -> str:
    path = result_path(reaction)
    with path.open() as handle:
        payload = json.load(handle)
    validate_release_payload(
        payload,
        reaction,
        count,
        target=target,
        epoch=epoch,
    )
    molecules = payload.get("molecules", [])
    if len(molecules) < count:
        raise ValueError(f"{path} contains {len(molecules)} candidates; need {count}")
    names: list[str] = []
    for item in molecules[:count]:
        name = str(item["name"])
        actual_reaction, _ = parse_reaction_name(name)
        if actual_reaction != reaction:
            raise ValueError(f"{name} is reaction {actual_reaction}, expected {reaction}")
        names.append(name)
    if len(set(names)) != count:
        raise ValueError(f"{path} contains duplicate identifiers")
    return ",".join(names)


def from_challenge(challenge_params: dict) -> str:
    allowed = str(challenge_params["allowed_reaction"])
    if not allowed.startswith("rxn:"):
        raise ValueError(f"invalid allowed reaction: {allowed}")
    targets = challenge_params.get("small_molecule_target") or []
    if isinstance(targets, str):
        targets = [targets]
    if len(targets) != 1:
        raise ValueError(f"expected one small-molecule target, received {targets!r}")
    epoch = challenge_params.get("epoch")
    return load_submission(
        int(allowed.split(":", 1)[1]),
        target=str(targets[0]),
        epoch=int(epoch) if epoch is not None else None,
    )
