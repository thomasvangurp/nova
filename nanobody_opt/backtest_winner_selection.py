"""Put a past epoch's actual winner through our own selection regime.

The question this answers: when we lose an epoch, is it because the winning
molecule was never in our pool, because our selection preferred a worse one, or
because the winner simply drew well on the day?

Those need different fixes, and the leaderboard cannot tell them apart -- it
reports one draw of the winner and one draw of ours.  Re-scoring the winner
ourselves, several times, against the same field our candidates were ranked in,
separates them:

* the winner measures clearly better than our best candidate
  -> the pool is the problem: it was reachable and we did not propose or keep it
* the winner measures the same as our candidates
  -> selection is sound and the epoch was decided by the draw
* the winner measures worse for us than the validator paid it
  -> our scorer and the validator disagree about that molecule

Run in two steps so the GPU cost is explicit and never accidental:

    # 1. write the sequence file, no GPU
    python -m nanobody_opt.backtest_winner_selection prepare \\
        --epoch 25052 --field-dir /root/artifacts/fields --output /root/winner.json

    # 2. score it with the live wave runner, then analyse
    python -m nanobody_opt.score_boltzgen_waves --sequences /root/winner.json ...
    python -m nanobody_opt.backtest_winner_selection analyse \\
        --epoch 25052 --field-dir /root/artifacts/fields \\
        --winner-exact /root/winner_wave/exact_all.json \\
        --ours /root/live/epoch25052/producer
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from nanobody_opt.rank_exact import leaderboard_field
from nanobody_opt.simulate_round import METRICS, rank_sum_in_field

METRIC_NAMES = list(METRICS)


def _payload(field_dir: Path, epoch: int) -> dict[str, Any]:
    path = field_dir / f"epoch{epoch}.json"
    if not path.is_file():
        raise SystemExit(f"no cached field for epoch {epoch}: {path}")
    return json.loads(path.read_text())


def winner_of(payload: dict[str, Any], target: str) -> tuple[str, float, Any]:
    """The sequence that actually won, with its published score and uid."""
    best: tuple[str, float, Any] | None = None
    for submission in payload.get("data", []):
        for nanobody in submission.get("nanobodies", []):
            if str(nanobody.get("protein_name")) != target:
                continue
            score = float(nanobody["final_score"])
            if best is None or score < best[1]:
                best = (
                    str(nanobody["sequence"]).strip().upper(),
                    score,
                    submission.get("uid"),
                )
    if best is None:
        raise SystemExit("no submission for this target in that epoch")
    return best


def competitor_field(
    payload: dict[str, Any], target: str, exclude_uid: int | None
) -> pd.DataFrame:
    """The field a candidate would be ranked against, minus our own entry."""
    if exclude_uid is not None:
        payload = dict(payload)
        payload["data"] = [
            item
            for item in payload.get("data", [])
            if not (isinstance(item, dict) and item.get("uid") == exclude_uid)
        ]
    return leaderboard_field(payload, target)


def observations_from(paths: list[Path], target: str) -> dict[str, list[dict[str, float]]]:
    """Metric vectors per sequence from exact artifacts."""
    found: dict[str, list[dict[str, float]]] = {}
    for path in paths:
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        components = document.get("per_nanobody_components", {})
        if not isinstance(components, dict):
            continue
        for wave in components.values():
            if not isinstance(wave, dict):
                continue
            for sequence, targets in wave.items():
                record = targets.get(target) if isinstance(targets, dict) else None
                if not isinstance(record, dict):
                    continue
                if all(name in record for name in METRIC_NAMES):
                    found.setdefault(str(sequence).strip().upper(), []).append(
                        {name: float(record[name]) for name in METRIC_NAMES}
                    )
    return found


def score_distribution(
    observations: list[dict[str, float]], field: pd.DataFrame
) -> list[float]:
    return [rank_sum_in_field(record, field) for record in observations]


def _summary(scores: list[float]) -> dict[str, float]:
    return {
        "observations": len(scores),
        "median": statistics.median(scores),
        "best": min(scores),
        "worst": max(scores),
        "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="write the winner's sequence file")
    prepare.add_argument("--epoch", required=True, type=int)
    prepare.add_argument("--field-dir", required=True, type=Path)
    prepare.add_argument("--target", default="P20809")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="how many copies to request; the wave runner deduplicates, so "
        "independent observations come from separate wave directories",
    )

    analyse = sub.add_parser("analyse", help="compare the winner against our pool")
    analyse.add_argument("--epoch", required=True, type=int)
    analyse.add_argument("--field-dir", required=True, type=Path)
    analyse.add_argument("--target", default="P20809")
    analyse.add_argument("--exclude-uid", type=int, default=8)
    analyse.add_argument(
        "--winner-exact", required=True, type=Path, action="append",
        help="exact artifact(s) holding our own observations of the winner",
    )
    analyse.add_argument(
        "--ours", required=True, type=Path,
        help="producer root for that epoch, whose observations we replicated",
    )
    args = parser.parse_args()

    payload = _payload(args.field_dir, args.epoch)
    sequence, published, uid = winner_of(payload, args.target)

    if args.command == "prepare":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"target": args.target, "sequences": [sequence]}) + "\n"
        )
        print(
            f"epoch {args.epoch} winner: uid={uid} published_score={published:.0f}"
        )
        print(f"wrote {args.output}; score it with score_boltzgen_waves")
        return

    field = competitor_field(payload, args.target, args.exclude_uid)
    if len(field) < 10:
        raise SystemExit("competitor field too small to rank against")

    winner_obs = observations_from(list(args.winner_exact), args.target).get(
        sequence, []
    )
    if not winner_obs:
        raise SystemExit("the winner's sequence is not in the supplied artifacts")
    ours = observations_from(
        [Path(p) for p in glob.glob(str(args.ours / "**/exact_all.json"), recursive=True)],
        args.target,
    )
    ours.pop(sequence, None)
    replicated = {
        seq: obs for seq, obs in ours.items() if len(obs) >= 5
    } or ours

    winner = _summary(score_distribution(winner_obs, field))
    print(f"epoch {args.epoch}: winner published {published:.0f} by uid {uid}")
    print(f"  field of {len(field)} competitors (ours excluded)")
    print()
    print("%-34s %5s %8s %8s %7s" % ("candidate", "n", "median", "best", "sd"))
    print(
        "%-34s %5d %8.0f %8.0f %7.1f"
        % ("THE WINNER, measured by us", winner["observations"], winner["median"],
           winner["best"], winner["sd"])
    )
    ranked = sorted(
        (
            (seq, _summary(score_distribution(obs, field)))
            for seq, obs in replicated.items()
        ),
        key=lambda item: item[1]["median"],
    )
    for seq, stats in ranked[:5]:
        print(
            "%-34s %5d %8.0f %8.0f %7.1f"
            % (f"ours {seq[:10]}", stats["observations"], stats["median"],
               stats["best"], stats["sd"])
        )
    if not ranked:
        print("  (no replicated candidates of our own for this epoch)")
        return
    best_ours = ranked[0][1]
    gap = best_ours["median"] - winner["median"]
    print()
    print(f"  our best candidate's median minus the winner's: {gap:+.0f} rank points")
    if gap > 25:
        print("  -> the winner is genuinely better in our hands: a POOL problem")
    elif gap < -25:
        print("  -> our candidate measures better: we lost on the DRAW, or on"
              " publication rather than selection")
    else:
        print("  -> indistinguishable: the epoch was decided by the draw")


if __name__ == "__main__":
    main()
