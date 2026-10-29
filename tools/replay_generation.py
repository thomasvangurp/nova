"""Offline replay of the current nanobody proposal recipe (HEAD a04b203).

Answers one question: does the deployed short-CDR3 recipe (germline CDR2 +
``RAFDY``/``RHFDY`` head for L < 12) fold at the rival 5EHS confidence band
(``design_iiptm`` ~0.42-0.50) or at the old flat-head recipe's 0.16-0.30?

Generation is CPU-only and reproduces ``nanobody_opt.pr100_proposal_pilot.propose``
exactly, including the length split at ``CDR3_TEMPLATE_MIN_LENGTH = 12``.  The
short-regime design space is a small deterministic pool (RA/RH at L=5, R[A/H]X
at L=6), so this script enumerates the full unique population rather than a
flat random draw.  Exact BoltzGen scoring is NOT performed here -- it runs on
the master GPU box through the warm scorer daemon (validator-matched pinned
``fast_score_batch.py``) via ``score_boltzgen_waves.py --scorer
nanobody_opt/score_boltzgen_fast_adapter.py``.  The ``report`` subcommand joins
that scorer's ``exact_all.json`` back to the generated manifest and prints the
per-length comparison against the 5EHS and old-recipe baselines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nanobody_opt.pr100_proposal_pilot import (  # noqa: E402
    CDR3_TEMPLATE_MIN_LENGTH,
    GENES,
    propose,
)
from pr100_germline import real_panel  # noqa: E402

# The 9 V genes the pinned PR100 challenge allows (germline_filtering V1..V4 in
# config/nanobody_pr100.json).  ``GENES`` is imported from the proposal module
# so the two can never drift; this is only asserted for documentation.
ALLOWED_V = tuple(GENES)

#: Lengths replayed.  5 and 6 are the 5EHS short-CDR3 winners (26009-26011);
#: 15 is the long regime the live field is currently at (validator log).
DEFAULT_LENGTHS = (5, 6, 15)

#: Rival 5EHS leaderboard ``design_iiptm`` by CDR3 length, from the short-CDR3
#: epochs: 0.42 (26009, len ~6), 0.44 (26010, len ~5), 0.50 (26011, len ~5).
FIVE_EHS_IIPTM = {
    5: (0.44, 0.50),   # 26010 / 26011
    6: (0.42,),        # 26009
}

#: Our old-recipe short-regime submissions folded at this band before a04b203.
OLD_RECIPE_IIPTM = (0.16, 0.30)


def _manifest_paths(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "sequences.json", out_dir / "manifest.json"


def generate(
    out_dir: Path,
    *,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    seed: int = 0,
    ordinals: int = 128,
) -> dict:
    """Enumerate the full unique candidate population for each requested length.

    Returns the manifest dict that ``report`` consumes.  The scorer input file
    ``sequences.json`` ({"sequences": [str, ...]}) and the richer
    ``manifest.json`` (gene/length/ordinal per sequence) are written side by
    side so the scored output can be joined back unambiguously.
    """
    panel = real_panel()
    sequences_file, manifest_file = _manifest_paths(out_dir)

    seen: set[str] = set()
    rows: list[dict] = []
    for length in lengths:
        for gene in ALLOWED_V:
            for ordinal in range(ordinals):
                try:
                    row = propose(panel, gene, length, seed=seed, ordinal=ordinal)
                except ValueError:
                    # A rare draw can fall under the 82% V-identity gate; the
                    # live generator also skips these and redraws.
                    continue
                sequence = row["sequence"]
                if sequence in seen:
                    continue
                seen.add(sequence)
                rows.append(
                    {
                        "sequence": sequence,
                        "gene": gene,
                        "requested_cdr3_length": length,
                        "seed": seed,
                        "ordinal": ordinal,
                        "v_reference_allele": row["v_reference_allele"],
                        "j_reference_allele": row["j_reference_allele"],
                    }
                )

    manifest = {
        "recipe_sha": "a04b203fbeb8def4d55c8cbd29fa9399c296d260",
        "cdr3_template_min_length": CDR3_TEMPLATE_MIN_LENGTH,
        "protocol": "pr100-conditional-germline-loop-v1",
        "allowed_v": list(ALLOWED_V),
        "lengths": list(lengths),
        "seed": seed,
        "ordinals": ordinals,
        "n_sequences": len(rows),
        "sequences": rows,
    }
    # The scorer accepts {"sequences": [str, ...]} (score_boltzgen_waves._sequences).
    sequences_file.write_text(
        json.dumps({"sequences": [r["sequence"] for r in rows]}, indent=2) + "\n"
    )
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"GENERATED {len(rows)} unique sequences -> {sequences_file} "
        f"(manifest {manifest_file})"
    )
    by_length: dict[int, int] = {}
    for r in rows:
        by_length[r["requested_cdr3_length"]] = (
            by_length.get(r["requested_cdr3_length"], 0) + 1
        )
    for length in sorted(by_length):
        print(f"  length {length}: {by_length[length]} sequences")
    return manifest


def _scored_components(exact_paths: list[Path]) -> tuple[dict, str | None]:
    """Merge per_nanobody_components from one or more exact_all.json files.

    Returns ``(components, target)`` where ``components[sequence] =
    {metric: value}``.  Every shard writes uid 0, so the merge is at the
    SEQUENCE level, mirroring score_boltzgen_batch.py's own merge guard.
    """
    components: dict[str, dict] = {}
    target: str | None = None
    for path in exact_paths:
        payload = json.loads(path.read_text())
        target = payload.get("target", target)
        for _uid, by_sequence in payload.get("per_nanobody_components", {}).items():
            for sequence, by_target in by_sequence.items():
                # by_target maps target -> metrics; keep the metrics of the
                # scored target (there is exactly one per replay).
                for _tgt, metrics in by_target.items():
                    components[sequence] = metrics
    return components, target


def report(
    manifest_path: Path,
    exact_paths: list[Path],
) -> str:
    """Print the per-length iiptm table and return it as text."""
    manifest = json.loads(manifest_path.read_text())
    by_sequence = {r["sequence"]: r for r in manifest["sequences"]}
    components, target = _scored_components(exact_paths)

    lines: list[str] = []
    header = (
        "length | n_gen | n_scored | design_iiptm mean/median/max | "
        "to_target_iptm mean/max | 5EHS iiptm | old-recipe iiptm"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for length in manifest["lengths"]:
        rows = [
            r for r in manifest["sequences"] if r["requested_cdr3_length"] == length
        ]
        scored = [
            (seq, components[seq])
            for r in rows
            if (seq := r["sequence"]) in components
            and "design_iiptm" in components[seq]
        ]
        if not scored:
            lines.append(f"{length} | {len(rows)} | 0 | - | - | - | -")
            continue
        iiptm = [float(m["design_iiptm"]) for _, m in scored]
        to_target = [float(m["design_to_target_iptm"]) for _, m in scored]
        five = FIVE_EHS_IIPTM.get(length)
        five_s = "-".join(f"{v:.2f}" for v in five) if five else "n/a"
        old_s = f"{OLD_RECIPE_IIPTM[0]:.2f}-{OLD_RECIPE_IIPTM[1]:.2f}"
        lines.append(
            f"{length} | {len(rows)} | {len(scored)} | "
            f"{statistics.mean(iiptm):.3f}/{statistics.median(iiptm):.3f}/"
            f"{max(iiptm):.3f} | {statistics.mean(to_target):.3f}/"
            f"{max(to_target):.3f} | {five_s} | {old_s}"
        )
        top3 = sorted(
            scored, key=lambda sm: float(sm[1]["design_iiptm"]), reverse=True
        )[:3]
        for seq, m in top3:
            gene = by_sequence.get(seq, {}).get("gene", "?")
            lines.append(
                f"    top {gene} iiptm={float(m['design_iiptm']):.3f} "
                f"to_target={float(m['design_to_target_iptm']):.3f} {seq}"
            )
    text = "\n".join(lines)
    print(text)
    return text


def selftest() -> None:
    """Cheap invariant checks that the short/long split is what HEAD ships."""
    panel = real_panel()
    short5 = {propose(panel, g, 5, seed=9, ordinal=0)["sequence"] for g in GENES}
    assert len(short5) == 9, "one short-5 emission per gene"
    for seq in short5:
        # CDR3 must be RA/RH + FDY (germline CDR2 kept, no charged motif).
        assert "FDY" in seq
        assert "DISSQGDQTD" not in seq
    long15 = {propose(panel, g, 15, seed=9, ordinal=0)["sequence"] for g in GENES}
    for seq in long15:
        assert "DISSQGDQTD" in seq
    print("SELFTEST PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="enumerate the full unique population")
    gen.add_argument("--out-dir", type=Path, default=Path("replay_out"))
    gen.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--ordinals", type=int, default=128)

    rep = sub.add_parser("report", help="join scores to the manifest and print")
    rep.add_argument("--manifest", type=Path, required=True)
    rep.add_argument("--scores", type=Path, nargs="+", required=True)

    parser.add_argument("--selftest", action="store_true")

    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.command == "generate":
        generate(args.out_dir, lengths=tuple(args.lengths), seed=args.seed,
                 ordinals=args.ordinals)
        return 0
    if args.command == "report":
        report(args.manifest, args.scores)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
