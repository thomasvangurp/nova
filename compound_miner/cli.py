from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import spearmanr

from .archive import CandidateArchive
from .database import ReactionDatabase
from .history import aggregate_observations, sync_history, sync_novelty
from .proxy import TargetProxy, chronological_evaluation
from .provider import load_submission
from .release import promote_exact_checkpoint
from .scorer import NIMScorer
from .search import SearchEngine
from .selection import select_submission
from .validity import HistoricalIndex, ValidityGate, maccs_entropy
from .types import Candidate, ScoredCandidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NOVA mechanism-0 small-molecule search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync-history")
    sync.add_argument("--output", default="history/P40261_scores.csv")
    sync.add_argument("--workers", type=int, default=12)

    novelty = subparsers.add_parser("sync-novelty")
    novelty.add_argument("--output", default="history/P40261_molecules.csv")

    train = subparsers.add_parser("train")
    train.add_argument("--history", default="history/P40261_scores.csv")
    train.add_argument("--model", default="models/P40261_proxy.joblib")
    train.add_argument("--trees", type=int, default=256)
    train.add_argument("--holdout-epochs", type=int, default=8)
    train.add_argument("--skip-evaluation", action="store_true")

    search = subparsers.add_parser("search")
    search.add_argument("--history", default="history/P40261_scores.csv")
    search.add_argument("--historical-molecules", required=True)
    search.add_argument("--model", default="models/P40261_proxy.joblib")
    search.add_argument("--archive", default="archives/P40261.sqlite")
    search.add_argument("--reaction", type=int, choices=range(1, 6), required=True)
    search.add_argument("--proposals", type=int, default=10000)
    search.add_argument("--exact-shortlist", type=int, default=64)
    search.add_argument("--similarity-shortlist", type=int, default=800)
    search.add_argument("--construction-workers", type=int, default=1)
    search.add_argument("--similarity-workers", type=int, default=1)
    search.add_argument("--result")
    search.add_argument("--nim-url")
    search.add_argument("--nim-concurrency", type=int, default=8)
    search.add_argument("--proxy-only", action="store_true")

    search_all = subparsers.add_parser("search-all")
    search_all.add_argument("--history", default="history/P40261_scores.csv")
    search_all.add_argument("--historical-molecules", default="history/P40261_molecules.csv")
    search_all.add_argument("--model", default="models/P40261_proxy.joblib")
    search_all.add_argument("--archive", default="archives/P40261.sqlite")
    search_all.add_argument("--proposals", type=int, default=10000)
    search_all.add_argument("--exact-shortlist", type=int, default=64)
    search_all.add_argument("--similarity-shortlist", type=int, default=800)
    search_all.add_argument("--construction-workers", type=int, default=1)
    search_all.add_argument("--similarity-workers", type=int, default=1)
    search_all.add_argument("--nim-url")
    search_all.add_argument("--nim-concurrency", type=int, default=8)
    search_all.add_argument("--proxy-only", action="store_true")
    search_all.add_argument("--rounds", type=int, default=1)

    benchmark = subparsers.add_parser("nim-benchmark")
    benchmark.add_argument("--history", default="history/P40261_scores.csv")
    benchmark.add_argument("--count", type=int, default=16)
    benchmark.add_argument("--concurrency", type=int, default=8)
    benchmark.add_argument("--nim-url")

    import_scores = subparsers.add_parser("import-exact")
    import_scores.add_argument("--scores", required=True)
    import_scores.add_argument("--archive", default="archives/P40261.sqlite")
    import_scores.add_argument("--result")

    qualify = subparsers.add_parser("qualify-exact")
    qualify.add_argument("--scores", required=True)
    qualify.add_argument("--reduced-scores")
    qualify.add_argument("--archive", default="archives/P40261.sqlite")
    qualify.add_argument("--result", required=True)
    qualify.add_argument(
        "--baselines", default="config/compound_release_baselines.json"
    )
    qualify.add_argument("--reaction", type=int, choices=range(1, 6), required=True)
    qualify.add_argument("--target", required=True)
    qualify.add_argument("--epoch", type=int, required=True)

    validate = subparsers.add_parser("validate-result")
    validate.add_argument("--result", required=True)
    validate.add_argument("--historical-molecules", required=True)
    validate.add_argument("--reaction", type=int, choices=range(1, 6), required=True)

    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "sync-history":
        frame = sync_history(Path(args.output), workers=args.workers)
        print(json.dumps({"rows": len(frame), "epochs": int(frame["epoch"].nunique()), "output": str(Path(args.output).resolve())}))
        return
    if args.command == "sync-novelty":
        output = sync_novelty(Path(args.output))
        print(json.dumps({"output": str(output.resolve()), "bytes": output.stat().st_size}))
        return
    if args.command == "train":
        frame = pd.read_csv(args.history)
        metrics = (
            {}
            if args.skip_evaluation
            else chronological_evaluation(
                frame,
                holdout_epochs=args.holdout_epochs,
                trees=min(args.trees, 96),
            )
        )
        model = TargetProxy(trees=args.trees).fit(aggregate_observations(frame))
        model.save(args.model)
        print(json.dumps({**metrics, "model": str(Path(args.model).resolve())}, indent=2))
        return
    if args.command == "search":
        history = pd.read_csv(args.history)
        database = ReactionDatabase()
        historical_index = HistoricalIndex(args.historical_molecules)
        scorer = None if args.proxy_only else NIMScorer(url=args.nim_url, concurrency=args.nim_concurrency)
        engine = SearchEngine(database, history, TargetProxy.load(args.model), CandidateArchive(args.archive), historical_index, scorer)
        result = args.result or f"artifacts/result_rxn{args.reaction}.json"
        metrics = engine.run_round(
            args.reaction,
            args.proposals,
            args.exact_shortlist,
            args.similarity_shortlist,
            result,
            args.construction_workers,
            args.similarity_workers,
        )
        print(json.dumps(metrics, indent=2))
        return
    if args.command == "search-all":
        history = pd.read_csv(args.history)
        database = ReactionDatabase()
        historical_index = HistoricalIndex(args.historical_molecules)
        scorer = None if args.proxy_only else NIMScorer(url=args.nim_url, concurrency=args.nim_concurrency)
        engine = SearchEngine(database, history, TargetProxy.load(args.model), CandidateArchive(args.archive), historical_index, scorer)
        measurements = []
        for round_index in range(args.rounds):
            for reaction in range(1, 6):
                result = engine.run_round(
                    reaction, args.proposals, args.exact_shortlist, args.similarity_shortlist,
                    f"artifacts/result_rxn{reaction}.json",
                    args.construction_workers,
                    args.similarity_workers,
                )
                measurements.append({"round": round_index + 1, "reaction": reaction, **result})
                print(json.dumps(measurements[-1]), flush=True)
        return
    if args.command == "nim-benchmark":
        frame = pd.read_csv(args.history).sort_values(["epoch", "score"], ascending=[False, False])
        latest_all = frame[frame["epoch"] == frame["epoch"].max()].drop_duplicates("name").sort_values("score", ascending=False)
        if len(latest_all) <= args.count:
            latest = latest_all
        else:
            import numpy as np

            positions = np.linspace(0, len(latest_all) - 1, args.count, dtype=int)
            latest = latest_all.iloc[positions]
        candidates = []
        expected = []
        for row in latest.itertuples(index=False):
            molecule = Chem.MolFromSmiles(row.smiles)
            components = (int(row.component_a), int(row.component_b))
            if int(row.component_c) >= 0:
                components += (int(row.component_c),)
            candidates.append(Candidate(row.name, row.smiles, int(row.reaction), components, Chem.MolToInchiKey(molecule), molecule.GetNumHeavyAtoms(), int(Descriptors.NumRotatableBonds(molecule))))
            expected.append(float(row.score))
        import time

        started = time.monotonic()
        scores = NIMScorer(url=args.nim_url, concurrency=args.concurrency).score(candidates)
        elapsed = time.monotonic() - started
        observed = [item.score for item in scores]
        print(json.dumps({
            "count": len(scores),
            "concurrency": args.concurrency,
            "wall_seconds": elapsed,
            "molecules_per_second": len(scores) / elapsed,
            "spearman": float(spearmanr(expected, observed).statistic),
            "mean_absolute_difference": float(sum(abs(a - b) for a, b in zip(expected, observed)) / len(scores)),
            "expected": expected,
            "observed": observed,
        }, indent=2))
        return
    if args.command == "import-exact":
        payload = json.loads(Path(args.scores).read_text())
        scored = []
        for item in payload["scores"]:
            candidate = Candidate(
                item["name"], item["smiles"], int(item["reaction"]), tuple(item["components"]),
                item["inchikey"], int(item["heavy_atoms"]), int(item["rotatable_bonds"]),
                float(item["proxy_mean"]), float(item["proxy_std"]),
            )
            scored.append(ScoredCandidate(
                candidate, float(item["affinity_probability_binary"]), float(item["affinity_pred_value"]),
                float(item["score"]), str(item["scorer"]), float(item["elapsed_seconds"]),
            ))
        archive = CandidateArchive(args.archive)
        archive.add_candidates(item.candidate for item in scored)
        archive.add_scores(scored)
        reactions = sorted({item.candidate.reaction for item in scored})
        summaries = []
        for reaction in reactions:
            selected = select_submission(archive.top(reaction, scored_only=True))
            if args.result and len(reactions) > 1:
                raise SystemExit("--result is only valid for a single-reaction score file")
            result = args.result or f"artifacts/result_rxn{reaction}.json"
            CandidateArchive.atomic_write_result(result, selected)
            summaries.append({"reaction": reaction, "imported": sum(item.candidate.reaction == reaction for item in scored), "submission_score": sum(item.score for item in selected), "result": str(Path(result).resolve())})
        print(json.dumps(summaries))
        return
    if args.command == "qualify-exact":
        manifest = promote_exact_checkpoint(
            exact_scores_path=args.scores,
            reduced_scores_path=args.reduced_scores,
            archive_path=args.archive,
            result_path=args.result,
            baseline_path=args.baselines,
            reaction=args.reaction,
            target=args.target,
            epoch=args.epoch,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if args.command == "validate-result":
        database = ReactionDatabase()
        gate = ValidityGate(HistoricalIndex(args.historical_molecules))
        names = [item["name"] for item in json.loads(Path(args.result).read_text())["molecules"]]
        candidates = []
        for name in names:
            parts = tuple(int(value) for value in name.split(":")[1:])
            candidate = database.candidate(parts[0], parts[1:])
            if candidate is None or not gate.accepts(candidate):
                raise SystemExit(f"invalid candidate: {name}")
            candidates.append(candidate)
        if len(candidates) != 20 or len({item.inchikey for item in candidates}) != 20:
            raise SystemExit("result must contain exactly 20 chemically unique candidates")
        entropy = maccs_entropy(candidates)
        if entropy < 0.1:
            raise SystemExit(f"entropy below gate: {entropy}")
        print(json.dumps({"valid": True, "count": len(candidates), "entropy": entropy}))


if __name__ == "__main__":
    main()
