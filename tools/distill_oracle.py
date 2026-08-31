#!/usr/bin/env python3
"""Train and operate the diffusion-free multi-target Boltz student."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compound_miner.distilled import (
    cache_esm2_embeddings,
    evaluate_cascade,
    evaluate_distilled,
    examples_from_archive,
    examples_from_bindingdb_tsv,
    examples_from_boltz_json,
    examples_from_public_csv,
    fetch_chembl_target,
    prepare_bindingdb_balanced,
    select_for_boltz,
    train_distilled_ensemble,
)
from compound_miner.targets import TARGETS


def _deterministic_public_limit(examples, limit):
    if not limit or len(examples) <= limit:
        return examples
    return sorted(
        examples,
        key=lambda value: hashlib.sha256(f"{value.target}\0{value.smiles}".encode()).digest(),
    )[:limit]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch-chembl", help="Fetch public affinity labels for one UniProt accession")
    fetch.add_argument("--accession", required=True)
    fetch.add_argument("--output", type=Path, required=True)

    prepare = commands.add_parser("prepare-bindingdb", help="Build a deterministic target-balanced BindingDB subset")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--maximum-per-target", type=int, default=64)
    prepare.add_argument("--minimum-per-target", type=int, default=8)

    embeddings = commands.add_parser("cache-esm", help="Cache ESM-2 embeddings; ESM is not needed at inference")
    embeddings.add_argument("--target", action="append", default=[], metavar="ACCESSION=SEQUENCE")
    embeddings.add_argument("--target-json", type=Path, help="JSON object mapping accession to sequence")
    embeddings.add_argument("--public-csv", action="append", type=Path, default=[])
    embeddings.add_argument("--public-limit", type=int)
    embeddings.add_argument("--output", type=Path, required=True)
    embeddings.add_argument("--device")
    embeddings.add_argument("--batch-size", type=int, default=8)

    train = commands.add_parser("train", help="Pretrain a general base and calibrate a competition adapter")
    train.add_argument("--public-csv", action="append", type=Path, default=[])
    train.add_argument("--public-limit", type=int, help="Deterministic broad subset for staged pretraining")
    train.add_argument("--bindingdb-tsv", action="append", type=Path, default=[])
    train.add_argument("--bindingdb-limit", type=int)
    train.add_argument("--archive", type=Path)
    train.add_argument("--boltz-json", action="append", type=Path, default=[])
    train.add_argument("--target", default="P40261")
    train.add_argument("--esm-embeddings", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--ensemble-size", type=int, default=5)
    train.add_argument("--pretrain-epochs", type=int, default=30)
    train.add_argument("--finetune-epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--hidden", type=int, default=192)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--device")
    train.add_argument("--workers", type=int, default=0)

    evaluate = commands.add_parser("evaluate", help="Evaluate held-out labelled molecules")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--archive", type=Path)
    evaluate.add_argument("--boltz-json", action="append", type=Path, default=[])
    evaluate.add_argument("--target", default="P40261")
    evaluate.add_argument("--top-k", type=int, default=20)
    evaluate.add_argument("--device")

    cascade = commands.add_parser("cascade-evaluate", help="Evaluate final decision quality under a bounded Boltz budget")
    cascade.add_argument("--model", type=Path, required=True)
    cascade.add_argument("--archive", type=Path)
    cascade.add_argument("--boltz-json", action="append", type=Path, default=[])
    cascade.add_argument("--target", default="P40261")
    cascade.add_argument("--reaction", type=int, choices=range(0, 6), default=0)
    cascade.add_argument("--fraction", type=float, default=0.01)
    cascade.add_argument("--minimum", type=int, default=100)
    cascade.add_argument("--uncertainty-fraction", type=float, default=0.25)
    cascade.add_argument("--submission-count", type=int, default=20)
    cascade.add_argument("--device")

    select = commands.add_parser("select", help="Select the small uncertain/promising fraction escalated to Boltz")
    select.add_argument("--model", type=Path, required=True)
    select.add_argument("--archive", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--target", default="P40261")
    select.add_argument("--fraction", type=float, default=0.01)
    select.add_argument("--minimum", type=int, default=64)
    select.add_argument("--maximum", type=int, default=1024)
    select.add_argument("--uncertainty-fraction", type=float, default=0.25)
    select.add_argument("--device")
    return root


def _objective_examples(args: argparse.Namespace):
    merged = {}
    if args.archive:
        for value in examples_from_archive(args.archive, args.target):
            merged[value.name] = value
    for value in examples_from_boltz_json(args.boltz_json, args.target):
        previous = merged.get(value.name)
        if previous is None or value.weight >= previous.weight:
            merged[value.name] = value
    return list(merged.values())


def main() -> None:
    args = parser().parse_args()
    if args.command == "fetch-chembl":
        print(json.dumps(fetch_chembl_target(args.accession, args.output), indent=2))
        return
    if args.command == "prepare-bindingdb":
        print(json.dumps(prepare_bindingdb_balanced(
            args.input, args.output, args.maximum_per_target, args.minimum_per_target,
        ), indent=2))
        return
    if args.command == "cache-esm":
        targets = dict(TARGETS)
        if args.target_json:
            targets.update(json.loads(args.target_json.read_text()))
        public = []
        for path in args.public_csv:
            public.extend(examples_from_public_csv(path))
        for example in _deterministic_public_limit(public, args.public_limit):
            if example.protein_sequence:
                targets[example.target] = example.protein_sequence
        for value in args.target:
            accession, sequence = value.split("=", 1)
            targets[accession] = sequence
        output = cache_esm2_embeddings(targets, args.output, args.device, args.batch_size)
        print(json.dumps({"targets": len(targets), "output": str(output.resolve())}))
        return
    if args.command == "train":
        public = []
        for path in args.public_csv:
            public.extend(examples_from_public_csv(path, args.target))
        for path in args.bindingdb_tsv:
            public.extend(examples_from_bindingdb_tsv(path, args.bindingdb_limit))
        public = _deterministic_public_limit(public, args.public_limit)
        objectives = _objective_examples(args)
        metrics = train_distilled_ensemble(
            public, objectives, args.output, args.ensemble_size, args.pretrain_epochs,
            args.finetune_epochs, args.batch_size, args.hidden, args.layers, args.device,
            args.esm_embeddings,
            args.workers,
        )
        print(json.dumps(metrics, indent=2))
        return
    if args.command == "evaluate":
        print(json.dumps(evaluate_distilled(args.model, _objective_examples(args), args.top_k, device=args.device), indent=2))
        return
    if args.command == "cascade-evaluate":
        examples = _objective_examples(args)
        reactions = sorted({value.reaction for value in examples}) if args.reaction == 0 else [args.reaction]
        metrics = {}
        for reaction in reactions:
            subset = [value for value in examples if value.reaction == reaction]
            if len(subset) >= args.submission_count:
                metrics[str(reaction)] = evaluate_cascade(
                    args.model, subset, args.fraction, args.minimum, args.uncertainty_fraction,
                    args.submission_count, device=args.device,
                )
        print(json.dumps(metrics, indent=2))
        return
    if args.command == "select":
        print(json.dumps(select_for_boltz(
            args.model, args.archive, args.output, args.target, args.fraction,
            args.minimum, args.maximum, args.uncertainty_fraction, args.device,
        ), indent=2))


if __name__ == "__main__":
    main()
