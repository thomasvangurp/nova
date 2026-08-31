#!/usr/bin/env python3
"""Freeze public history at an epoch boundary for a leakage-free replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def prepare(history: Path, epoch: int, output_dir: Path) -> dict[str, object]:
    frame = pd.read_csv(history)
    required = {
        "epoch",
        "uid",
        "submission_score",
        "name",
        "smiles",
        "score",
        "reaction",
        "component_a",
        "component_b",
        "component_c",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")

    prior = frame[frame["epoch"] < epoch].copy()
    held_out = frame[frame["epoch"] == epoch].copy()
    if prior.empty:
        raise ValueError(f"no history exists before epoch {epoch}")
    if held_out.empty:
        raise ValueError(f"epoch {epoch} is absent from history")

    reactions = sorted(int(value) for value in held_out["reaction"].dropna().unique())
    if len(reactions) != 1:
        raise ValueError(f"epoch {epoch} has ambiguous reactions: {reactions}")

    submissions = held_out.groupby("uid", as_index=False)["submission_score"].max()
    submissions = submissions.sort_values(
        ["submission_score", "uid"], ascending=[False, True]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "history_before.csv"
    novelty_path = output_dir / "novelty_before.csv"
    metadata_path = output_dir / "metadata.json"
    winner_reference_path = output_dir / "winner_reference.json"
    _atomic_csv(prior, training_path)

    novelty = prior[["smiles"]].dropna().drop_duplicates().rename(
        columns={"smiles": "SMILES"}
    )
    novelty["InChI_Key"] = [
        Chem.MolToInchiKey(molecule) if molecule is not None else None
        for molecule in map(Chem.MolFromSmiles, novelty["SMILES"].astype(str))
    ]
    novelty = novelty.dropna(subset=["InChI_Key"])
    _atomic_csv(novelty, novelty_path)

    winner_uid = int(submissions.iloc[0]["uid"])
    winner_rows = held_out[held_out["uid"] == winner_uid].sort_values(
        ["score", "name"], ascending=[False, True]
    )
    winner_records = []
    for row in winner_rows.itertuples(index=False):
        molecule = Chem.MolFromSmiles(str(row.smiles))
        if molecule is None:
            raise ValueError(f"historical winner contains invalid SMILES: {row.name}")
        components = [int(row.component_a), int(row.component_b)]
        if int(row.component_c) >= 0:
            components.append(int(row.component_c))
        winner_records.append(
            {
                "name": str(row.name),
                "smiles": Chem.MolToSmiles(molecule),
                "reaction": int(row.reaction),
                "components": components,
                "inchikey": Chem.MolToInchiKey(molecule),
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "rotatable_bonds": int(Descriptors.NumRotatableBonds(molecule)),
                "proxy_mean": 0.0,
                "proxy_std": 0.0,
                "score": float(row.score),
            }
        )
    winner_payload = {
        "epoch": epoch,
        "uid": winner_uid,
        "archived_submission_score": float(submissions.iloc[0]["submission_score"]),
        "scores": winner_records,
    }
    temporary = winner_reference_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(winner_payload, indent=2) + "\n")
    temporary.replace(winner_reference_path)

    metadata: dict[str, object] = {
        "epoch": epoch,
        "reaction": reactions[0],
        "training_max_epoch": int(prior["epoch"].max()),
        "training_epochs": int(prior["epoch"].nunique()),
        "training_rows": len(prior),
        "historical_molecules": len(novelty),
        "held_out_rows": len(held_out),
        "held_out_submissions": len(submissions),
        "winner_uid": winner_uid,
        "winner_score": float(submissions.iloc[0]["submission_score"]),
        "runner_up_score": (
            float(submissions.iloc[1]["submission_score"])
            if len(submissions) > 1
            else None
        ),
        "training_history": str(training_path.resolve()),
        "novelty_history": str(novelty_path.resolve()),
        "winner_reference": str(winner_reference_path.resolve()),
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.history, args.epoch, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
