"""Build a BoltzGen nanobody-design specification from the live NOVA target.

The scorer wrapper accepts an already chosen sequence.  This utility builds the
different input needed by BoltzGen's generative pipeline: the current target
structure and epitope plus a set of VHH scaffold templates whose loops may be
redesigned.  Challenge values are always read from the fetched validator config;
no protein, structure, chain, residue range, or epitope is pinned here.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SCAFFOLDS = ("7eow", "7xl0", "8coh", "8z8v")


def _required(mapping: dict[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nanobody challenge lacks a nonempty {name!r}")
    return value.strip()


def build_spec(
    config: dict[str, Any],
    *,
    structure_dir: Path,
    msa_dir: Path,
    scaffold_paths: list[Path],
) -> tuple[str, dict[str, Any]]:
    """Return the resolved target name and a BoltzGen design specification."""
    try:
        challenge = config["protein_selection"]["nanobody"]
    except (KeyError, TypeError) as exc:
        raise ValueError("config lacks protein_selection.nanobody") from exc
    if not isinstance(challenge, dict):
        raise ValueError("protein_selection.nanobody must be a mapping")

    target = _required(challenge, "target")
    structure = _required(challenge, "structure")
    chain = _required(challenge, "structure_chain")
    residues = _required(challenge, "structure_res_index")
    binding_site = _required(challenge, "structure_binding_site")
    if not scaffold_paths:
        raise ValueError("at least one VHH scaffold is required")

    spec = {
        "entities": [
            {
                "file": {
                    "path": str(structure_dir / f"{structure}.cif"),
                    "include": [
                        {
                            "chain": {
                                "id": chain,
                                "res_index": residues,
                                "msa": str(msa_dir / f"{target}.a3m"),
                            }
                        }
                    ],
                    "binding_types": [
                        {
                            "chain": {
                                "id": chain,
                                "binding": binding_site,
                            }
                        }
                    ],
                }
            },
            {
                "file": {
                    "path": [str(path) for path in scaffold_paths],
                }
            },
        ]
    }
    return target, spec


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--structure-dir", required=True, type=Path)
    parser.add_argument("--msa-dir", required=True, type=Path)
    parser.add_argument("--scaffold", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    target, spec = build_spec(
        config,
        structure_dir=args.structure_dir,
        msa_dir=args.msa_dir,
        scaffold_paths=args.scaffold,
    )
    _atomic_yaml(args.output, spec)
    print(
        f"DESIGN_SPEC_READY target={target} scaffolds={len(args.scaffold)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
