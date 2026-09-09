"""Build a BoltzGen nanobody-design specification from the live NOVA target.

The scorer wrapper accepts an already chosen sequence.  This utility builds the
different input needed by BoltzGen's generative pipeline: the current target
structure and a set of VHH scaffold templates whose loops may be redesigned.
Challenge identity and structure values are always read from the fetched
validator config.  The epitope can optionally come from a target-matched,
machine-generated structural-teacher summary; it is never pinned in source.
"""
from __future__ import annotations

import argparse
import json
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
    binding_site: str | None = None,
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
    configured_binding_site = _required(challenge, "structure_binding_site")
    resolved_binding_site = (
        binding_site.strip()
        if isinstance(binding_site, str) and binding_site.strip()
        else configured_binding_site
    )
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
                                "binding": resolved_binding_site,
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


def binding_site_from_summary(
    config: dict[str, Any], summary: dict[str, Any]
) -> str:
    """Validate a teacher summary against the live challenge and return its site."""
    challenge = config["protein_selection"]["nanobody"]
    for name in ("target", "structure"):
        if str(summary.get(name, "")) != str(challenge.get(name, "")):
            raise ValueError(
                f"interface summary {name} {summary.get(name)!r} does not match "
                f"live challenge {challenge.get(name)!r}"
            )
    site = summary.get("recommended_binding_site")
    if not isinstance(site, str) or not site.strip():
        raise ValueError("interface summary lacks recommended_binding_site")
    if int(summary.get("teacher_count", 0)) < 3:
        raise ValueError("interface summary needs at least three teachers")
    return site.strip()


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
    parser.add_argument(
        "--interface-summary",
        type=Path,
        help="target-matched consensus contacts from public structural teachers",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    binding_site = None
    binding_source = "validator-config"
    if args.interface_summary:
        summary = json.loads(args.interface_summary.read_text())
        binding_site = binding_site_from_summary(config, summary)
        binding_source = "structural-teacher-consensus"
    target, spec = build_spec(
        config,
        structure_dir=args.structure_dir,
        msa_dir=args.msa_dir,
        scaffold_paths=args.scaffold,
        binding_site=binding_site,
    )
    _atomic_yaml(args.output, spec)
    print(
        f"DESIGN_SPEC_READY target={target} scaffolds={len(args.scaffold)} "
        f"binding_source={binding_source} output={args.output}"
    )


if __name__ == "__main__":
    main()
