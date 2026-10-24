"""Drive the fast exact scorer through the live scorer contract.

``score_boltzgen_waves`` invokes a scorer with a fixed argument list:
``--sequences --config --target --gpus --shards-per-gpu --batch-size
--work-root --out``, plus ``--structure-dir`` and ``--require-structures`` when
structures must be retained.  The fast scorer spells two of those differently
(``--shard-root`` and ``--capture-structures``) and takes tuning knobs the
contract has no room for.

This adapter absorbs that difference so the pipeline keeps one scorer contract
and a replacement stays one path change away from rollback.  It deliberately
does not reinterpret anything else: the sequences, config, target, GPU list,
shard count and batch size are passed straight through, and the scorer's own
output is written to ``--out`` untouched for the wave validator to check.

Box-specific tuning lives in a JSON file rather than in this repository, since
the reference design and scorer location differ per host.  The file is read
from ``NANOBODY_FAST_SCORER_CONFIG`` or ``/root/artifacts/nanobody_fast_scorer.json``:

    {
      "scorer": "/root/fastscore/release/fast_score_batch.py",
      "python": "/root/fastscore/release/release_python.sh",
      "synth_reference": "/root/fastscore/release/refdesign/<design>.cif",
      "scorer_sha256": "<sha256 of the validated scorer>",
      "fold_sampling_steps": 50,
      "fold_diffusion_samples": 5
    }

``synth_reference`` replaces the design stage with side-chain grafting and is
what makes the scorer fast.  It is only passed when every candidate sequence has
the same length, because grafting requires a reference whose designed chain
matches the candidates; otherwise the adapter omits it and the scorer runs its
diffusion design stage, which is slower but correct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# The live path invokes this as a script, not a module, so the repository root
# is not on sys.path.  Insert it before importing the shared contract.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt.diffusion_sample_selection import (  # noqa: E402
    EXPECTED_DIFFUSION_SAMPLES,
)

try:
    from nanobody_opt import scorer_daemon as _scorer_daemon
except Exception:                       # the adapter must run without it
    _scorer_daemon = None

DEFAULT_CONFIG_PATH = Path("/root/artifacts/nanobody_fast_scorer.json")
DEFAULT_SCORER = Path("/root/fastscore/release/fast_score_batch.py")


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Read box-specific scorer settings, tolerating an absent file."""
    location = path or Path(
        os.environ.get("NANOBODY_FAST_SCORER_CONFIG", DEFAULT_CONFIG_PATH)
    )
    if not location.is_file():
        return {}
    settings = json.loads(location.read_text())
    if not isinstance(settings, dict):
        raise ValueError(f"fast scorer settings must be an object: {location}")
    return settings


def require_pinned_scorer(scorer: Path, expected: str | None) -> None:
    """Refuse a scorer whose contents are not the ones that were validated.

    A scorer under active development is not a scorer: an edit lands in the live
    miner with no deploy, no test and no notice, and the first symptom is every
    shard raising mid-epoch.  Pinning ``scorer_sha256`` in the settings makes the
    live path refuse to run anything but the validated bytes.  Omit it only when
    deliberately running against a tree you expect to change.
    """
    if not expected:
        return
    digest = hashlib.sha256()
    with scorer.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"refusing to run {scorer}: sha256 {actual} does not match the "
            f"pinned {expected}; the validated scorer has been modified"
        )


def _sequence_lengths(path: Path) -> set[int]:
    payload = json.loads(path.read_text())
    items = (
        payload
        if isinstance(payload, list)
        else payload.get("sequences") or payload.get("cases") or []
    )
    lengths = set()
    for item in items:
        sequence = item.get("sequence") if isinstance(item, dict) else item
        if isinstance(sequence, str) and sequence.strip():
            lengths.add(len(sequence.strip()))
    return lengths


def build_command(
    args: argparse.Namespace, settings: dict[str, Any]
) -> list[str]:
    """Translate the contract arguments into the fast scorer's own flags."""
    scorer = Path(settings.get("scorer", DEFAULT_SCORER))
    launcher = settings.get("python")
    command = [str(launcher)] if launcher else [sys.executable]
    command.append(str(scorer))
    command.extend([
        "--sequences", str(args.sequences),
        "--config", str(args.config),
        "--target", args.target,
        "--gpus", args.gpus,
        "--shards-per-gpu", str(args.shards_per_gpu),
        "--batch-size", str(args.batch_size),
        "--shard-root", str(args.work_root),
        "--out", str(args.out),
    ])
    if args.structure_dir is not None:
        command.extend(["--capture-structures", str(args.structure_dir)])
    if args.require_structures:
        command.append("--require-structures")

    lengths = _sequence_lengths(Path(args.sequences))
    if len(lengths) != 1:
        raise SystemExit(
            f"fast scorer wave has mixed candidate lengths {sorted(lengths)}; "
            "partition the admitted candidates by length before dispatch"
        )
    length = next(iter(lengths))
    # A single-length wave still cannot use a reference of ANOTHER length.
    # Reference registrations are validated at installation; an unregistered
    # length uses the pinned scorer's ordinary design stage instead of killing
    # every resident worker with SystemExit inside synthesis.
    references = settings.get("synth_references") or {}
    reference = references.get(f"{args.target}:{length}")
    if (not reference and settings.get("synth_reference_length") == length
            and settings.get("synth_reference_target") == args.target):
        reference = settings.get("synth_reference")
    if reference:
        command.extend(["--synth-reference", str(reference)])
    for key, flag in (
        ("fold_sampling_steps", "--fold-sampling-steps"),
        ("fold_diffusion_samples", "--fold-diffusion-samples"),
        ("design_sampling_steps", "--design-sampling-steps"),
        ("fold_matmul_precision", "--fold-matmul-precision"),
    ):
        if settings.get(key) is not None:
            command.extend([flag, str(settings[key])])
    if getattr(args, "allow_replicates", False):
        if settings.get("worker_argument_schema") != "replicates-v1":
            raise SystemExit("replicate scoring requires the registered replicates-v1 worker")
        command.append("--allow-replicates")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shards-per-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--structure-dir", type=Path)
    parser.add_argument("--require-structures", action="store_true")
    parser.add_argument("--allow-replicates", action="store_true")
    parser.add_argument(
        "--settings",
        type=Path,
        help="box-specific scorer settings; defaults to the documented location",
    )
    args = parser.parse_args()

    settings = load_settings(args.settings)
    samples = settings.get("fold_diffusion_samples")
    if samples is not None and int(samples) != EXPECTED_DIFFUSION_SAMPLES:
        raise SystemExit(
            f"refusing to run with fold_diffusion_samples={samples}: the "
            f"reported metrics are a maximum over samples, and the wave "
            f"validator requires {EXPECTED_DIFFUSION_SAMPLES}"
        )
    require_pinned_scorer(
        Path(settings.get("scorer", DEFAULT_SCORER)),
        settings.get("scorer_sha256"),
    )
    command = build_command(args, settings)
    if "--synth-reference" not in command:
        print("FAST_SCORER_DESIGN_FALLBACK no registered reference for "
              f"target={args.target} lengths={sorted(_sequence_lengths(args.sequences))}",
              flush=True)

    # A warm daemon holds the BoltzGen import, the template library and the
    # analysis pool across waves, so the ~33 s of fixed setup in the measured
    # "32.9 + 1.02*candidates" is paid once for the process instead of once
    # per wave. `command` here is fast_score_batch's own argv, so the daemon
    # runs the same shards, the same chunk composition and the same merge --
    # the metric is unchanged, and a daemon fault falls back to this very
    # subprocess rather than losing the wave.
    if _scorer_daemon is not None and (_scorer_daemon.ENABLED or settings.get("use_daemon") is True):
        try:
            if _scorer_daemon.available():
                job = _scorer_daemon.job_from_command(command)
                job["worker_argument_schema"] = settings.get("worker_argument_schema", "base-v1")
                _scorer_daemon.score(job)
                raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as error:
            print(
                "FAST_SCORER_DAEMON_FALLBACK "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )

    completed = subprocess.run(command)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
