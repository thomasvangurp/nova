"""Refresh dashboard references and export with one pinned release on master."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
import shlex


def export_command(*, replay: bool = False) -> str:
    """Select an immutable release explicitly; never import the legacy /root tree."""
    root = os.environ.get("NOVA_DASH_MASTER_RELEASE_DIR", "")
    if (not re.fullmatch(r"/root/(?:dashboard-)?releases/[0-9a-f]{40}", root)):
        raise ValueError("dashboard_master_release_required")
    timeout = 1100 if replay else 240
    module = "replay_refresh" if replay else "master_refresh"
    return (f"cd {shlex.quote(root)} && NOVA_LOCAL_SQLITE=1 "
            f"PYTHONPATH={shlex.quote(root)} timeout --kill-after=10 {timeout} "
            f"/root/.pixi/envs/bgen/bin/python -m nanobody_dash.{module}")


def refresh(*, epoch: int, database: Path, output: Path) -> dict:
    from nanobody_opt.rules import ACTIVATION_EPOCH, target_for_epoch
    from nanobody_opt.seed_mutation_database import fetch_observations
    from nanobody_opt.mutation_network import update_observation_store
    from tools.build_multipass_comparison import build

    # The public API may lag the chain. Sync only recent, completed epochs of
    # this target; replay means still require independent stored multipass data.
    first = max(ACTIVATION_EPOCH, epoch - 20)
    observations = fetch_observations(
        range(first, epoch), target=target_for_epoch(epoch),
        api_root="https://compound-api.metanova-labs.ai", workers=4)
    written = update_observation_store(database, observations) if observations else 0
    exported = build(output, lo=None, hi=epoch, recent=6)
    newest = max((int(k) for k in exported if str(k).isdecimal()), default=None)
    return {"ok": True, "chain_epoch": epoch, "reference_rows": written,
            "multipass_epoch": newest}


def main() -> int:
    # Imported collectors can emit API/DB exception text. Keep their streams
    # private and return only fixed operational fields to the scheduled job.
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            epoch = int(json.loads(Path("/root/artifacts/chain_state.json").read_text())["epoch"])
            result = refresh(epoch=epoch,
                             database=Path("/root/artifacts/nanobody_mutations.sqlite"),
                             output=Path("/root/artifacts/multipass_comparison.json"))
    except Exception as error:
        result = {"ok": False, "reason": type(error).__name__}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
