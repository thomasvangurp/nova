#!/usr/bin/env python3
"""Exit 0 = rival coverage complete (nothing left for the rival replay loop),
1 = some recent epoch's leaderboard submissions are still short of PASSES
canonical passes, 2 = could not tell. Reuses collect_slave_multipass_seqs, the
same test the rival loop's own "nothing left to score" uses."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/tools")
import collect_slave_multipass_seqs as c  # noqa: E402

PASSES, LOOKBACK = 10, 20
DONE = Path("/root/artifacts/slave_multipass_all_done")


def main() -> int:
    latest = int(json.load(open("/root/artifacts/chain_state.json"))["epoch"])
    for epoch in range(latest - 1, latest - 1 - LOOKBACK, -1):
        if (DONE / f"{epoch}.done").exists():
            continue
        try:
            board = c.leaderboard_sequences(epoch)
        except Exception as error:  # noqa: BLE001
            print(f"RIVAL_COVERAGE_UNKNOWN epoch={epoch}: {error}")
            return 2
        if board is None:
            continue
        wanted = sorted(s for s in set(board) | set(c.qualified_sequences(epoch))
                        if epoch >= c.ACTIVATION_EPOCH or len(s) == c.SUPPORTED_LENGTH)
        counts = c.pass_counts(wanted, c.target_for_epoch(epoch))
        short = sum(1 for s in wanted if counts.get(s, 0) < PASSES)
        if short:
            print(f"RIVAL_COVERAGE_PENDING epoch={epoch} short={short}/{len(wanted)}")
            return 1
    print("RIVAL_COVERAGE_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
