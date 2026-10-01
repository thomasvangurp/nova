#!/usr/bin/env python3
"""Rebuild the public Metanova label set locally (no credentials, no box).

These are *validator* scores for every molecule on every historical leaderboard
— the real objective, not our NIM acquisition proxy. The endpoint is public and
unauthenticated; the epoch list comes from the frozen challenge file so the
training cutoff is preserved exactly.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from compound_miner.history import _epoch_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", required=True, help="challenge.json with training_epochs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    challenge = json.loads(Path(arguments.challenge).read_text())
    epochs = challenge["training_epochs"]
    cutoff = challenge["maximum_training_epoch"]
    started = time.monotonic()
    rows, failed = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {pool.submit(_epoch_rows, epoch): epoch for epoch in epochs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                rows.extend(future.result())
            except Exception as error:
                failed.append(dict(epoch=futures[future], error_class=type(error).__name__))
            if done % 50 == 0:
                print(json.dumps(dict(event="progress", epochs_done=done, total=len(epochs),
                                      rows=len(rows))), flush=True)
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        print(json.dumps(dict(error="no_rows", failed=len(failed))))
        return 1
    # Preserve the training cutoff: never let a tested epoch leak into history.
    frame = frame[frame["epoch"] <= cutoff]
    if not set(frame["epoch"]).issubset(set(epochs)):
        print(json.dumps(dict(error="epoch_leakage")))
        return 1
    frame = frame.sort_values(["epoch", "uid", "score"], ascending=[True, True, False])
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    print(json.dumps(dict(event="complete", rows=int(len(frame)),
                          epochs=int(frame["epoch"].nunique()),
                          maximum_training_epoch=int(frame["epoch"].max()),
                          reaction_3_rows=int((frame["reaction"] == 3).sum()),
                          failed_epochs=len(failed),
                          seconds=round(time.monotonic() - started, 1),
                          output=str(output)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
