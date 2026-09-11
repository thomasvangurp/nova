"""Would the public-neighbourhood shell have contained each epoch's actual winner?

For every recent epoch, take the winner's sequence and ask how far it sits from
the nearest sequence published in any STRICTLY EARLIER epoch.  That is the only
information a miner has when proposing, so a distance of one means the winner was
reachable inside the shell we now screen exhaustively.
"""
import glob
import json

import numpy as np

METRIC = "final_score"


def load(field_dir="fields"):
    by_epoch = {}
    for path in sorted(glob.glob(f"{field_dir}/epoch*.json")):
        epoch = int(path.split("epoch")[1].split(".")[0])
        payload = json.load(open(path))
        rows = []
        for submission in payload.get("data", []):
            for nb in submission.get("nanobodies", []):
                if nb.get("protein_name") != "P20809":
                    continue
                sequence = str(nb.get("sequence") or "").strip().upper()
                if len(sequence) == 127:
                    rows.append((sequence, float(nb[METRIC]), submission.get("uid")))
        if rows:
            by_epoch[epoch] = rows
    return by_epoch


def encode(sequences):
    return np.frombuffer("".join(sequences).encode(), dtype=np.uint8).reshape(
        len(sequences), 127
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-dir", default="fields")
    args = parser.parse_args()
    by_epoch = load(args.field_dir)
    epochs = sorted(by_epoch)
    print("epoch  winner  dist_to_prior_public  in_shell  our_rank  our_score")
    hits = 0
    counted = 0
    for epoch in epochs:
        prior = []
        for earlier in epochs:
            if earlier >= epoch:
                break
            prior.extend(sequence for sequence, _, _ in by_epoch[earlier])
        if len(prior) < 50:
            continue
        rows = sorted(by_epoch[epoch], key=lambda r: r[1])
        winner_seq, winner_score, _ = rows[0]
        reference = encode(sorted(set(prior)))
        query = encode([winner_seq])
        distance = int((query[:, None, :] != reference[None, :, :]).sum(2).min())
        ours = [(s, sc) for s, sc, uid in by_epoch[epoch] if uid == 8]
        our_rank = next(
            (i + 1 for i, (s, sc, uid) in enumerate(rows) if uid == 8), None
        )
        counted += 1
        hits += distance <= 1
        print(
            "%5d  %6.0f  %20d  %8s  %8s  %9s"
            % (
                epoch,
                winner_score,
                distance,
                "YES" if distance <= 1 else "no",
                our_rank if our_rank else "-",
                f"{ours[0][1]:.0f}" if ours else "-",
            )
        )
    print()
    print(
        "winners within distance 1 of a previously published sequence: "
        f"{hits}/{counted}"
    )
    print(
        "  this is the ceiling a distance-limited shell can reach: a winner "
        "further out than the cutoff cannot be proposed at all"
    )


if __name__ == "__main__":
    main()
