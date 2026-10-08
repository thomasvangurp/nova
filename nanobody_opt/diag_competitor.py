#!/usr/bin/env python3
"""Find 5GhREcsK and dissect its recent winners: parent, parent rank, edits."""
import json
import urllib.request

import numpy as np

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"
LEN = 127


def fetch(epoch):
    p = json.load(urllib.request.urlopen(
        f"{API}/api/competitions/leaderboard/{epoch}/nanobodies", timeout=25))
    rows = []
    for sub in p.get("data") or []:
        sc = sub.get("final_score")
        if sc is None:
            continue
        for nb in sub.get("nanobodies") or []:
            if nb.get("protein_name") == TARGET:
                rows.append((str(nb.get("sequence") or "").strip().upper(),
                             sc, sub.get("coldkey", "")))
    return rows


def main():
    # 1) locate 5GhREcsK
    found = {}
    for ep in range(25455, 25438, -1):
        try:
            rows = fetch(ep)
        except Exception:
            continue
        for s, sc, ck in rows:
            if "5GhREcsK" in ck and len(s) == LEN:
                found.setdefault(ep, []).append((sc, s))
    print("5GhREcsK submissions (epoch -> best scores):")
    for ep in sorted(found):
        best = sorted(found[ep])[0]
        print(f"  epoch {ep}: score={best[0]:.0f} seq={best[1][:28]}")

    # 2) build public pool BEFORE 5GhREcsK's most recent win
    win_epoch = max(found) if found else 25455
    pool = {}
    for ep in range(win_epoch - 1, win_epoch - 1 - 60, -1):
        try:
            for s, sc, _ in fetch(ep):
                if len(s) == LEN:
                    pool[s] = min(pool.get(s, 1e9), sc)
        except Exception:
            continue
    seqs = list(pool)
    A = np.array([[ord(c) for c in s] for s in seqs], dtype=np.uint8)
    ranked = sorted(pool.items(), key=lambda x: x[1])

    # 3) for each 5GhREcsK recent win, find the nearest parent
    print(f"\n=== 5GhREcsK recent wins dissected (parent = nearest prior public seq) ===")
    for ep in sorted(found)[-3:]:
        for sc, s in sorted(found[ep]):
            a = np.frombuffer(s.encode(), dtype=np.uint8)
            d = (A != a).sum(axis=1)
            j = int(d.argmin())
            parent = seqs[j]
            diffs = [(i, parent[i], s[i]) for i in range(LEN) if parent[i] != s[i]]
            prank = next(i for i, (x, _) in enumerate(ranked) if x == parent)
            print(f"  epoch {ep} score={sc:.0f}  parent_score={pool[parent]:.0f} "
                  f"parent_rank={prank}/{len(pool)}  edits={len(diffs)} "
                  f"diffs={diffs[:6]}")


if __name__ == "__main__":
    main()
