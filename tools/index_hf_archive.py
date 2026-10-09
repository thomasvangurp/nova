#!/usr/bin/env python3
"""Index the validator's exact HF archive top-50 and test 5GhREcsK's submissions.

Downloads Metanova/Submission-Archive/P20809_nanobodies.csv (main revision) with
the read-only HF token, ranks it with the validator's own rank_binders (Algorithm-2),
and reports the Hamming distance of 5GhREcsK's recent submissions to that top-50.
The token is read from /root/.hftoken and never printed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_CSV = "/root/P20809_nanobodies.csv"
GH_SEQS = "/tmp/gh_seqs.txt"
URL = "https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main/P20809_nanobodies.csv"


def main() -> None:
    # 1. Download (token read in-process, never echoed).
    token = Path("/root/.hftoken").read_text().strip()
    if not token:
        raise SystemExit("no HF token at /root/.hftoken")
    if not Path(TARGET_CSV).exists():
        req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        Path(TARGET_CSV).write_bytes(data)
        print(f"downloaded {len(data)} bytes", flush=True)

    # 2. Load + inspect.
    frame = pd.read_csv(TARGET_CSV)
    print(f"HF archive: rows={len(frame)} columns={list(frame.columns)}", flush=True)

    # 3. The validator's own rank_binders (loaded by path, not via utils/__init__).
    source = "/root/utils/minmax_weighted_rank.py"
    spec = importlib.util.spec_from_file_location("_nova_minmax_weighted_rank", source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load rank_binders from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    top = module.rank_binders(frame, k=50, max_liability_violations=50).head(50)
    top_seqs = [str(s).strip().upper() for s in top["sequence"]]
    print(f"HF top-50: n={len(top_seqs)} distinct={len(set(top_seqs))}", flush=True)

    # 4. Compare 5GhREcsK's recent submissions (Hamming distance to HF top-50).
    gh = [l.strip().upper() for l in Path(GH_SEQS).read_text().splitlines() if l.strip()]
    mlen = Counter(len(s) for s in top_seqs).most_common(1)[0][0]
    kept = [s for s in top_seqs if len(s) == mlen]
    top_arr = np.frombuffer("".join(kept).encode(), dtype=np.uint8).reshape(len(kept), mlen)

    dists: Counter[int] = Counter()
    novel = 0
    for s in gh:
        if len(s) != mlen:
            continue
        a = np.frombuffer(s.encode(), dtype=np.uint8)
        d = int((top_arr != a).sum(axis=1).min())
        dists[d] += 1
        if d >= 7:
            novel += 1
    print(f"5GhREcsK vs HF top-50: n={sum(dists.values())} novel(>=7)={novel}", flush=True)
    print(f"distance histogram: {dict(sorted(dists.items()))}", flush=True)

    Path("/tmp/hf_index_result.json").write_text(json.dumps({
        "hf_rows": len(frame),
        "hf_top50_n": len(top_seqs),
        "gh_n": sum(dists.values()),
        "gh_novel_ge7": novel,
        "distance_histogram": {str(k): v for k, v in sorted(dists.items())},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
