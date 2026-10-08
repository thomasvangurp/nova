#!/usr/bin/env python3
"""Diagnostic: how far are recent winners from the current top-50, and do the
2 sequences at 7 edits score competitively?"""
import json
import urllib.request

import numpy as np

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"

idx = json.load(open("/root/artifacts/top50_cdr3_index.json"))
T = np.array([[ord(c) for c in s] for s in idx["sequences"] if len(s) == 127], dtype=np.uint8)


def dmin(seq: str) -> int:
    a = np.frombuffer(seq.encode(), dtype=np.uint8)
    return int((T != a).sum(axis=1).min())


for ep in [25453, 25454, 25455]:
    try:
        p = json.load(urllib.request.urlopen(
            f"{API}/api/competitions/leaderboard/{ep}/nanobodies", timeout=30))
    except Exception as e:
        print(f"epoch {ep}: err {e}")
        continue
    data = sorted([s for s in (p.get("data") or []) if s.get("final_score") is not None],
                  key=lambda x: x["final_score"])
    print(f"epoch {ep}: field={len(data)}")
    for s in data[:3]:
        for n in s.get("nanobodies") or []:
            if n.get("protein_name") == TARGET:
                print(f"   score {int(s['final_score'])}  min-edit-to-top50={dmin(n['sequence'].upper())}")
                break

# the 2 sequences at 7 edits: what are their final_scores?
print("\n7-edit sequences and their scores:")
for ep in range(25455, 25455 - 14, -1):
    try:
        p = json.load(urllib.request.urlopen(
            f"{API}/api/competitions/leaderboard/{ep}/nanobodies", timeout=30))
    except Exception:
        continue
    for s in p.get("data") or []:
        for n in s.get("nanobodies") or []:
            if n.get("protein_name") == TARGET:
                seq = str(n.get("sequence") or "").strip().upper()
                if len(seq) == 127 and dmin(seq) >= 7:
                    print(f"   epoch {ep} score {s.get('final_score')} edits={dmin(seq)}")
