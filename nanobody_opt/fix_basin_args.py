#!/usr/bin/env python3
"""Fix the producer launch argv: basin-fraction 0.7 -> 0.35, exploit-basin-fraction
0.95 -> 0.5 (the convergence revert, applied to the NUL-separated argv)."""
P = "/root/artifacts/producer_argv.txt"
data = open(P, "rb").read()
tokens = data.split(b"\0")
changed = []
for i, t in enumerate(tokens):
    if t == b"--basin-fraction" and i + 1 < len(tokens):
        changed.append(("basin-fraction", tokens[i + 1].decode(), "0.35"))
        tokens[i + 1] = b"0.35"
    elif t == b"--exploit-basin-fraction" and i + 1 < len(tokens):
        changed.append(("exploit-basin-fraction", tokens[i + 1].decode(), "0.5"))
        tokens[i + 1] = b"0.5"
open(P, "wb").write(b"\0".join(tokens))
print("changed:", changed)
