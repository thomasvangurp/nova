# SN68 release checklist

1. Commit the intended runtime, launch contract, and regression tests. Use a clean checkout of that commit.
2. Require the GitHub **SN68 release checks** job to pass for that exact SHA. No skipped suites or informational-only tests count.
3. Download `sn68-tested-release` from that run. Verify its manifest, dependency lock, source hashes, test receipt, and archive inventory.
4. Install those verified bytes. Include the v2 producer tree and run `check_launch_contract.py` before activation. Check installed hashes before stopping existing services.
5. Verify the new processes use the tested code, old epoch process trees are absent, and the current artifact and confirmation are advancing.
6. Retain the commit, CI run, deployment receipt, and observed result. A directory name, tmux session, or previous epoch's artifact is not proof of a successful current submission.

See [the selection and deadline lessons](PARENT_SELECTION_LESSONS_20261022.md).
