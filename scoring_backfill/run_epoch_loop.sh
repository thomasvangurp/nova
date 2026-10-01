#!/bin/bash
# One epoch cycle of the bank-update loop. Run after each epoch close.
cd /Users/tvgdeena/nova-nanobody/scoring_backfill
python3 epoch_loop.py --epochs 3 >> /Users/tvgdeena/nova-nanobody/data/epoch_loop.log 2>&1
