#!/bin/bash
# Wait for the main sweep (18 runs), then run exploratory experiment E.
cd /c/Users/Timur/Documents/listen
until [ "$(ls runs/f*_s*/episodes.json 2>/dev/null | wc -l)" -ge 18 ]; do sleep 30; done
python scripts/build_pool.py --few
python scripts/run_sweep.py --regime few --fractions 0 0.25 --seeds 0 1 2
