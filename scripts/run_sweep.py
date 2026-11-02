"""Run the full pre-registered sweep: every f, every seed, train + evaluate. Resumable.

    python scripts/run_sweep.py --seeds 0 1 2

Seed-major order, so if it stops early every condition has the same number of runs.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from listen.design import FRACTIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    t0 = time.perf_counter()
    for seed in args.seeds:
        for f in FRACTIONS:
            run = ROOT / "runs" / f"f{f:g}_s{seed}"
            if not (run / "policy.pt").exists():
                subprocess.run([sys.executable, "scripts/train.py", "--f", str(f), "--seed", str(seed)],
                               cwd=ROOT, check=True)
            if not (run / "episodes.json").exists():
                subprocess.run([sys.executable, "scripts/evaluate.py", str(run)], cwd=ROOT, check=True)
            print(f"== done f={f:g} seed={seed} after {(time.perf_counter() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
