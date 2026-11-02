"""Collect the expert demonstration pool (40 scenes x 3 targets x 30 demos).

    python scripts/build_pool.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from listen.data import build_pool  # noqa: E402

if __name__ == "__main__":
    build_pool(ROOT / "data" / "pool")
