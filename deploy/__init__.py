"""CPU deployment validation for the current G1 AdaPT experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "src/rsl_rl_lib"):
  if str(directory) not in sys.path:
    sys.path.insert(0, str(directory))
