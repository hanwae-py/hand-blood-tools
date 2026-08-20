"""Expose the bundled algorithm source to compatibility tests."""
from pathlib import Path
import sys

ALGORITHM_SOURCE = Path(__file__).resolve().parents[4] / "algorithm" / "src"
if str(ALGORITHM_SOURCE) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_SOURCE))
